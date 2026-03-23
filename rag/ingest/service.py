"""
入库编排层：把「磁盘上的单个文件」写成 Chroma 里的一条 collection。

**本文件主要做什么**

1. 用 ``resolve_parser`` + ``extract_pages`` 得到按页（或逻辑页）的纯文本；
2. 用 ``chunk_pages`` 按 ``max_chunk_chars`` 再切成多条 chunk，并带上元数据；
3. 用 ``QwenTextEmbedding`` 对每条 chunk 做向量；
4. 可选先删掉同名 collection，再把向量与文本写入 Chroma；
5. 可选把本次入库清单导出为 JSON（路径见返回值）。

**输入（``run_ingest`` 关键字参数）**

- ``file_path``：已落盘的临时文件或本地路径（调用方负责上传后保存）。
- ``original_filename``：原始文件名，用于后缀推断（``parser=auto``）及元数据里的来源名。
- ``dept_tag`` / ``kb_id``：部门与知识库标识，参与 **collection 命名**（与检索时必须一致）。
- ``parser``：解析器选择，如 ``auto``、``pdf``、``pdf_pypdf`` 等，详见 ``rag.ingest.parsers``。
- ``max_chunk_chars``：解析完成后、每条 chunk 的字符上限（在 ``chunking`` 中切分）。
- ``replace_collection``：为 True 时先 ``delete_collection`` 再写入（整库替换该 kb 下该 collection）。
- ``export_manifest``：是否在 **已计算 embedding** 后、写入 Chroma 前生成 manifest（含 ``embedding_dim``）。
- ``export_chunks_pre_embed``：是否在 **调用嵌入 API 之前** 导出纯文本块 JSON（无向量，便于核对解析/分块）。
- ``extract_pdf_images``：对 **.pdf** 是否用 ``pypdf`` 再抽一层内嵌图，写入 ``web/static/images/``，并在各 chunk 的 ``metadata["image_uri"]`` 中存 **JSON 字符串列表**（路径均为 ``/static/images/...``，无域名）。

**输出（返回值 ``dict``，供 HTTP / 脚本展示）**

- ``ok``：成功为 True。
- ``collection_name``：实际写入的 Chroma collection 名。
- ``parser_requested`` / ``parser_used``：用户选的 vs 解析后的（``auto`` 时会变成 ``pdf``/``txt`` 等）。
- ``extract_engine`` / ``extract_detail`` / ``extract_warnings``：解析引擎与说明、告警（如 deepdoc 回退）。
- ``page_count`` / ``chunk_count``：解析得到的页条目数、最终块数。
- ``ingest_export_path``：manifest 路径（``export_manifest``），未导出则为 None。
- ``ingest_chunks_pre_embed_path``：嵌入前 chunk 导出路径（``export_chunks_pre_embed``），未导出则为 None。
- 另有 ``max_chunk_chars``、``vector_db_dir`` 等辅助字段。

HTTP 入口见 ``main.py`` 的 ``POST /rag/ingest``，内部调用 ``run_ingest``。
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

from rag.utils.ingest_progress import ingest_log, ingest_log_done, ingest_log_start

_log_svc = logging.getLogger(__name__)

from rag.embedding.qwen_embed import QwenTextEmbedding
from rag.ingest.chunking import chunk_pages
from rag.ingest.pdf_embedded_images import attach_image_uris_to_metadatas, extract_pdf_embedded_images
from rag.ingest.parsers import extract_pages, resolve_parser
from rag.retrieval.chroma_client import ChromaRagStore, collection_name
from rag.retrieval.json_export import (
    export_ingest_chunks_pre_embed,
    suggest_ingest_export_path,
    suggest_pre_embed_export_path,
)


def run_ingest(
    *,
    file_path: Path,
    original_filename: str,
    dept_tag: str,
    kb_id: str,
    parser: str = "auto",
    max_chunk_chars: int = 1500,  # 解析后再分块的字符上限
    replace_collection: bool = True,
    export_manifest: bool = True,
    export_chunks_pre_embed: bool = True,
    extract_pdf_images: bool = True,
) -> dict[str, Any]:
    """
    执行完整入库流水线。参数与返回值含义见本模块顶部的模块文档字符串。
    """
    # 1) 解析：auto 时按 original_filename 后缀决定 pdf/txt/docx 等
    resolved = resolve_parser(original_filename, parser)
    ingest_log(
        f"文件={original_filename!r} parser请求={parser!r} → resolved={resolved!r} "
        f"dept={dept_tag!r} kb={kb_id!r}（[ingest] 进度；关闭: RAG_INGEST_PROGRESS=0）"
    )
    t_parse = ingest_log_start(f"解析 extract_pages({resolved!r})，长 PDF+deepdoc 可能极慢")
    pe = extract_pages(file_path, resolved)
    ingest_log_done("解析", t_parse)
    pages = pe.pages
    if not pages:
        raise ValueError(
            "未解析出任何文本（空文件、扫描 PDF 若 deepdoc 未就绪可试 pdf_pypdf 或检查 onnx/模型）"
        )

    # 2) 分块：超长单页会按 max_chars 再切，meta 里含 page、source 等
    ingest_log(f"解析结果: {len(pages)} 页/段, engine={pe.engine!r}, warnings={len(pe.warnings)}")
    t_chunk = ingest_log_start("分块 chunk_pages")
    chunks_meta = chunk_pages(
        pages,
        max_chars=max_chunk_chars,
        source_name=original_filename,
    )
    ingest_log_done("分块", t_chunk)
    for _, meta in chunks_meta:
        meta["parser"] = resolved
        meta["extract_engine"] = pe.engine

    texts = [c[0] for c in chunks_meta]
    metadatas = [c[1] for c in chunks_meta]

    # PDF 内嵌图：与 deepdoc 文本并行，按物理页绑定到 chunk（metadata.page）
    pdf_image_pages = 0
    pdf_image_files = 0
    _env_img = (os.getenv("RAG_EXTRACT_PDF_IMAGES", "true") or "true").strip().lower()
    _allow_img = _env_img in ("1", "true", "yes", "on")
    if (
        extract_pdf_images
        and _allow_img
        and Path(original_filename).suffix.lower() == ".pdf"
    ):
        try:
            t_img = ingest_log_start("PDF 内嵌图提取")
            page_to_uris = extract_pdf_embedded_images(file_path)
            ingest_log_done("PDF 内嵌图提取", t_img)
            pdf_image_pages = len(page_to_uris)
            pdf_image_files = sum(len(v) for v in page_to_uris.values())
            attach_image_uris_to_metadatas(metadatas, page_to_uris)
        except Exception as e:
            # 抽图失败不阻断入库（仅无 image_uri）
            _log_svc.warning("PDF 内嵌图提取失败（已忽略）: %s", e)

    # 3) 向量库文档 id：replace 时用稳定 id 便于同结构重跑；追加模式加 uuid 后缀防撞
    if replace_collection:
        ids = [
            f"p{int(m.get('page', 1))}_i{i}"
            for i, m in enumerate(metadatas)
        ]
    else:
        u = uuid.uuid4().hex[:8]
        ids = [f"p{int(m.get('page', 1))}_i{i}_{u}" for i, m in enumerate(metadatas)]

    col_name = collection_name(dept=dept_tag, kb_id=kb_id)
    pre_embed_path: str | None = None
    if export_chunks_pre_embed:
        ingest_log("写入嵌入前 chunk JSON …")
        pre_embed_path = str(suggest_pre_embed_export_path(col_name))
        export_ingest_chunks_pre_embed(
            collection_name=col_name,
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            original_filename=original_filename,
            parser_used=resolved,
            extract_engine=pe.engine,
            extract_detail=pe.detail,
            extract_warnings=pe.warnings,
            max_chunk_chars=max_chunk_chars,
            export_path=pre_embed_path,
        )

    # 4) 嵌入：调用 DashScope 等（见 qwen_embed 与 .env）
    ingest_log(f"嵌入 API: 共 {len(texts)} 条 chunk（逐条请求，耗时可观）")
    t_emb = ingest_log_start("嵌入 embed_documents")
    emb = QwenTextEmbedding()
    embeddings = emb.embed_documents(texts)
    ingest_log_done("嵌入", t_emb)

    # 5) 目标 collection：dept + kb 隔离，检索时必须同一命名规则
    ingest_log(f"Chroma collection={col_name!r} replace={replace_collection}")
    store = ChromaRagStore()

    if replace_collection:
        try:
            store._client.delete_collection(col_name)
        except Exception:
            # 不存在或首次写入时删除失败可忽略
            pass

    export_path: str | None = None
    if export_manifest:
        ingest_log("写入 ingest manifest JSON …")
        export_path = str(suggest_ingest_export_path(col_name))

    t_ch = ingest_log_start("Chroma 写入")
    store.add(
        collection_name=col_name,
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        export_manifest_path=export_path,
    )
    ingest_log_done("Chroma 写入", t_ch)
    ingest_log(f"全部完成: chunks={len(chunks_meta)} collection={col_name!r}")

    return {
        "ok": True,
        "collection_name": col_name,
        "dept_tag": dept_tag,
        "kb_id": kb_id,
        "parser_requested": parser,
        "parser_used": resolved,
        "extract_engine": pe.engine,
        "extract_detail": pe.detail,
        "extract_warnings": pe.warnings,
        "original_filename": original_filename,
        "page_count": len(pages),
        "chunk_count": len(chunks_meta),
        "max_chunk_chars": max_chunk_chars,
        "max_chunk_chars_hint": "解析后单段若超过该字符数再切分；非 deepdoc 内部 token 逻辑",
        "replace_collection": replace_collection,
        "ingest_export_path": export_path,
        "ingest_chunks_pre_embed_path": pre_embed_path,
        "vector_db_dir": os.getenv("VECTOR_DB_DIR", "vector_db/chroma"),
        "pdf_image_pages": pdf_image_pages,
        "pdf_image_files": pdf_image_files,
    }
