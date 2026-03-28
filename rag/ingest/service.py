"""
入库编排层：把「磁盘上的单个文件」写成 Chroma 里的一条 collection。

**本文件主要做什么**

1. 用 ``resolve_parser`` + ``extract_pages`` 得到按页（或逻辑页）的纯文本；
2. 分块：优先使用各解析器/路由输出的结构化结果，**不再做 token/字符二次切分**。
3. 用 ``QwenTextEmbedding`` 对每条 chunk 做向量；
4. 可选先删掉同名 collection，再把向量与文本写入 Chroma；
5. 可选把本次入库清单导出为 JSON（路径见返回值）。

**输入（``run_ingest`` 关键字参数）**

- ``file_path``：已落盘的临时文件或本地路径（调用方负责上传后保存）。
- ``original_filename``：原始文件名，用于后缀推断（``parser=auto``）及元数据里的来源名。
- ``dept_tag`` / ``kb_id``：部门与知识库标识，参与 **collection 命名**（与检索时必须一致）。
- ``parser``：解析器选择，如 ``auto``、``pdf``、``pdf_pypdf`` 等，详见 ``rag.ingest.parsers``。
- ``replace_collection``：为 True 时先 ``delete_collection`` 再写入（整库替换该 kb 下该 collection）。
- ``export_manifest``：是否在 **已计算 embedding** 后、写入 Chroma 前生成 manifest（含 ``embedding_dim``）。
- ``export_chunks_pre_embed``：是否在 **调用嵌入 API 之前** 导出纯文本块 JSON（无向量，便于核对解析/分块）。
- ``extract_pdf_images``：对 **.pdf** 是否用 ``pypdf`` 再抽一层内嵌图，写入 ``web/static/images/``，并在各 chunk 的 ``metadata["image_uri"]`` 中存 **JSON 字符串列表**（路径均为 ``/static/images/...``，无域名）。
- ``llm_chunk_summary``：为 True 时，在嵌入前对每个 chunk 调 LLM 生成 **增强摘要**；**向量用摘要**，Chroma 仍存原文；``metadata["chunk_summary"]`` 供 ``/rag/qa`` 拼 ``[增强摘要]+[正文]``（chunk 多时会显著变慢，可用 ``RAG_LLM_PRE_SUMMARY_MAX_CHUNKS`` 等限流）。

**输出（返回值 ``dict``，供 HTTP / 脚本展示）**

- ``ok``：成功为 True。
- ``collection_name``：实际写入的 Chroma collection 名。
- ``parser_requested`` / ``parser_used``：用户选的 vs 解析后的（``auto`` 时会变成 ``pdf``/``txt`` 等）。
- ``extract_engine`` / ``extract_detail`` / ``extract_warnings``：解析引擎与说明、告警（如 deepdoc 回退）。
- ``page_count`` / ``chunk_count``：解析得到的页条目数、最终块数。
- ``ingest_export_path``：manifest 路径（``export_manifest``），未导出则为 None。
- ``ingest_chunks_pre_embed_path``：嵌入前 chunk 导出路径（``export_chunks_pre_embed``），未导出则为 None。
- 另有 ``vector_db_dir`` 等辅助字段。

HTTP 入口见 ``main.py`` 的 ``POST /rag/ingest``，内部调用 ``run_ingest``。
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from rag.utils.ingest_progress import ingest_log, ingest_log_done, ingest_log_start

_log_svc = logging.getLogger(__name__)

from rag.embedding.qwen_embed import QwenTextEmbedding
from rag.llm.chat_model import chat_completions
from rag.ingest.pdf_embedded_images import attach_image_uris_to_metadatas, extract_pdf_embedded_images
from rag.ingest.parsers import PageExtract, extract_pages, resolve_parser, _strip_position_tags
from rag.retrieval.chroma_client import ChromaRagStore, collection_name
from rag.retrieval.json_export import (
    export_ingest_chunks_pre_embed,
    suggest_ingest_export_path,
    suggest_pre_embed_export_path,
)

_PDF_DOC_TYPE_CHOICES = {"auto", "paper", "laws", "book", "one"}


def _detect_pdf_doc_type(pages: list[tuple[int, str]]) -> str:
    import re

    sample = "\n".join((t or "")[:400] for _, t in pages[:20]).lower()
    if not sample.strip():
        return "one"
    if re.search(r"\babstract\b|关键词|references|introduction|method|实验|结论", sample):
        return "paper"
    if re.search(r"第[一二三四五六七八九十百千万0-9]+条|第[一二三四五六七八九十百千万0-9]+章|法|条例|规定", sample):
        return "laws"
    if re.search(r"目录|chapter|节|前言", sample):
        return "book"
    if re.search(r"问[:：]|答[:：]|q[:：]|a[:：]|question|answer", sample):
        return "laws"
    return "one"


def _pdf_page_count(path: Path) -> int:
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(path)).pages)
    except Exception:
        return 0


def _paper_callback_proxy(*args: Any, **kwargs: Any) -> None:
    """RAGFlow ``paper.chunk`` 的 ``callback`` 可能是 ``(msg=...)`` 或 ``(progress, msg)``。"""
    msg = kwargs.get("msg")
    if msg is not None:
        ingest_log(f"[paper] {msg}")
        return
    if len(args) >= 2 and isinstance(args[0], (int, float)):
        ingest_log(f"[paper] {args[1]}")
    elif args:
        ingest_log(f"[paper] {args!r}")


def _invoke_ragflow_paper_chunk(
    *,
    file_path: Path,
) -> list[dict[str, Any]]:
    from rag.app.paper import chunk as paper_chunk

    layout = (os.getenv("RAG_PAPER_LAYOUT_RECOGNIZE", "DeepDOC") or "DeepDOC").strip()
    delim = os.getenv("RAG_PAPER_CHUNK_DELIMITER") or "\n!?。；！？"
    parser_config = {
        "chunk_token_num": 0,
        "delimiter": delim,
        "layout_recognize": layout,
        "table_context_size": int(os.getenv("RAG_PAPER_TABLE_CONTEXT", "0") or "0"),
        "image_context_size": int(os.getenv("RAG_PAPER_IMAGE_CONTEXT", "0") or "0"),
    }
    lang = (os.getenv("RAG_PAPER_LANG", "Chinese") or "Chinese").strip()
    fn = str(file_path.resolve())
    return paper_chunk(
        filename=fn,
        binary=None,
        from_page=0,
        to_page=100000,
        lang=lang,
        callback=_paper_callback_proxy,
        parser_config=parser_config,
    )


def _pages_to_chunks_no_split(
    pages: list[tuple[int, str]],
    *,
    source_name: str,
    doc_type: str,
) -> list[tuple[str, dict[str, Any]]]:
    """不做 token/字符二次切分：每页（或每段）直接作为一个 chunk。"""
    out: list[tuple[str, dict[str, Any]]] = []
    part_by_page: dict[int, int] = {}
    for page_num, text in pages:
        txt = _strip_position_tags((text or "")).strip()
        if not txt:
            continue
        try:
            page = int(page_num)
        except (TypeError, ValueError):
            page = 1
        part = part_by_page.get(page, 0)
        part_by_page[page] = part + 1
        out.append((txt, {
            "page": page,
            "part": part,
            "source": source_name,
            "doc_type": doc_type,
            "chunk_route": "no_split",
        }))
    return out


def _paper_ragflow_docs_to_chunks(
    raw: list[dict[str, Any]],
    *,
    source_name: str,
    doc_type: str,
) -> list[tuple[str, dict[str, Any]]]:
    """将 ``rag.app.paper.chunk`` 返回的 ES 风格 dict 转为 Chroma 可用的 (text, metadata)。"""
    out: list[tuple[str, dict[str, Any]]] = []
    part_by_page: dict[int, int] = {}
    for d in raw:
        if not isinstance(d, dict):
            continue
        txt = d.get("content_with_weight")
        if not isinstance(txt, str):
            txt = ""
        txt = _strip_position_tags(txt).strip()
        if not txt:
            continue
        page = 1
        pni = d.get("page_num_int")
        if isinstance(pni, list) and pni:
            try:
                page = int(pni[0])
            except (TypeError, ValueError):
                page = 1
        elif isinstance(pni, int):
            page = pni
        part = part_by_page.get(page, 0)
        part_by_page[page] = part + 1
        meta: dict[str, Any] = {
            "page": page,
            "part": part,
            "source": source_name,
            "doc_type": doc_type,
            "chunk_route": "rag_app_paper",
        }
        dtk = d.get("doc_type_kwd")
        if dtk is not None:
            meta["doc_type_kwd"] = dtk if isinstance(dtk, str) else str(dtk)
        ik = d.get("important_kwd")
        if ik is not None:
            meta["important_kwd"] = json.dumps(ik, ensure_ascii=False) if not isinstance(ik, str) else ik
        out.append((txt, meta))
    return out


def run_ingest(
    *,
    file_path: Path,
    original_filename: str,
    dept_tag: str,
    kb_id: str,
    parser: str = "auto",
    replace_collection: bool = True,
    export_manifest: bool = True,
    export_chunks_pre_embed: bool = True,
    extract_pdf_images: bool = True,
    pdf_doc_type: str = "auto",
    llm_chunk_summary: bool = False,
) -> dict[str, Any]:
    """
    执行完整入库流水线。参数与返回值含义见本模块顶部的模块文档字符串。
    """
    resolved = resolve_parser(original_filename, parser)
    ingest_log(
        f"文件={original_filename!r} parser请求={parser!r} → resolved={resolved!r} "
        f"dept={dept_tag!r} kb={kb_id!r}（[ingest] 进度；关闭: RAG_INGEST_PROGRESS=0）"
    )

    req_doc_type = (pdf_doc_type or "auto").strip().lower()
    if req_doc_type not in _PDF_DOC_TYPE_CHOICES:
        raise ValueError(f"pdf_doc_type 不支持: {pdf_doc_type!r}，可选 auto/paper/laws/book/one")

    is_pdf = Path(original_filename).suffix.lower() == ".pdf"
    explicit_paper = is_pdf and req_doc_type == "paper"

    pe: PageExtract
    chunks_meta: list[tuple[str, dict[str, Any]]]
    resolved_doc_type = "one"

    # --- PDF + 显式「论文」：优先 RAGFlow rag.app.paper.chunk（跳过 extract_pages）---
    if explicit_paper:
        resolved_doc_type = "paper"
        ingest_log("pdf_doc_type=paper：尝试 rag.app.paper.chunk（跳过 extract_pages）")
        try:
            t_paper = ingest_log_start("RAGFlow paper.chunk")
            raw_docs = _invoke_ragflow_paper_chunk(
                file_path=file_path,
            )
            ingest_log_done("RAGFlow paper.chunk", t_paper)
            chunks_meta = _paper_ragflow_docs_to_chunks(
                raw_docs,
                source_name=original_filename,
                doc_type=resolved_doc_type,
            )
            if not chunks_meta:
                raise ValueError("paper.chunk 未产生任何非空文本块")
            npg = _pdf_page_count(file_path)
            pe = PageExtract(
                pages=[(i, "") for i in range(1, npg + 1)] if npg else [(1, "")],
                engine="rag_app_paper",
                detail="RAGFlow rag.app.paper.chunk；未执行 extract_pages",
                warnings=[],
            )
        except Exception as e:
            _log_svc.warning("rag.app.paper.chunk 失败，回退 extract_pages 原始页段入库: %s", e)
            t_parse = ingest_log_start(f"解析 extract_pages({resolved!r})，长 PDF+deepdoc 可能极慢")
            pe = extract_pages(file_path, resolved)
            ingest_log_done("解析", t_parse)
            pages = pe.pages
            if not pages:
                raise ValueError(
                    "未解析出任何文本（空文件、扫描 PDF 若 deepdoc 未就绪可试 pdf_pypdf 或检查 onnx/模型）"
                )
            ingest_log(f"解析结果: {len(pages)} 页/段, engine={pe.engine!r}, warnings={len(pe.warnings)}")
            chunks_meta = _pages_to_chunks_no_split(
                pages,
                source_name=original_filename,
                doc_type=resolved_doc_type,
            )
    else:
        # 1) 解析：auto 时按 original_filename 后缀决定 pdf/txt/docx 等
        t_parse = ingest_log_start(f"解析 extract_pages({resolved!r})，长 PDF+deepdoc 可能极慢")
        pe = extract_pages(file_path, resolved)
        ingest_log_done("解析", t_parse)
        pages = pe.pages
        if not pages:
            raise ValueError(
                "未解析出任何文本（空文件、扫描 PDF 若 deepdoc 未就绪可试 pdf_pypdf 或检查 onnx/模型）"
            )

        if is_pdf:
            resolved_doc_type = _detect_pdf_doc_type(pages) if req_doc_type == "auto" else req_doc_type
        else:
            # 非 PDF 时 pdf_doc_type 不参与（与旧行为一致）
            resolved_doc_type = "one"

        ingest_log(f"解析结果: {len(pages)} 页/段, engine={pe.engine!r}, warnings={len(pe.warnings)}")

        chunks_meta = []
        if is_pdf and resolved_doc_type == "paper":
            try:
                t_paper = ingest_log_start("RAGFlow paper.chunk（auto 识别为论文）")
                raw_docs = _invoke_ragflow_paper_chunk(
                    file_path=file_path,
                )
                ingest_log_done("RAGFlow paper.chunk（auto 识别为论文）", t_paper)
                chunks_meta = _paper_ragflow_docs_to_chunks(
                    raw_docs,
                    source_name=original_filename,
                    doc_type=resolved_doc_type,
                )
                if chunks_meta:
                    pe.warnings.append(
                        "auto 识别为论文：已用 rag.app.paper.chunk 重新分块（与 extract_pages 并行，解析成本高）"
                    )
                    pe.engine = f"{pe.engine}+rag_app_paper"
                else:
                    raise ValueError("paper.chunk 未产生任何非空文本块")
            except Exception as e:
                _log_svc.warning("rag.app.paper.chunk 失败，回退 chunk_pages: %s", e)
                chunks_meta = []

        if not chunks_meta:
            chunks_meta = _pages_to_chunks_no_split(
                pages,
                source_name=original_filename,
                doc_type=resolved_doc_type,
            )

    for _, meta in chunks_meta:
        meta["parser"] = resolved
        meta["extract_engine"] = pe.engine
        meta["pdf_doc_type"] = resolved_doc_type

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
            max_chunk_chars=0,
            export_path=pre_embed_path,
        )

    # 4) 可选：入库前对每个 chunk 生成「增强检索摘要」
    #   - embedding 使用摘要（embed_texts）
    #   - Chroma documents 仍为原文
    #   - metadatas["chunk_summary"] 供 /rag/qa 拼接上下文
    embed_texts = texts
    if llm_chunk_summary:
        from rag.prompts.generator import chunk_enhanced_summary_prompt

        ingest_log(
            f"LLM 预摘要: chunks={len(texts)}（会显著变慢；可用 RAG_LLM_PRE_SUMMARY_MAX_CHUNKS 限制条数）"
        )
        t_sum = ingest_log_start("LLM 生成 chunk_enhanced_summaries")

        sys_prompt = "你是严谨的知识助手。请严格按用户给出的任务要求输出，不要额外解释。"
        user_prefix = chunk_enhanced_summary_prompt()

        max_chunk_chars_for_llm = int(os.getenv("RAG_LLM_PRE_SUMMARY_MAX_CHUNK_CHARS", "2500") or "2500")
        max_chunks = int(os.getenv("RAG_LLM_PRE_SUMMARY_MAX_CHUNKS", "0") or "0")
        model_summary = (os.getenv("LLM_SUMMARY_MODEL") or os.getenv("LLM_MODEL") or "").strip() or None

        summaries: list[str] = []
        n = len(texts)
        for i, t in enumerate(texts):
            if max_chunks > 0 and i >= max_chunks:
                summaries.append(t)
                continue

            t0 = (t or "").strip()
            if max_chunk_chars_for_llm > 0 and len(t0) > max_chunk_chars_for_llm:
                t0 = t0[:max_chunk_chars_for_llm] + "…"

            if n > 0 and (i == 0 or i == n - 1 or (i + 1) % 5 == 0):
                ingest_log(f"LLM 预摘要进度 {i + 1}/{n}")

            try:
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prefix + t0},
                ]
                summary = chat_completions(messages=messages, model=model_summary)
                summary = (summary or "").strip()
                if not summary:
                    summary = t0
            except Exception as e:
                _log_svc.warning("chunk summary failed: i=%s err=%s（已回退原文）", i, e)
                summary = t0

            metadatas[i]["chunk_summary"] = summary
            summaries.append(summary)

        embed_texts = summaries
        ingest_log_done("LLM 生成 chunk_enhanced_summaries", t_sum)

    ingest_log(f"嵌入 API: 共 {len(embed_texts)} 条（embed_texts 在预摘要开启时为摘要）")
    t_emb = ingest_log_start("嵌入 embed_documents")
    emb = QwenTextEmbedding()
    embeddings = emb.embed_documents(embed_texts)
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
        "page_count": len(pe.pages),
        "chunk_count": len(chunks_meta),
        "pdf_doc_type_requested": req_doc_type,
        "pdf_doc_type_resolved": resolved_doc_type,
        "chunking_hint": "当前已禁用 token/字符二次切分：优先使用 app 解析结果直接入库",
        "llm_chunk_summary": llm_chunk_summary,
        "replace_collection": replace_collection,
        "ingest_export_path": export_path,
        "ingest_chunks_pre_embed_path": pre_embed_path,
        "vector_db_dir": os.getenv("VECTOR_DB_DIR", "vector_db/chroma"),
        "pdf_image_pages": pdf_image_pages,
        "pdf_image_files": pdf_image_files,
    }
