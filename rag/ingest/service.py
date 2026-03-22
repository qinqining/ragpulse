"""上传文件 → 解析 → 分块 → 嵌入 → Chroma。"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from rag.embedding.qwen_embed import QwenTextEmbedding
from rag.ingest.chunking import chunk_pages
from rag.ingest.parsers import extract_pages, resolve_parser
from rag.retrieval.chroma_client import ChromaRagStore, collection_name
from rag.retrieval.json_export import suggest_ingest_export_path


def run_ingest(
    *,
    file_path: Path,
    original_filename: str,
    dept_tag: str,
    kb_id: str,
    parser: str = "auto",
    max_chunk_chars: int = 1500,
    replace_collection: bool = True,
    export_manifest: bool = True,
) -> dict[str, Any]:
    resolved = resolve_parser(original_filename, parser)
    pe = extract_pages(file_path, resolved)
    pages = pe.pages
    if not pages:
        raise ValueError(
            "未解析出任何文本（空文件、扫描 PDF 若 deepdoc 未就绪可试 pdf_pypdf 或检查 onnx/模型）"
        )

    chunks_meta = chunk_pages(
        pages,
        max_chars=max_chunk_chars,
        source_name=original_filename,
    )
    for _, meta in chunks_meta:
        meta["parser"] = resolved
        meta["extract_engine"] = pe.engine

    texts = [c[0] for c in chunks_meta]
    metadatas = [c[1] for c in chunks_meta]
    # 稳定 id：带页码；replace=False 时用 uuid 防冲突
    if replace_collection:
        ids = [
            f"p{int(m.get('page', 1))}_i{i}"
            for i, m in enumerate(metadatas)
        ]
    else:
        u = uuid.uuid4().hex[:8]
        ids = [f"p{int(m.get('page', 1))}_i{i}_{u}" for i, m in enumerate(metadatas)]

    emb = QwenTextEmbedding()
    embeddings = emb.embed_documents(texts)

    col_name = collection_name(dept=dept_tag, kb_id=kb_id)
    store = ChromaRagStore()

    if replace_collection:
        try:
            store._client.delete_collection(col_name)
        except Exception:
            pass

    export_path: str | None = None
    if export_manifest:
        export_path = str(suggest_ingest_export_path(col_name))

    store.add(
        collection_name=col_name,
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        export_manifest_path=export_path,
    )

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
        "vector_db_dir": os.getenv("VECTOR_DB_DIR", "vector_db/chroma"),
    }
