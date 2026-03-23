"""
RAG 检索 / 入库 chunk 的 JSON 落盘，便于人工检查（不启动服务也可用）。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag.utils.verbose import rag_print


def default_export_dir() -> Path:
    """默认导出目录：环境变量 RAG_EXPORT_DIR，相对路径相对仓库根。"""
    root = Path(__file__).resolve().parents[2]
    d = (os.getenv("RAG_EXPORT_DIR") or "data/rag_exports").strip() or "data/rag_exports"
    p = Path(d)
    return p if p.is_absolute() else root / p


def utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_json(data: Any, path: str | Path, *, indent: int = 2) -> Path:
    """原子写入 UTF-8 JSON（先写 .tmp 再 replace）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    tmp.replace(p)
    rag_print(f"save_json -> {p} ({p.stat().st_size} bytes)", tag="rag.export")
    return p


def export_ingest_manifest(
    *,
    collection_name: str,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    embeddings: list[list[float]],
    export_path: str | Path,
    include_embedding_preview: bool = False,
    preview_n: int = 8,
) -> Path:
    """
    写入 Chroma **之前** 导出本批 chunk，便于检查 id / 正文 / 元数据 / 向量维度。

    ``include_embedding_preview=True`` 时仅写入每条向量前 ``preview_n`` 维（避免 JSON 过大）。
    """
    chunks: list[dict[str, Any]] = []
    for i, cid in enumerate(ids):
        doc = documents[i] if i < len(documents) else ""
        meta = metadatas[i] if i < len(metadatas) else {}
        emb = embeddings[i] if i < len(embeddings) else []
        row: dict[str, Any] = {
            "id": cid,
            "document": doc,
            "metadata": meta,
            "embedding_dim": len(emb),
        }
        if include_embedding_preview and emb:
            row["embedding_preview"] = [float(x) for x in emb[:preview_n]]
        chunks.append(row)

    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "kind": "ingest_manifest",
        "collection_name": collection_name,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    rag_print(f"export_ingest_manifest chunks={len(chunks)} -> {export_path}", tag="rag.export")
    return save_json(payload, export_path)


def export_retrieval_results(
    *,
    query: str,
    hits: list[dict[str, Any]],
    export_path: str | Path,
    collection_name: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """检索命中落盘（含 query、distance、document、metadata）。"""
    payload: dict[str, Any] = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "kind": "retrieval_hits",
        "query": query,
        "collection_name": collection_name,
        "hit_count": len(hits),
        "hits": hits,
    }
    if extra:
        payload["extra"] = extra
    rag_print(f"export_retrieval_results hits={len(hits)} -> {export_path}", tag="rag.export")
    return save_json(payload, export_path)


def suggest_ingest_export_path(collection_name: str) -> Path:
    """默认文件名：``data/rag_exports/ingest_{collection}_{utc}.json``（安全化 collection）。"""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in collection_name)[:80]
    return default_export_dir() / f"ingest_{safe}_{utc_slug()}.json"


def export_ingest_chunks_pre_embed(
    *,
    collection_name: str,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    original_filename: str,
    parser_used: str,
    extract_engine: str,
    extract_detail: str,
    extract_warnings: list[str],
    max_chunk_chars: int,
    export_path: str | Path,
) -> Path:
    """
    **嵌入 API 调用之前** 导出：仅解析 + ``chunk_pages`` 后的文本块与元数据，**不含向量**。

    与 ``export_ingest_manifest``（在拿到 embedding 之后、写入 Chroma 之前写出，带 ``embedding_dim``）用途不同。
    """
    chunks: list[dict[str, Any]] = []
    for i, cid in enumerate(ids):
        chunks.append(
            {
                "id": cid,
                "document": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
            }
        )
    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "kind": "ingest_chunks_pre_embed",
        "collection_name": collection_name,
        "original_filename": original_filename,
        "parser_used": parser_used,
        "extract_engine": extract_engine,
        "extract_detail": extract_detail,
        "extract_warnings": extract_warnings,
        "max_chunk_chars": max_chunk_chars,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "note": "无 embedding；若需检查入库前向量，请用 kind=ingest_manifest 的 JSON（embedding_dim / 可选 preview）",
    }
    rag_print(f"export_ingest_chunks_pre_embed chunks={len(chunks)} -> {export_path}", tag="rag.export")
    return save_json(payload, export_path)


def suggest_pre_embed_export_path(collection_name: str) -> Path:
    """默认文件名：``chunks_pre_embed_{collection}_{utc}.json``。"""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in collection_name)[:80]
    return default_export_dir() / f"chunks_pre_embed_{safe}_{utc_slug()}.json"


def suggest_retrieval_export_path(query: str) -> Path:
    """默认文件名：``data/rag_exports/retrieve_{slug}_{utc}.json``。"""
    q = "".join(c if c.isalnum() or c in "-_" else "_" for c in query.strip()[:40])
    if not q:
        q = "query"
    return default_export_dir() / f"retrieve_{q}_{utc_slug()}.json"
