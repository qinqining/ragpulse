"""Chroma 持久化与 collection 命名。"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import chromadb
from rag.utils.verbose import rag_print

_log = logging.getLogger(__name__)
from chromadb.config import Settings as ChromaSettings


def _persist_path() -> str:
    p = (os.getenv("VECTOR_DB_DIR") or "vector_db/chroma").strip() or "vector_db/chroma"
    path = Path(p)
    if not path.is_absolute():
        root = Path(__file__).resolve().parents[2]
        path = root / p
    path.parent.mkdir(parents=True, exist_ok=True)
    s = str(path)
    rag_print(f"Chroma persist_path={s}", tag="rag.chroma")
    _log.info("Chroma persist_path=%s", s)
    return s


def collection_name(*, dept: str, kb_id: str, kind: str = "default") -> str:
    def safe(s: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:48]

    return f"ragpulse__{safe(dept)}__{safe(kb_id)}__{safe(kind)}"


class ChromaRagStore:
    def __init__(self) -> None:
        rag_print("ChromaRagStore __init__ PersistentClient...", tag="rag.chroma")
        self._client = chromadb.PersistentClient(
            path=_persist_path(),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        rag_print("ChromaRagStore client ready", tag="rag.chroma")

    def get_collection(self, name: str):
        return self._client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def add(
        self,
        *,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        export_manifest_path: str | Path | None = None,
        export_embedding_preview: bool = False,
    ) -> None:
        """
        ``export_manifest_path``：在写入 Chroma **之前** 将本批 chunk 导出为 JSON，便于检查。
        """
        rag_print(
            f"add collection={collection_name!r} chunks={len(ids)} export={export_manifest_path!r}",
            tag="rag.chroma",
        )
        _log.info("add collection=%s n=%s", collection_name, len(ids))
        if export_manifest_path:
            from rag.retrieval.json_export import export_ingest_manifest

            export_ingest_manifest(
                collection_name=collection_name,
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
                export_path=export_manifest_path,
                include_embedding_preview=export_embedding_preview,
            )
        col = self.get_collection(collection_name)
        col.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        rag_print(f"add done collection={collection_name!r}", tag="rag.chroma")

    def query(
        self,
        *,
        collection_name: str,
        query_embedding: list[float],
        n_results: int = 8,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rag_print(
            f"query collection={collection_name!r} n_results={n_results} q_dim={len(query_embedding)}",
            tag="rag.chroma",
        )
        _log.info("query collection=%s n_results=%s", collection_name, n_results)
        col = self.get_collection(collection_name)
        raw = col.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )
        n = len((raw.get("ids") or [[]])[0])
        rag_print(f"query done hits={n}", tag="rag.chroma")
        return raw
