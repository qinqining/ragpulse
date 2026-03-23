"""检索编排：嵌入查询 + Chroma + 可选融合（RAPTOR/重排可后续接）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rag.embedding.qwen_embed import QwenTextEmbedding
from rag.ingest.pdf_embedded_images import image_uri_list_from_metadata
from rag.retrieval.chroma_client import ChromaRagStore, collection_name
from rag.utils.verbose import rag_print

_log = logging.getLogger(__name__)


def retrieve_for_query(
    *,
    query: str,
    user_id: str,
    dept_tag: str,
    kb_id: str,
    top_k: int = 8,
    export_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    name = collection_name(dept=dept_tag, kb_id=kb_id)
    rag_print(
        f"retrieve_for_query start query={query[:120]!r}... collection={name!r} top_k={top_k}",
        tag="rag.retrieve",
    )
    _log.info("retrieve_for_query collection=%s top_k=%s", name, top_k)
    emb = QwenTextEmbedding()
    qv = emb.embed_query(query)
    store = ChromaRagStore()
    raw = store.query(collection_name=name, query_embedding=qv, n_results=top_k)
    ids = raw.get("ids", [[]])[0]
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    dists = raw.get("distances", [[]])[0]
    hits: list[dict[str, Any]] = []
    for i, cid in enumerate(ids):
        if not cid:
            continue
        meta = metas[i] if i < len(metas) else {}
        hits.append(
            {
                "id": cid,
                "document": docs[i] if i < len(docs) else "",
                "metadata": meta,
                "image_uris": image_uri_list_from_metadata(meta if isinstance(meta, dict) else {}),
                "distance": float(dists[i]) if i < len(dists) else 1.0,
            }
        )

    if export_path:
        from rag.retrieval.json_export import export_retrieval_results

        export_retrieval_results(
            query=query,
            hits=hits,
            export_path=export_path,
            collection_name=name,
            extra={"user_id": user_id, "dept_tag": dept_tag, "kb_id": kb_id, "top_k": top_k},
        )
    rag_print(f"retrieve_for_query done hits={len(hits)} export={export_path!r}", tag="rag.retrieve")
    _log.info("retrieve_for_query done hits=%s", len(hits))
    return hits
