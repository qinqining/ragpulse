"""Chroma 持久化与 collection 命名。"""

from __future__ import annotations

import logging
import os
import time
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


def parse_collection_name(name: str) -> dict[str, str] | None:
    """
    反解析 Chroma collection 名到 ``dept_tag`` / ``kb_id``。

    collection 命名规则：
    ``ragpulse__{safe(dept)}__{safe(kb_id)}__{safe(kind)}``

    返回示例：
    ``{"dept_tag": "...", "kb_id": "...", "kind": "...", "collection_name": "..."}``
    """
    if not name or not name.startswith("ragpulse__"):
        return None
    # delimiter 是双下划线 '__'，collection_name 内部一般只有单下划线
    parts = name.split("__", 3)
    if len(parts) != 4:
        return None
    prefix, dept_safe, kb_safe, kind_safe = parts
    if prefix != "ragpulse":
        return None
    return {
        "dept_tag": dept_safe,
        "kb_id": kb_safe,
        "kind": kind_safe,
        "collection_name": name,
    }


def _chroma_add_batch_size() -> int:
    """``RAG_CHROMA_ADD_BATCH_SIZE``：>0 时分批 ``collection.add``；0 或未设置则一次性写入。"""
    try:
        n = int(os.getenv("RAG_CHROMA_ADD_BATCH_SIZE", "0").strip() or "0")
    except ValueError:
        n = 0
    return max(0, n)


def _chroma_add_retry_config() -> tuple[int, float]:
    try:
        retries = int(os.getenv("RAG_CHROMA_ADD_MAX_RETRIES", "3").strip() or "3")
    except ValueError:
        retries = 3
    retries = max(1, min(retries, 10))
    try:
        base = float(os.getenv("RAG_CHROMA_ADD_RETRY_SLEEP", "1.0").strip() or "1.0")
    except ValueError:
        base = 1.0
    return retries, max(0.1, base)


def _save_failed_chroma_batch(
    *,
    collection_name: str,
    batch_index: int,
    error: str,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> Path:
    """某批 ``col.add`` 彻底失败时落盘，便于补写（不含 embedding，需重新 embed 或从 manifest 找回）。"""
    from rag.retrieval.json_export import default_export_dir, save_json, utc_slug

    name_safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in collection_name)[:64]
    path = default_export_dir() / f"failed_chroma_batch_{name_safe}_{batch_index}_{utc_slug()}.json"
    payload = {
        "collection_name": collection_name,
        "batch_index": batch_index,
        "error": error,
        "count": len(ids),
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
        "hint": "未写入向量；可修复后对该批重新 embed 再 col.add，或从同次 ingest 的 manifest 取 embeddings",
    }
    save_json(payload, path)
    return path


class ChromaRagStore:
    def __init__(self) -> None:
        rag_print("ChromaRagStore __init__ PersistentClient...", tag="rag.chroma")
        # chromadb telemetry（PostHog）在部分环境里可能上报失败并刷屏；对本项目无业务影响，直接静音。
        logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
        logging.getLogger("chromadb.telemetry.product").setLevel(logging.CRITICAL)
        logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
        self._client = chromadb.PersistentClient(
            path=_persist_path(),
            settings=ChromaSettings(anonymized_telemetry=True),
        )
        rag_print("ChromaRagStore client ready", tag="rag.chroma")

    def get_collection(self, name: str):
        return self._client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def delete_collection(self, *, dept: str, kb_id: str, kind: str = "default") -> bool:
        """删除指定 collection。返回 True/False 表示是否成功。"""
        name = collection_name(dept=dept, kb_id=kb_id, kind=kind)
        try:
            self._client.delete_collection(name=name)
            _log.info("deleted collection=%s", name)
            rag_print(f"delete_collection name={name}", tag="rag.chroma")
            return True
        except Exception as e:
            _log.error("delete_collection failed name=%s: %s", name, e)
            return False

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
        bs = _chroma_add_batch_size()
        n = len(ids)
        retries, base_sleep = _chroma_add_retry_config()

        if bs <= 0 or n <= bs:
            self._add_with_retry(
                col,
                collection_name=collection_name,
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                batch_index=0,
                retries=retries,
                base_sleep=base_sleep,
            )
        else:
            total = (n + bs - 1) // bs
            for b in range(total):
                lo, hi = b * bs, min((b + 1) * bs, n)
                rag_print(
                    f"add batch {b + 1}/{total} rows [{lo}:{hi}) collection={collection_name!r}",
                    tag="rag.chroma",
                )
                self._add_with_retry(
                    col,
                    collection_name=collection_name,
                    ids=ids[lo:hi],
                    embeddings=embeddings[lo:hi],
                    documents=documents[lo:hi],
                    metadatas=metadatas[lo:hi],
                    batch_index=b,
                    retries=retries,
                    base_sleep=base_sleep,
                )
                # 与旧 LangChain 脚本里 sleep 类似，减轻本地 Chroma 连续写入压力（可选）
                pause = float(os.getenv("RAG_CHROMA_ADD_BATCH_PAUSE_SEC", "0") or "0")
                if pause > 0 and b < total - 1:
                    time.sleep(pause)

        rag_print(f"add done collection={collection_name!r} total_rows={n}", tag="rag.chroma")

    def _add_with_retry(
        self,
        col,
        *,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        batch_index: int,
        retries: int,
        base_sleep: float,
    ) -> None:
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                col.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                return
            except Exception as e:
                last_err = e
                wait = base_sleep * (2**attempt)
                _log.warning(
                    "chroma add failed batch=%s attempt=%s/%s: %s; sleep %.1fs",
                    batch_index,
                    attempt + 1,
                    retries,
                    e,
                    wait,
                )
                rag_print(
                    f"chroma add retry batch={batch_index} {attempt + 1}/{retries} err={e!r} sleep={wait:.1f}s",
                    tag="rag.chroma",
                )
                if attempt < retries - 1:
                    time.sleep(wait)

        assert last_err is not None
        fail_path = _save_failed_chroma_batch(
            collection_name=collection_name,
            batch_index=batch_index,
            error=repr(last_err),
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        _log.error("chroma add gave up batch=%s saved=%s", batch_index, fail_path)
        raise RuntimeError(
            f"Chroma 写入失败（batch={batch_index}），已落盘: {fail_path}。原因: {last_err}"
        ) from last_err

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
