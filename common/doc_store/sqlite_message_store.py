"""
对话消息存储：SQLite 实现，替代 ES/Infinity/OceanBase。
满足 memory/services/messages.py 对 msgStoreConn 的调用约定。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


def _norm_index_names(index_names: str | list[str]) -> list[str]:
    if isinstance(index_names, str):
        return [index_names]
    return list(index_names or [])


class SqliteMessageStore:
    def __init__(self, db_path: str) -> None:
        self._path = db_path
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        cx = sqlite3.connect(self._path, check_same_thread=False)
        cx.row_factory = sqlite3.Row
        return cx

    def _init_schema(self) -> None:
        with self._lock, self._conn() as cx:
            cx.execute(
                """
                CREATE TABLE IF NOT EXISTS msg_docs (
                    id TEXT PRIMARY KEY,
                    index_name TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    doc_json TEXT NOT NULL
                )
                """
            )
            cx.execute(
                "CREATE INDEX IF NOT EXISTS idx_msg_idx_mem ON msg_docs(index_name, memory_id)"
            )

    def index_exist(self, index: str, memory_id: str) -> bool:
        with self._lock, self._conn() as cx:
            cur = cx.execute(
                "SELECT 1 FROM msg_docs WHERE index_name=? AND memory_id=? LIMIT 1",
                (index, memory_id),
            )
            return cur.fetchone() is not None

    def create_idx(self, index: str, memory_id: str, vector_size: int) -> bool:
        return True

    def delete_idx(self, index: str, memory_id: str) -> bool:
        with self._lock, self._conn() as cx:
            cx.execute("DELETE FROM msg_docs WHERE index_name=? AND memory_id=?", (index, memory_id))
        return True

    def insert(self, messages: list[dict], index: str, memory_id: str) -> bool:
        with self._lock, self._conn() as cx:
            for m in messages:
                doc = dict(m)
                rid = doc.get("id") or f'{memory_id}_{doc.get("message_id", "")}'
                cx.execute(
                    "INSERT OR REPLACE INTO msg_docs (id, index_name, memory_id, doc_json) VALUES (?,?,?,?)",
                    (rid, index, memory_id, json.dumps(doc, ensure_ascii=False)),
                )
        return True

    def update(self, condition: dict, update_dict: dict, index: str, memory_id: str) -> bool:
        rows = self._fetch_all(index, [memory_id])
        with self._lock, self._conn() as cx:
            for row in rows:
                doc = json.loads(row["doc_json"])
                if not self._match_condition(doc, condition):
                    continue
                doc.update(update_dict)
                cx.execute(
                    "UPDATE msg_docs SET doc_json=? WHERE id=?",
                    (json.dumps(doc, ensure_ascii=False), row["id"]),
                )
        return True

    def delete(self, condition: dict, index: str, memory_id: str) -> bool:
        rows = self._fetch_all(index, [memory_id])
        with self._lock, self._conn() as cx:
            for row in rows:
                doc = json.loads(row["doc_json"])
                if self._match_condition(doc, condition):
                    cx.execute("DELETE FROM msg_docs WHERE id=?", (row["id"],))
        return True

    def _fetch_all(self, index: str, memory_ids: list[str]) -> list[sqlite3.Row]:
        with self._conn() as cx:
            q = "SELECT * FROM msg_docs WHERE index_name=? AND memory_id IN (%s)" % (
                ",".join("?" * len(memory_ids)),
            )
            return list(cx.execute(q, (index, *memory_ids)))

    def _fetch_multi_index(
        self, index_names: list[str], memory_ids: list[str]
    ) -> list[sqlite3.Row]:
        if not index_names or not memory_ids:
            return []
        with self._conn() as cx:
            in_idx = ",".join("?" * len(index_names))
            in_mem = ",".join("?" * len(memory_ids))
            q = f"SELECT * FROM msg_docs WHERE index_name IN ({in_idx}) AND memory_id IN ({in_mem})"
            return list(cx.execute(q, (*index_names, *memory_ids)))

    @staticmethod
    def _match_condition(doc: dict, cond: dict) -> bool:
        for k, v in (cond or {}).items():
            if k not in doc:
                return False
            dv = doc[k]
            if isinstance(v, list):
                if dv not in v:
                    return False
            elif dv != v:
                return False
        return True

    def search(
        self,
        select_fields: list[str],
        highlight_fields: list[str],
        condition: dict,
        match_expressions: list,
        order_by: Any,
        offset: int,
        limit: int,
        index_names: str | list[str],
        memory_ids: list[str],
        agg_fields: list,
        hide_forgotten: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        del highlight_fields, match_expressions, agg_fields  # SQLite MVP 忽略
        idxs = _norm_index_names(index_names)
        rows = self._fetch_multi_index(idxs, memory_ids)
        docs: list[dict[str, Any]] = []
        for row in rows:
            doc = json.loads(row["doc_json"])
            if hide_forgotten and doc.get("forget_at"):
                continue
            if not self._match_condition(doc, condition):
                continue
            docs.append({"id": row["id"], **doc})

        orders = getattr(order_by, "_orders", []) or []
        for direction, field in reversed(orders):
            rev = direction == "desc"
            docs.sort(key=lambda d: d.get(field, 0) or 0, reverse=rev)

        total = len(docs)
        page = docs[offset : offset + limit]
        return page, total

    def get_fields(self, res: list[dict], fields: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        if not fields:
            return {}
        for doc in res:
            rid = doc.get("id")
            if not rid:
                continue
            m: dict[str, Any] = {}
            for n in fields:
                if n not in doc:
                    continue
                v = doc[n]
                if isinstance(v, list):
                    m[n] = v
                elif n in ("message_id", "source_id", "valid_at", "invalid_at", "forget_at", "status") and isinstance(
                    v, (int, float, bool)
                ):
                    m[n] = v
                elif not isinstance(v, str):
                    m[n] = str(v)
                else:
                    m[n] = v
            if m:
                out[str(rid)] = m
        return out

    def get(self, doc_id: str, index: str, memory_ids: list[str]) -> dict | None:
        with self._lock, self._conn() as cx:
            cur = cx.execute(
                "SELECT doc_json FROM msg_docs WHERE id=? AND index_name=? AND memory_id IN (%s)"
                % ",".join("?" * len(memory_ids)),
                (doc_id, index, *memory_ids),
            )
            r = cur.fetchone()
            if not r:
                return None
            return json.loads(r[0])

    def get_forgotten_messages(self, select_fields: list[str], index_name: str, memory_id: str) -> list[dict]:
        del select_fields
        rows = self._fetch_all(index_name, [memory_id])
        out: list[dict[str, Any]] = []
        for row in rows:
            d = json.loads(row["doc_json"])
            if d.get("forget_at"):
                out.append({"id": row["id"], **d})
        return out

    def get_missing_field_message(
        self,
        *,
        select_fields: list[str],
        index_name: str,
        memory_id: str,
        field_name: str,
    ) -> list[dict]:
        rows = self._fetch_all(index_name, [memory_id])
        out = []
        for row in rows:
            doc = json.loads(row["doc_json"])
            if field_name not in doc or doc[field_name] is None or doc[field_name] == "":
                out.append({"id": row["id"], **doc})
        return out
