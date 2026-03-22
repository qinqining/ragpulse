"""
全局配置：.env + SQLite 消息存储（无 ES/Infinity）。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")


def _messages_db_path() -> str:
    p = (os.getenv("MESSAGES_DB_PATH") or "data/messages.db").strip()
    if not p:
        p = "data/messages.db"
    path = Path(p)
    if not path.is_absolute():
        path = _ROOT / path
    return str(path)


class Settings:
    """与 RAGFlow `from common import settings` 对齐的最小实现。"""

    def __init__(self) -> None:
        from common.doc_store.sqlite_message_store import SqliteMessageStore

        self.msgStoreConn = SqliteMessageStore(_messages_db_path())
        # 文档引擎：ragpulse 不使用 Infinity，分词走本地 jieba
        self.DOC_ENGINE_INFINITY = (
            os.getenv("DOC_ENGINE_INFINITY", "").strip().lower() in ("1", "true", "yes")
        )
        # PDF/OCR 并行设备数（0 表示不并行，1 为单路）
        try:
            self.PARALLEL_DEVICES = max(0, int(os.getenv("PARALLEL_DEVICES", "1")))
        except ValueError:
            self.PARALLEL_DEVICES = 1


settings = Settings()
