"""
RAG 全链路可观测：设置环境变量 RAG_VERBOSE=1 时向终端打印/打日志。

测试脚本可调用 setup_rag_logging() 启用标准 logging。
"""

from __future__ import annotations

import logging
import os
import sys


def is_rag_verbose() -> bool:
    return os.getenv("RAG_VERBOSE", "").strip().lower() in ("1", "true", "yes", "on")


def rag_print(msg: str, *, tag: str = "rag") -> None:
    """仅在 RAG_VERBOSE 开启时 print，前缀 [tag]。"""
    if is_rag_verbose():
        print(f"[{tag}] {msg}", file=sys.stderr, flush=True)


def setup_rag_logging(level: int = logging.DEBUG) -> None:
    """
    在 RAG_VERBOSE=1 时配置根 logger（若已有 handler 则仅把 rag 相关 logger 调到 DEBUG）。
    """
    if not is_rag_verbose():
        return
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=fmt, datefmt=datefmt, stream=sys.stderr)
    else:
        for h in root.handlers:
            h.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.setLevel(level)
    for name in ("rag", "rag.embedding", "rag.retrieval", "rag.llm", "rag.app"):
        logging.getLogger(name).setLevel(level)
