"""
入库进度：默认向 stderr 打印 ``[embed] ...``（flush），便于 uvicorn 终端观察长耗时任务。

关闭：``RAG_INGEST_PROGRESS=0`` 或 ``false``。
"""

from __future__ import annotations

import os
import sys
import time


def ingest_log(msg: str) -> None:
    v = (os.getenv("RAG_INGEST_PROGRESS", "1") or "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return
    print(f"[embed] {msg}", file=sys.stderr, flush=True)


def ingest_log_start(phase: str) -> float:
    """返回 t0，供 ``ingest_log_done`` 打印耗时。"""
    ingest_log(f"开始: {phase}")
    return time.perf_counter()


def ingest_log_done(phase: str, t0: float) -> None:
    dt = time.perf_counter() - t0
    ingest_log(f"完成: {phase}（用时 {dt:.1f}s）")
