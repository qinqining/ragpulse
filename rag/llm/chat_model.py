"""
OpenAI 兼容 Chat Completions（百炼 compatible-mode）。

仅负责 ``messages`` → HTTP → 回复文本；**业务提示词**在 ``rag/prompts/generator.py`` 等处组装后再传入。
"""

from __future__ import annotations

import json
import logging
import os

import requests

from rag.utils.verbose import rag_print

_log = logging.getLogger(__name__)


def chat_completions(
    *,
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.2,
    timeout: int = 180,
) -> str:
    url = (os.getenv("LLM_API_URL") or "").strip()
    key = (os.getenv("LLM_API_KEY") or "").strip()
    model = model or os.getenv("LLM_MODEL", "qwen-plus").strip()
    print(f"[DEBUG chat_completions] url={url[:50]}... model={model} messages={len(messages)}")
    if not url or not key:
        print("[ERROR] LLM_API_URL or LLM_API_KEY not configured!")
        raise RuntimeError("请配置 LLM_API_URL 与 LLM_API_KEY")
    rag_print(f"chat_completions model={model} messages={len(messages)}", tag="rag.llm")
    _log.info("chat_completions model=%s", model)
    payload = {"model": model, "messages": messages, "temperature": temperature}
    print(f"[DEBUG chat_completions] Sending request to LLM API...")
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        data=json.dumps(payload, ensure_ascii=False).encode(),
        timeout=timeout,
    )
    print(f"[DEBUG chat_completions] Response status={r.status_code}")
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        print("[WARN] chat_completions got empty choices")
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    out = content if isinstance(content, str) else str(content)
    rag_print(f"chat_completions done reply_len={len(out)}", tag="rag.llm")
    print(f"[DEBUG chat_completions] reply_len={len(out)}")
    return out
