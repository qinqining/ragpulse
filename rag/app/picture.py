"""
多模态图片描述（RAGFlow rag.app.picture 精简版）。
使用与 rag.llm.chat_model 相同的环境变量：LLM_API_URL / LLM_API_KEY。
"""

from __future__ import annotations

import base64
import io
import json
import os
import logging
from typing import Any

import requests
from PIL import Image

from rag.utils.verbose import rag_print

_log = logging.getLogger(__name__)


def _pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def vision_llm_chunk(
    *,
    binary: Any,
    vision_model: Any,
    prompt: str,
    callback=None,
) -> str:
    if callback:
        try:
            callback(0.55, "Calling vision LLM...")
        except Exception:
            pass

    if not isinstance(binary, Image.Image):
        rag_print("vision_llm_chunk skip: binary is not PIL.Image", tag="rag.picture")
        return ""

    url = (os.getenv("LLM_API_URL") or "").strip()
    key = (os.getenv("LLM_API_KEY") or "").strip()
    if not url or not key:
        msg = "未配置 LLM_API_URL / LLM_API_KEY，跳过图片描述"
        if callback:
            try:
                callback(0.6, msg)
            except Exception:
                pass
        rag_print(msg, tag="rag.picture")
        return ""

    model = getattr(vision_model, "model", None) or getattr(vision_model, "llm_config", {}).get(
        "model", os.getenv("VISION_MODEL", "qwen-vl-plus")
    )
    rag_print(f"vision_llm_chunk POST model={model} image={binary.size}", tag="rag.picture")
    _log.info("vision_llm_chunk model=%s size=%s", model, binary.size)

    content = [
        {"type": "text", "text": prompt or "请简要描述图片内容。"},
        {"type": "image_url", "image_url": {"url": _pil_to_data_url(binary)}},
    ]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.2,
    }
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=180,
    )
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        rag_print("vision_llm_chunk empty choices", tag="rag.picture")
        return ""
    msg = choices[0].get("message") or {}
    c = msg.get("content")
    out = c if isinstance(c, str) else str(c or "")
    rag_print(f"vision_llm_chunk done len={len(out)}", tag="rag.picture")
    return out
