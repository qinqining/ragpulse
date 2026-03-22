"""
LLM 绑定占位：RAGFlow 的 LLMBundle 精简版，供 deepdoc 视觉解析传入模型名等配置。
"""

from __future__ import annotations

import os
from typing import Any


class LLMBundle:
    def __init__(self, tenant_id: str, llm_config: dict[str, Any] | None = None) -> None:
        self.tenant_id = tenant_id or ""
        self.llm_config = dict(llm_config or {})
        self.model = (
            self.llm_config.get("model")
            or os.getenv("VISION_MODEL", "").strip()
            or os.getenv("LLM_MODEL", "qwen-vl-plus").strip()
        )
