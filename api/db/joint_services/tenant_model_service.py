"""租户默认模型（占位）：从环境变量读取，避免依赖 RAGFlow 数据库。"""

from __future__ import annotations

import os

from common.constants import LLMType


def get_tenant_default_model_by_type(tenant_id: str, llm_type: LLMType) -> dict:
    _ = tenant_id
    if llm_type == LLMType.IMAGE2TEXT:
        model = (
            os.getenv("VISION_MODEL", "").strip()
            or os.getenv("LLM_MODEL", "qwen-vl-plus").strip()
        )
    else:
        model = os.getenv("LLM_MODEL", "qwen-plus").strip()
    return {"model": model, "llm_type": llm_type}
