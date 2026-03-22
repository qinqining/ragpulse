"""字符串工具（替代 RAGFlow common.string_utils 的最小实现）。"""

from __future__ import annotations

import re


def remove_redundant_spaces(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text, flags=re.UNICODE).strip()
