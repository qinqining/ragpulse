"""
兼容层：将项目根目录的 `nlp/` 包暴露为 `rag.nlp` 命名空间。

一些（迁移自 ragflow 的）代码使用 `from rag.nlp import ...`，
而本项目实际实现位于顶层 `nlp/__init__.py`。
"""

from __future__ import annotations

import importlib
from typing import Any

_src = importlib.import_module("nlp")

# 将源模块的所有公开符号转发到当前命名空间。
globals().update({k: v for k, v in _src.__dict__.items() if not k.startswith("_")})

# 兼容：迁移自 ragflow 的代码常写 `from rag.nlp import rag_tokenizer`
# 这里提供同名引用（是模块对象，本体在 `nlp/rag_tokenizer.py`）。
try:
    globals()["rag_tokenizer"] = importlib.import_module("nlp.rag_tokenizer")
except Exception:  # pragma: no cover
    # 若本环境缺该模块，后续调用点会再报错更具体的信息
    pass

# 让 `help(rag.nlp)` / `dir(rag.nlp)` 更一致
__all__: list[str] = [k for k in globals().keys() if not k.startswith("_")]

