from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class DocStoreConnection:
    """文档库连接占位类型（ES/Infinity/Chroma 等实现需满足 nlp/search.Dealer 用法）。"""

    pass


class MatchExpr:
    """检索表达式占位（SQLite 后端主要用 condition 字典）。"""


class MatchTextExpr(MatchExpr):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class MatchDenseExpr(MatchExpr):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class FusionExpr:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


@dataclass
class OrderByExpr:
    _orders: list[tuple[str, str]] = field(default_factory=list)

    def desc(self, field_name: str) -> OrderByExpr:
        self._orders.append(("desc", field_name))
        return self

    def asc(self, field_name: str) -> OrderByExpr:
        self._orders.append(("asc", field_name))
        return self
