from __future__ import annotations

from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def singleton(cls: type) -> type:
    instances: dict[type, object] = {}

    def _wrap(*args: Any, **kwargs: Any) -> object:
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    class Proxy:
        def __new__(cls2, *a: Any, **kw: Any) -> object:
            return _wrap(*a, **kw)

    Proxy.__name__ = cls.__name__
    Proxy.__doc__ = cls.__doc__
    return Proxy  # type: ignore[return-value]
