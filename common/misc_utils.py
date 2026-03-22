"""杂项（deepdoc 等占位调用）。"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ragpulse")


def thread_pool_exec(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return fn(*args, **kwargs)
