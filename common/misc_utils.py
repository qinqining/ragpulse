"""杂项：与 RAGFlow / deepdoc 对齐的最小实现。"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable

_log = logging.getLogger(__name__)

# deepdoc / nlp 中 ``await thread_pool_exec(sync_fn, *args)`` 使用
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ragpulse")


def pip_install_torch() -> None:
    """
    RAGFlow 原实现会在运行时尝试 ``pip install torch``。

    ragpulse **不**自动安装，以免污染环境；PDF 里 xgboost 上下拼接模型无 CUDA 时仍可 CPU 跑。
    若需 ``torch.cuda`` 加速 xgboost 权重加载，请自行: ``pip install torch``。
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        _log.info("torch 未安装，deepdoc/xgb 相关将跳过 CUDA 分支（可忽略或手动 pip install torch）")


async def thread_pool_exec(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    在线程池中执行阻塞函数，供 asyncio 代码 ``await``。

    RAGFlow 同名 API；此前同步版会导致 ``await thread_pool_exec(...)`` 在运行时异常。
    """
    loop = asyncio.get_running_loop()
    if kwargs:
        bound = partial(fn, *args, **kwargs)
        return await loop.run_in_executor(_executor, bound)
    return await loop.run_in_executor(_executor, fn, *args)
