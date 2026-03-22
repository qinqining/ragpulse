"""连接/调用辅助（RAGFlow 兼容占位）。"""

from __future__ import annotations

import concurrent.futures
import functools
import logging


def timeout(seconds: float, retries: int = 1):
    """
    装饰器：在独立线程中执行函数，单次调用最多等待 ``seconds`` 秒；
    失败或超时则最多重试 ``retries`` 次（与 RAGFlow 用法兼容）。
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: BaseException | None = None
            attempts = max(1, int(retries))
            for attempt in range(attempts):
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(fn, *args, **kwargs)
                    try:
                        return fut.result(timeout=seconds)
                    except concurrent.futures.TimeoutError as e:
                        last_exc = e
                        logging.warning(
                            "%s timeout (%.1fs) attempt %s/%s",
                            getattr(fn, "__name__", "fn"),
                            seconds,
                            attempt + 1,
                            attempts,
                        )
                    except Exception as e:
                        last_exc = e
                        logging.warning(
                            "%s error attempt %s/%s: %s",
                            getattr(fn, "__name__", "fn"),
                            attempt + 1,
                            attempts,
                            e,
                        )
            if last_exc:
                raise last_exc
            raise RuntimeError("timeout decorator: no result")

        return wrapper

    return decorator
