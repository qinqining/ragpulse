"""
阿里云 DashScope **文本向量**（HTTP 调用，非本地模型）。

- 默认模型 ``text-embedding-v3``，与入库 ``run_ingest``、检索 ``rag_retrieval`` 共用同一套向量空间。
- 每条文本 **单独请求**（与部分 LangChain ``DashScopeEmbeddings`` 行为类似）；大批量时可配 ``EMBEDDING_INTERVAL_SEC`` 做简单限流。

**环境变量（常用）**

- ``EMBEDDING_API_KEY`` 或 ``LLM_API_KEY``：Bearer Token。
- ``EMBEDDING_API_URL``：默认 DashScope embeddings 端点。
- ``EMBEDDING_MODEL``：默认 ``text-embedding-v3``。
- ``EMBEDDING_TIMEOUT``：单次 POST 超时秒数。
- ``EMBEDDING_MAX_RETRIES`` / ``EMBEDDING_RETRY_SLEEP``：连接失败时的重试次数与退避基数。
- ``EMBEDDING_INTERVAL_SEC``：相邻两条 embedding 请求之间的休眠（可选）。

详细见项目根目录 ``.env.example``。
"""

import json
import logging
import os
import time
from typing import Any, List

import requests
from requests.adapters import HTTPAdapter

from rag.utils.ingest_progress import ingest_log
from rag.utils.verbose import rag_print

_log = logging.getLogger(__name__)

# 进程内复用 Session，减少每条文本都新建 TCP/TLS 连接（易触发 reset / 慢）
# 注意：下面 Session 是「懒」的——模块被 import 时不会创建，第一次调 embedding 时才创建。
_session: requests.Session | None = None


def _get_session() -> requests.Session:
    """
    懒初始化：第一次调用时才创建全局唯一的 ``requests.Session``，之后一直复用，并挂上连接池。

    「懒」= 推迟到真正需要时再分配资源；避免仅 import 本模块就发起网络相关初始化。
    """
    global _session
    if _session is None:
        s = requests.Session()
        # pool_*：同一 host 上并发/串行多请求时复用底层连接
        adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _session = s
    return _session


def _headers() -> dict:
    """构造 Authorization；密钥优先专用 embedding，否则复用 LLM key。"""
    key = (os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY") or "").strip()
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _parse_vec(data: dict) -> List[float]:
    """从 DashScope JSON 响应里取出第一条向量的浮点列表；结构不对则抛错。"""
    out = data.get("output") or {}
    embs = out.get("embeddings")
    if embs and isinstance(embs, list) and embs[0].get("embedding"):
        return list(embs[0]["embedding"])
    raise RuntimeError("bad embedding response")


def _retry_config() -> tuple[int, float]:
    """读取重试策略：``(最大尝试次数, 首次休眠秒数)``；休眠按指数退避倍增。"""
    try:
        n = int(os.getenv("EMBEDDING_MAX_RETRIES", "5"))
    except ValueError:
        n = 5
    n = max(1, min(n, 15))
    try:
        base = float(os.getenv("EMBEDDING_RETRY_SLEEP", "1.0"))
    except ValueError:
        base = 1.0
    return n, max(0.1, base)


def _post_embed(url: str, payload: dict, timeout: int) -> requests.Response:
    """对 embedding 端点 POST；遇连接类错误按配置重试（TLS reset、超时等）。"""
    session = _get_session()
    max_attempts, base_sleep = _retry_config()
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            r = session.post(
                url,
                headers=_headers(),
                data=json.dumps(payload),
                timeout=timeout,
            )
            return r
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_err = e
            wait = base_sleep * (2**attempt)
            _log.warning(
                "embedding POST failed (%s/%s): %s; retry in %.1fs",
                attempt + 1,
                max_attempts,
                e,
                wait,
            )
            rag_print(
                f"retry {attempt + 1}/{max_attempts} after {e!r}, sleep {wait:.1f}s",
                tag="rag.embed",
            )
            if attempt == max_attempts - 1:
                raise
            time.sleep(wait)
    assert last_err is not None
    raise last_err


def embed_texts(texts: List[str], model: str = None, url: str = None) -> List[List[float]]:
    """
    将多条字符串依次编码为向量列表（与 ``texts`` 顺序一致）。

    每条调用一次 API（payload 内 ``texts`` 仅含当前一条），便于对齐 DashScope 文档与错误定位。
    HTTP 4xx/5xx 由 ``raise_for_status`` 抛出；连接问题走 ``_post_embed`` 重试。
    """
    model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-v3").strip()
    url = url or os.getenv(
        "EMBEDDING_API_URL",
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
    ).strip()
    key = (os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY") or "").strip()
    if not url or not key:
        raise RuntimeError("missing EMBEDDING_API_URL or API key")
    try:
        timeout = int(os.getenv("EMBEDDING_TIMEOUT", "120"))
    except ValueError:
        timeout = 120
    try:
        interval = float(os.getenv("EMBEDDING_INTERVAL_SEC", "0") or "0")
    except ValueError:
        interval = 0.0

    rag_print(f"embed_texts start count={len(texts)} model={model}", tag="rag.embed")
    _log.info("embed_texts count=%s model=%s url=%s", len(texts), model, url[:48] + "...")
    n = len(texts)
    out: List[List[float]] = []
    for i, t in enumerate(texts):
        if interval > 0 and i > 0:
            time.sleep(interval)
        # 入库终端进度：每 5 条或首尾打印（避免刷屏）
        if n > 0 and (i == 0 or i == n - 1 or (i + 1) % 5 == 0):
            ingest_log(f"嵌入请求 {i + 1}/{n}")
        preview = (t[:80] + "…") if len(t) > 80 else t
        rag_print(f"  [{i+1}/{len(texts)}] len={len(t)} preview={preview!r}", tag="rag.embed")
        # DashScope 单条 text-embedding 请求体格式
        payload = {"model": model, "input": {"texts": [t]}}
        r = _post_embed(url, payload, timeout=timeout)
        r.raise_for_status()
        vec = _parse_vec(r.json())
        rag_print(f"  [{i+1}/{len(texts)}] ok dim={len(vec)}", tag="rag.embed")
        out.append(vec)
    rag_print(f"embed_texts done total={len(out)}", tag="rag.embed")
    return out


class QwenTextEmbedding:
    """
    供 ``run_ingest`` / ``rag_retrieval`` 使用的薄封装：固定模型名，统一 ``embed_documents`` / ``embed_query`` 接口。
    """

    def __init__(self, model: str = None) -> None:
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
        rag_print(f"QwenTextEmbedding init model={self.model}", tag="rag.embed")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文档块转向量（入库时调用）。"""
        return embed_texts(texts, model=self.model)

    def embed_query(self, text: str) -> List[float]:
        """单条查询转向量（检索时调用）；内部仍走 ``embed_texts([text])``。"""
        rag_print(f"embed_query len={len(text)} preview={text[:100]!r}...", tag="rag.embed")
        qv = embed_texts([text], model=self.model)[0]
        rag_print(f"embed_query done dim={len(qv)}", tag="rag.embed")
        return qv
