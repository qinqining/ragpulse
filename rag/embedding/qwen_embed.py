import json
import logging
import os
import time
from typing import Any, List

import requests
from requests.adapters import HTTPAdapter

from rag.utils.verbose import rag_print

_log = logging.getLogger(__name__)

# 复用 TCP 连接，减少「每条文本新建 HTTPS」导致的握手失败（如 Connection reset by peer）
_session: requests.Session | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        # 连接池：同 host 多请求复用连接
        adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _session = s
    return _session


def _headers() -> dict:
    key = (os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY") or "").strip()
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _parse_vec(data: dict) -> List[float]:
    out = data.get("output") or {}
    embs = out.get("embeddings")
    if embs and isinstance(embs, list) and embs[0].get("embedding"):
        return list(embs[0]["embedding"])
    raise RuntimeError("bad embedding response")


def _retry_config() -> tuple[int, float]:
    """(max_attempts, base_sleep_sec)"""
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
    """带重试的 POST（应对 TLS reset、瞬时断网等）。"""
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
    out: List[List[float]] = []
    for i, t in enumerate(texts):
        if interval > 0 and i > 0:
            time.sleep(interval)
        preview = (t[:80] + "…") if len(t) > 80 else t
        rag_print(f"  [{i+1}/{len(texts)}] len={len(t)} preview={preview!r}", tag="rag.embed")
        payload = {"model": model, "input": {"texts": [t]}}
        r = _post_embed(url, payload, timeout=timeout)
        r.raise_for_status()
        vec = _parse_vec(r.json())
        rag_print(f"  [{i+1}/{len(texts)}] ok dim={len(vec)}", tag="rag.embed")
        out.append(vec)
    rag_print(f"embed_texts done total={len(out)}", tag="rag.embed")
    return out


class QwenTextEmbedding:
    def __init__(self, model: str = None) -> None:
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
        rag_print(f"QwenTextEmbedding init model={self.model}", tag="rag.embed")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return embed_texts(texts, model=self.model)

    def embed_query(self, text: str) -> List[float]:
        rag_print(f"embed_query len={len(text)} preview={text[:100]!r}...", tag="rag.embed")
        qv = embed_texts([text], model=self.model)[0]
        rag_print(f"embed_query done dim={len(qv)}", tag="rag.embed")
        return qv
