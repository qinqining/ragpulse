"""
解析完成后的最终分块（token 主切分 + 字符兜底 cap）。

与 deepdoc / pypdf 的关系：
- deepdoc / pypdf 都先产出 ``(page_num, text)``。
- 本模块先按页合并，再按 ``chunk_token_num`` 作为主阈值切分。
- ``max_chars`` 仅作为兜底，防止异常长文本进入 embedding。

这不是 RAGFlow ``naive_merge`` 的 token 策略，而是 ragpulse 入库侧的 **字符长度上限**（可调）。
"""

from __future__ import annotations

import re
from typing import Any

from nlp import rag_tokenizer


def _token_count(text: str) -> int:
    return len((rag_tokenizer.tokenize(text or "") or "").split())


def _split_by_rule(text: str, *, doc_type: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    if doc_type == "laws":
        parts = re.split(r"(?=(?:第[一二三四五六七八九十百千万0-9]+条|第[一二三四五六七八九十百千万0-9]+章))", t)
        xs = [p.strip() for p in parts if p and p.strip()]
        return xs or [t]
    # paper/book/one 统一按常见句末标点断句
    parts = re.split(r"([。！？!?；;\n])", t)
    out: list[str] = []
    buf = ""
    for p in parts:
        if not p:
            continue
        if re.fullmatch(r"[。！？!?；;\n]", p):
            buf += p
            if buf.strip():
                out.append(buf.strip())
            buf = ""
        else:
            buf += p
    if buf.strip():
        out.append(buf.strip())
    return out or [t]


def chunk_pages(
    pages: list[tuple[int, str]],
    *,
    chunk_token_num: int = 512,
    max_chars: int = 1500,
    source_name: str,
    doc_type: str = "one",
    merge_by_page: bool = True,
) -> list[tuple[str, dict[str, Any]]]:
    """(chunk_text, metadata)，含 page/part/source/doc_type。

    先 token 主切分，再 max_chars 兜底切分。
    """
    out: list[tuple[str, dict[str, Any]]] = []
    if not pages:
        return out

    if merge_by_page:
        merged: dict[int, list[str]] = {}
        for page_num, text in pages:
            text = (text or "").strip()
            if not text:
                continue
            merged.setdefault(page_num, []).append(text)

        for page_num in sorted(merged.keys()):
            text = "\n\n".join(merged[page_num]).strip()
            if not text:
                continue
            units = _split_by_rule(text, doc_type=doc_type)
            cur = ""
            part = 0
            for u in units:
                if not u:
                    continue
                cand = (cur + ("\n" if cur else "") + u).strip()
                if chunk_token_num > 0 and cur and _token_count(cand) > chunk_token_num:
                    out.append((cur, {"page": page_num, "part": part, "source": source_name, "doc_type": doc_type}))
                    part += 1
                    cur = u
                else:
                    cur = cand
            if cur:
                out.append((cur, {"page": page_num, "part": part, "source": source_name, "doc_type": doc_type}))

    # merge_by_page=False：仅保留旧行为（每个段独立进入切分）
    for page_num, text in pages:
        text = (text or "").strip()
        if not text:
            continue
        out.append((text, {"page": page_num, "source": source_name, "doc_type": doc_type}))

    # max_chars 兜底：对 token 切分结果再做字符 cap
    final_out: list[tuple[str, dict[str, Any]]] = []
    for text, meta in out:
        if max_chars <= 0 or len(text) <= max_chars:
            final_out.append((text, meta))
            continue
        part = int(meta.get("part", 0) or 0)
        start = 0
        while start < len(text):
            piece = text[start : start + max_chars].strip()
            if piece:
                m = dict(meta)
                m["part"] = part
                final_out.append((piece, m))
                part += 1
            start += max_chars
    return final_out
