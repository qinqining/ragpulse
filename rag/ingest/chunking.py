"""
解析完成后的二次分块（字符窗口）。

与 deepdoc / pypdf 的关系：
- **deepdoc** 已在 PDF 内做版面合并，输出带 ``\\n\\n`` 分隔的语义段；本模块只在「单段仍过长」时按
  ``max_chars`` 做滑动切分，避免单条 embedding 超长。
- **pypdf 按页** 时，一页可能很长，同样靠 ``max_chars`` 切开。

这不是 RAGFlow ``naive_merge`` 的 token 策略，而是 ragpulse 入库侧的 **字符长度上限**（可调）。
"""

from __future__ import annotations

from typing import Any


def chunk_pages(
    pages: list[tuple[int, str]],
    *,
    max_chars: int = 1500,
    source_name: str,
) -> list[tuple[str, dict[str, Any]]]:
    """(chunk_text, metadata)，含 page / part / source。"""
    out: list[tuple[str, dict[str, Any]]] = []
    for page_num, text in pages:
        if len(text) <= max_chars:
            out.append((text, {"page": page_num, "source": source_name}))
            continue
        start = 0
        part = 0
        while start < len(text):
            piece = text[start : start + max_chars].strip()
            if piece:
                out.append(
                    (
                        piece,
                        {"page": page_num, "part": part, "source": source_name},
                    )
                )
            start += max_chars
            part += 1
    return out
