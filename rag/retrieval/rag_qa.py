"""
RAG 全链路最后一步：检索 → 拼上下文 → 调用 LLM 生成回答。

- **纯文本**：默认；将 ``hits`` 中的 ``document`` 拼成「检索片段」写入 user 消息。
- **多模态（可选）**：当 ``use_vision=True`` 且配置了 ``RAG_PUBLIC_BASE_URL`` 时，将命中里的 ``image_uris``
  拼成 **可公网访问的绝对 URL** 一并传给兼容 OpenAI 视觉格式的 Chat API（如百炼 ``qwen-vl-*``）。

飞书 / 微信 / 自建 App：在各自服务端 ``POST`` 与本服务相同的 JSON 到 ``/rag/qa`` 即可复用（需网络可达本服务）。
"""

from __future__ import annotations

import os
from typing import Any

from rag.llm.chat_model import chat_completions
from rag.retrieval.rag_retrieval import retrieve_for_query
from rag.utils.verbose import rag_print


def _absolute_image_urls_from_hits(hits: list[dict[str, Any]], *, max_images: int) -> tuple[list[str], str | None]:
    """
    将 metadata 中的相对 ``/static/...`` 转为公网 URL。
    未配置 ``RAG_PUBLIC_BASE_URL`` 时返回 ([], reason)。
    """
    base = (os.getenv("RAG_PUBLIC_BASE_URL") or "").strip().rstrip("/")
    if not base:
        return [], "未设置 RAG_PUBLIC_BASE_URL，LLM 无法拉取图片（浏览器仍可本地看 /static）"

    out: list[str] = []
    seen: set[str] = set()
    for h in hits:
        for u in h.get("image_uris") or []:
            if not u or u in seen:
                continue
            seen.add(u)
            path = u if str(u).startswith("/") else f"/{u}"
            out.append(f"{base}{path}")
            if len(out) >= max_images:
                return out, None
    return out, None if out else "本批命中无 image_uris"


def run_rag_qa(
    *,
    query: str,
    dept_tag: str,
    kb_id: str,
    top_k: int = 5,
    use_vision: bool = False,
    vision_max_images: int = 4,
    user_id: str = "default",
) -> dict[str, Any]:
    """
    :return: ``hits``、``answer``、``used_vision``、``vision_note`` 等，供 HTTP / 飞书 / 微信转发。
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("query 不能为空")

    hits = retrieve_for_query(
        query=q,
        user_id=user_id,
        dept_tag=dept_tag,
        kb_id=kb_id,
        top_k=top_k,
        export_path=None,
    )

    blocks = []
    for i, h in enumerate(hits):
        doc = (h.get("document") or "").strip()
        src = ""
        meta = h.get("metadata") or {}
        chunk_summary = None
        if isinstance(meta, dict):
            src = str(meta.get("source") or meta.get("page") or "")
            v = meta.get("chunk_summary")
            if isinstance(v, str) and v.strip():
                chunk_summary = v.strip()

        if chunk_summary:
            blocks.append(
                f"[片段{i + 1}]{(' 来源:' + src) if src else ''}\n"
                f"[增强摘要]\n{chunk_summary}\n\n"
                f"[正文]\n{doc}"
            )
        else:
            blocks.append(f"[片段{i + 1}]{(' 来源:' + src) if src else ''}\n{doc}")
    context = "\n\n".join(blocks) if blocks else "（无检索命中）"

    system = (
        "你是严谨的知识助手。请**仅根据**下面「检索片段」回答用户问题；"
        "若片段不足以回答，请明确说明。可结合用户提供的插图（若有）理解文档。"
        "使用简体中文，条理清晰。"
    )
    user_text = f"用户问题：{q}\n\n--- 检索片段 ---\n{context}"

    vision_note: str | None = None
    used_vision = False
    image_urls: list[str] = []

    if use_vision:
        image_urls, vision_note = _absolute_image_urls_from_hits(hits, max_images=max(1, vision_max_images))
        used_vision = len(image_urls) > 0

    if used_vision:
        # 优先使用显式的视觉模型；否则回退到 .env 的 VISION_MODEL（仓库默认就用它），再退回纯文本 LLM_MODEL。
        model = (
            os.getenv("LLM_VISION_MODEL")
            or os.getenv("VISION_MODEL")
            or os.getenv("LLM_MODEL")
            or "qwen-vl-plus"
        ).strip()
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for u in image_urls:
            content.append({"type": "image_url", "image_url": {"url": u}})
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        rag_print(f"run_rag_qa vision images={len(image_urls)} model={model}", tag="rag.qa")
        answer = chat_completions(messages=messages, model=model)
    else:
        if use_vision and vision_note:
            rag_print(f"run_rag_qa vision skipped: {vision_note}", tag="rag.qa")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]
        answer = chat_completions(messages=messages)

    return {
        "query": q,
        "dept_tag": dept_tag,
        "kb_id": kb_id,
        "top_k": top_k,
        "hits": hits,
        "answer": answer,
        "used_vision": used_vision,
        "vision_note": vision_note,
        "vision_image_urls_sent_to_llm": image_urls if used_vision else [],
    }
