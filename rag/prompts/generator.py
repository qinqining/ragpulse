"""RAGFlow prompts.generator 的最小占位实现。"""

from __future__ import annotations


def vision_llm_figure_describe_prompt() -> str:
    return (
        "请用中文简要描述图中关键信息（对象、文字、结构、数据关系），"
        "便于后续文档检索与问答，不要寒暄。"
    )


def vision_llm_figure_describe_prompt_with_context(*, context_above: str, context_below: str) -> str:
    above = (context_above or "").strip()
    below = (context_below or "").strip()
    return (
        "以下是图片在文档中的上文与下文片段，请结合上下文用中文描述图片内容，"
        "突出与上下文的关联。\n\n"
        f"【上文】\n{above}\n\n【下文】\n{below}\n\n请直接输出图片描述。"
    )


def vision_llm_describe_prompt(page: int = 1) -> str:
    return (
        f"这是 PDF 第 {page} 页的页面图像。请用中文提取可读文字与版面结构要点，"
        "用于检索；无文字则说明页面类型（封面/目录/图表等）。"
    )


async def relevant_chunks_with_toc(query: str, toc: list, chat_mdl, topn: int) -> list:
    """
    目录增强检索占位：未接 LLM 时返回空列表，nlp/search 将回退为向量检索结果。
    后续可在此调用 chat_mdl 对 toc 条目打分并返回 [(chunk_id, sim), ...]。
    """
    _ = (query, toc, chat_mdl, topn)
    return []
