"""RAG 侧提示词集中管理（与 ``rag/llm/chat_model.py`` 解耦：后者只负责 HTTP Chat，不写业务 prompt）。"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# LLM 增强总结 / 视觉描述 —— 「可检索描述」统一任务说明
# （入库或 deepdoc 插图、整页 need_image 时，经 picture.vision_llm_chunk 发给多模态模型）
# ---------------------------------------------------------------------------

RETRIEVABLE_DESCRIPTION_TASK_ZH = """你的任务：
生成一份全面、便于检索的描述，需涵盖以下内容：
- 来自文本和表格的关键事实、数字与数据要点
- 所讨论的主要主题与核心概念
- 此内容能够回答的问题
- 视觉内容分析（图表、示意图、图片中的规律等）
- 用户可能使用的替代搜索词

请确保描述详细且便于检索 —— 优先考虑可查找性，而非简洁性。
直接输出上述「可检索描述」正文，不要寒暄、不要分点标题以外的多余套话。"""


def vision_llm_figure_describe_prompt() -> str:
    """文档内单张插图/截图，无上下文字段时的系统侧任务说明。"""
    return (
        RETRIEVABLE_DESCRIPTION_TASK_ZH
        + "\n\n【当前输入】文档中的一张图片或截图。若图中几乎没有可读信息，说明页面元素类型即可。"
    )


def vision_llm_figure_describe_prompt_with_context(*, context_above: str, context_below: str) -> str:
    """插图 + 文档上下文：在同一任务说明下结合上文下文。"""
    above = (context_above or "").strip()
    below = (context_below or "").strip()
    return (
        RETRIEVABLE_DESCRIPTION_TASK_ZH
        + "\n\n以下是图片在文档中的上文与下文片段，请结合上下文完成「可检索描述」。\n\n"
        f"【上文】\n{above}\n\n【下文】\n{below}\n\n请直接输出描述正文。"
    )


def vision_llm_describe_prompt(page: int = 1) -> str:
    """PDF 整页渲染图（``need_image`` 路径）：整页视觉 + 可读文字与版面。"""
    return (
        RETRIEVABLE_DESCRIPTION_TASK_ZH
        + f"\n\n【当前输入】PDF 第 {page} 页的页面图像。请结合图中文字、表格与图形完成「可检索描述」。"
    )


async def relevant_chunks_with_toc(query: str, toc: list, chat_mdl, topn: int) -> list:
    """
    目录增强检索占位：未接 LLM 时返回空列表，nlp/search 将回退为向量检索结果。
    后续可在此调用 chat_mdl 对 toc 条目打分并返回 [(chunk_id, sim), ...]。
    """
    _ = (query, toc, chat_mdl, topn)
    return []
