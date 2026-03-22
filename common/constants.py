from enum import Enum


class MemoryType(Enum):
    RAW = "RAW"


class LLMType(Enum):
    """与 RAGFlow 对齐的最小枚举（仅 deepdoc 视觉解析等需要）。"""

    CHAT = "chat"
    IMAGE2TEXT = "image2text"


# nlp/search 等字段名占位（无 ES 时仍作键名保留）
PAGERANK_FLD = "pagerank_fea"
TAG_FLD = "tag_kwd"
