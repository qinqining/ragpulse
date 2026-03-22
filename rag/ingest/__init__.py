"""文件入库：按类型解析 → 分块 → 嵌入 → Chroma。"""

from rag.ingest.parsers import PARSER_CHOICES, PageExtract, extract_pages, resolve_parser
from rag.ingest.service import run_ingest

__all__ = ["run_ingest", "PARSER_CHOICES", "PageExtract", "extract_pages", "resolve_parser"]
