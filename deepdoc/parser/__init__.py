#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#
#  ragpulse: 延迟导入各解析器，避免未安装 python-docx 等时整个包无法加载。
#

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "PdfParser",
    "PlainParser",
    "DocxParser",
    "EpubParser",
    "ExcelParser",
    "PptParser",
    "HtmlParser",
    "JsonParser",
    "MarkdownParser",
    "TxtParser",
    "MarkdownElementExtractor",
]

_LAZY = {
    "DocxParser": ("deepdoc.parser.docx_parser", "RAGFlowDocxParser"),
    "EpubParser": ("deepdoc.parser.epub_parser", "RAGFlowEpubParser"),
    "ExcelParser": ("deepdoc.parser.excel_parser", "RAGFlowExcelParser"),
    "HtmlParser": ("deepdoc.parser.html_parser", "RAGFlowHtmlParser"),
    "JsonParser": ("deepdoc.parser.json_parser", "RAGFlowJsonParser"),
    "MarkdownParser": ("deepdoc.parser.markdown_parser", "RAGFlowMarkdownParser"),
    "MarkdownElementExtractor": (
        "deepdoc.parser.markdown_parser",
        "MarkdownElementExtractor",
    ),
    "PdfParser": ("deepdoc.parser.pdf_parser", "RAGFlowPdfParser"),
    "PlainParser": ("deepdoc.parser.pdf_parser", "PlainParser"),
    "PptParser": ("deepdoc.parser.ppt_parser", "RAGFlowPptParser"),
    "TxtParser": ("deepdoc.parser.txt_parser", "RAGFlowTxtParser"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_path, attr = _LAZY[name]
    mod = importlib.import_module(mod_path)
    val = getattr(mod, attr)
    globals()[name] = val
    return val


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
