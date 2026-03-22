"""
按类型抽取文本；PDF 默认走 deepdoc（RAGFlowPdfParser），可回退 pypdf。

max_chunk_chars 在 chunking 中生效，含义见 rag/ingest/chunking.py 文档字符串。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

_log = logging.getLogger(__name__)

TAG_RE = re.compile(r"@@[0-9.\t-]+##")

PARSER_CHOICES: list[dict[str, str]] = [
    {"value": "auto", "label": "自动（.pdf→deepdoc 优先，失败再 pypdf）"},
    {"value": "pdf", "label": "PDF（同上，推荐）"},
    {"value": "pdf_deepdoc", "label": "PDF 仅 deepdoc（版面/OCR，依赖重）"},
    {"value": "pdf_pypdf", "label": "PDF 仅 pypdf（纯文本层）"},
    {"value": "txt", "label": "纯文本 .txt"},
    {"value": "md", "label": "Markdown .md"},
    {"value": "docx", "label": "Word .docx（需 python-docx）"},
]

_SUFFIX_TO_PARSER: dict[str, str] = {
    ".pdf": "pdf",
    ".txt": "txt",
    ".text": "txt",
    ".md": "md",
    ".markdown": "md",
    ".docx": "docx",
}


@dataclass
class PageExtract:
    pages: list[tuple[int, str]]
    engine: str = ""
    detail: str = ""
    warnings: list[str] = field(default_factory=list)


def resolve_parser(filename: str, explicit: str) -> str:
    explicit = (explicit or "auto").strip().lower()
    if explicit != "auto":
        return explicit
    suf = Path(filename).suffix.lower()
    return _SUFFIX_TO_PARSER.get(suf, "txt")


def _strip_position_tags(text: str) -> str:
    return TAG_RE.sub("", text).strip()


def _page_from_segment(segment: str) -> int:
    m = re.search(r"@@([0-9]+)", segment)
    if m:
        return int(m.group(1))
    return 1


def _extract_pdf_pypdf(path: Path) -> PageExtract:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i + 1, text))
    return PageExtract(pages=pages, engine="pdf_pypdf", detail="pypdf 按页 extract_text")


def _extract_pdf_deepdoc(path: Path, *, need_image: bool = False) -> PageExtract:
    try:
        from deepdoc.parser.pdf_parser import RAGFlowPdfParser
    except ImportError as e:
        raise RuntimeError(
            "无法导入 deepdoc PDF 模块："
            f"{e!s}。"
            "请安装：pip install -r requirements.txt（deepdoc 段）。"
            "排查：PYTHONPATH=. python -c \"from deepdoc.parser.pdf_parser import RAGFlowPdfParser\""
        ) from e

    try:
        parser = RAGFlowPdfParser()
    except Exception as e:
        raise RuntimeError(
            "deepdoc RAGFlowPdfParser 初始化失败（模型/HF 下载/rag/res/deepdoc 等）："
            f"{e!s}。可设 HF_ENDPOINT；详见 readme「deepdoc PDF」"
        ) from e

    text_blob, _tbls = parser(str(path), need_image=need_image, zoomin=3)
    if not text_blob or not str(text_blob).strip():
        return PageExtract(pages=[], engine="pdf_deepdoc", detail="deepdoc 无文本输出")

    parts = [p.strip() for p in str(text_blob).split("\n\n") if p.strip()]
    pages: list[tuple[int, str]] = []
    for p in parts:
        pg = _page_from_segment(p)
        clean = _strip_position_tags(p)
        if clean:
            pages.append((pg, clean))

    return PageExtract(
        pages=pages,
        engine="pdf_deepdoc",
        detail="deepdoc RAGFlowPdfParser；段按 \\n\\n 拆；入库文本已去 @@ 位置标签",
    )


def extract_pages(path: Path, parser_kind: str) -> PageExtract:
    kind = parser_kind.strip().lower()

    if kind == "pdf_pypdf":
        return _extract_pdf_pypdf(path)

    if kind == "pdf_deepdoc":
        return _extract_pdf_deepdoc(path)

    if kind == "pdf":
        try:
            return _extract_pdf_deepdoc(path)
        except Exception as e:
            _log.warning("PDF deepdoc 失败，回退 pypdf: %s", e)
            pe = _extract_pdf_pypdf(path)
            pe.warnings.append(f"deepdoc 失败已回退 pypdf: {e}")
            pe.detail = (pe.detail or "") + "（回退）"
            return pe

    if kind in ("txt", "text", "md", "markdown"):
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            return PageExtract(pages=[], engine=kind, detail="空文件")
        return PageExtract(pages=[(1, text)], engine=kind, detail="整文件单页")

    if kind == "docx":
        try:
            import docx  # noqa: F401
        except ImportError as e:
            raise RuntimeError("请安装: pip install python-docx") from e
        from docx import Document

        doc = Document(str(path))
        lines = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        text = "\n".join(lines).strip()
        if not text:
            return PageExtract(pages=[], engine="docx", detail="无段落")
        return PageExtract(pages=[(1, text)], engine="docx", detail="段落合并单页")

    raise ValueError(
        f"不支持的 parser: {parser_kind!r}。可选: pdf, pdf_deepdoc, pdf_pypdf, txt, md, docx, auto"
    )
