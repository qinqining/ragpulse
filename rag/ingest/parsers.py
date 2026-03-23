"""
按文件类型把内容抽成「(页码, 文本)」列表，供后续分块与嵌入。

- PDF：`pdf` / `auto`+`.pdf` 时优先 **deepdoc（RAGFlowPdfParser）**，失败则 **pypdf** 按页抽字。
- 纯文本 / Markdown：整文件当作第 1 页。
- Word：段落拼成一页文本。

注意：**字符长度上限 `max_chunk_chars` 不在这里**，而在 `rag/ingest/chunking.py` 里对解析结果再切分。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from rag.utils.ingest_progress import ingest_log

_log = logging.getLogger(__name__)

# deepdoc 输出里常见的页码/位置标记，入库前需去掉，避免污染向量文本
TAG_RE = re.compile(r"@@[0-9.\t-]+##")

# 供 API / 前端下拉展示：value 写入库请求，label 给人看
PARSER_CHOICES: list[dict[str, str]] = [
    {"value": "auto", "label": "自动（.pdf→deepdoc 优先，失败再 pypdf）"},
    {"value": "pdf", "label": "PDF（同上，推荐）"},
    {"value": "pdf_deepdoc", "label": "PDF 仅 deepdoc（版面/OCR，依赖重）"},
    {"value": "pdf_pypdf", "label": "PDF 仅 pypdf（纯文本层）"},
    {"value": "txt", "label": "纯文本 .txt"},
    {"value": "md", "label": "Markdown .md"},
    {"value": "docx", "label": "Word .docx（需 python-docx）"},
]

# 当 parser=auto 时，用扩展名决定实际解析器；未列出的后缀默认按 txt 读
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
    """单次解析结果：多页文本 + 元信息。"""

    # (页码, 该页/该段正文)；非 PDF 时通常只有 [(1, 全文)]
    pages: list[tuple[int, str]]
    # 实际使用的引擎标识，如 pdf_deepdoc、pdf_pypdf、txt、docx
    engine: str = ""
    # 人类可读说明（策略、是否回退等）
    detail: str = ""
    # 非致命问题：例如 deepdoc 失败已回退 pypdf
    warnings: list[str] = field(default_factory=list)


def resolve_parser(filename: str, explicit: str) -> str:
    """
    根据显式 parser 与文件名后缀，得到最终解析器关键字。

    - explicit 非 ``auto``：原样返回（小写）。
    - ``auto``：按 ``filename`` 后缀查表，默认 ``txt``。
    """
    explicit = (explicit or "auto").strip().lower()
    if explicit != "auto":
        return explicit
    suf = Path(filename).suffix.lower()
    return _SUFFIX_TO_PARSER.get(suf, "txt")


def _strip_position_tags(text: str) -> str:
    """去掉 deepdoc 段落里的 ``@@...##`` 位置标签，保留可读正文。"""
    return TAG_RE.sub("", text).strip()


def _page_from_segment(segment: str) -> int:
    """从段落前缀解析页码（形如 ``@@12``）；没有则默认为 1。"""
    m = re.search(r"@@([0-9]+)", segment)
    if m:
        return int(m.group(1))
    return 1


def _extract_pdf_pypdf(path: Path) -> PageExtract:
    """仅用 PDF 文本层按页抽取（轻量；扫描件/复杂版式效果差）。"""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i + 1, text))
    return PageExtract(pages=pages, engine="pdf_pypdf", detail="pypdf 按页 extract_text")


def _extract_pdf_deepdoc(path: Path, *, need_image: bool = False) -> PageExtract:
    """
    使用 deepdoc 的 RAGFlowPdfParser：版面、表格、OCR 等（依赖重）。

    返回的 ``pages`` 由大段文本按 ``\\n\\n`` 切开；页码尽量从 ``@@数字`` 推断。
    """
    try:
        from deepdoc.parser.pdf_parser import RAGFlowPdfParser
    except ImportError as e:
        # 顶层 import 链会拉 pdfplumber/cv2/onnx 等，缺任一都会在这里失败
        raise RuntimeError(
            "无法导入 deepdoc PDF 模块："
            f"{e!s}。"
            "请安装：pip install -r requirements.txt（deepdoc 段）。"
            "排查：PYTHONPATH=. python -c \"from deepdoc.parser.pdf_parser import RAGFlowPdfParser\""
        ) from e

    try:
        ingest_log("deepdoc: 正在初始化 RAGFlowPdfParser（加载 xgb/版面模型等，可能较慢）…")
        parser = RAGFlowPdfParser()
    except Exception as e:
        raise RuntimeError(
            "deepdoc RAGFlowPdfParser 初始化失败（模型/HF 下载/rag/res/deepdoc 等）："
            f"{e!s}。可设 HF_ENDPOINT；详见 readme「deepdoc PDF」"
        ) from e

    # need_image：是否走插图/视觉描述路径；zoomin 与 RAGFlow 默认渲染倍数一致
    ingest_log("deepdoc: 正在解析 PDF 正文（长文档可能数分钟）…")
    text_blob, _tbls = parser(str(path), need_image=need_image, zoomin=3)
    if not text_blob or not str(text_blob).strip():
        return PageExtract(pages=[], engine="pdf_deepdoc", detail="deepdoc 无文本输出")

    # 双换行分段；每段内可能含 @@页码，下面会清洗并归页
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
    """
    入口：按 ``parser_kind`` 从磁盘文件抽取页面级文本。

    ``parser_kind`` 一般为 ``resolve_parser`` 的结果，如 ``pdf``、``txt``、``docx``。
    ``pdf`` 在 deepdoc 异常时会捕获并回退 pypdf，并把原因写入 ``warnings``。
    """
    kind = parser_kind.strip().lower()
    ingest_log(f"extract_pages: kind={kind!r} file={path.name!r}")

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
