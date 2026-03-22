#!/usr/bin/env python3
"""
RAG 端到端自测（不启动 HTTP）：
  PDF 文本抽取 → 分块 → DashScope 嵌入 → Chroma 入库（含 ingest manifest JSON）→ 检索（含 hits JSON）→ 可选 LLM 总结。

用法（仓库根目录）：
  export RAG_VERBOSE=1
  pip install -r requirements.txt
  # 将 attention_is_all_you_need.pdf 放在项目根目录（或修改下方 PDF_CANDIDATES）
  PYTHONPATH=. python test.py

详见 TEST_RAG.md
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# -----------------------------------------------------------------------------
# 0) 路径与 .env（须先于 rag 导入）
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("RAG_VERBOSE", "1")

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from rag.retrieval import setup_rag_logging
from rag.utils.verbose import is_rag_verbose, rag_print

setup_rag_logging()

# 候选 PDF 文件名（按顺序查找）
PDF_CANDIDATES = [
    ROOT / "attention_is_all_you_need.pdf",
    ROOT / "attention_is_all_your_need.pdf",
]


def log(msg: str) -> None:
    """测试脚本主流程：始终打印到 stdout。"""
    print(f"[test] {msg}", flush=True)


def find_pdf() -> Path:
    for p in PDF_CANDIDATES:
        if p.is_file():
            return p
    found = list(ROOT.glob("*.pdf"))
    if len(found) == 1:
        return found[0]
    if not found:
        raise FileNotFoundError(
            "未找到 PDF。请将论文 PDF 放到项目根目录，命名为 "
            "attention_is_all_you_need.pdf（或修改 test.py 中 PDF_CANDIDATES）。"
        )
    raise FileNotFoundError(f"根目录有多个 PDF，请指定其一：{[f.name for f in found]}")


def extract_pdf_pages(path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i + 1, text))
    return pages


def chunk_pages(
    pages: list[tuple[int, str]],
    *,
    max_chars: int = 1500,
    source_name: str,
) -> list[tuple[str, dict]]:
    """(chunk_text, metadata)"""
    out: list[tuple[str, dict]] = []
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


def main() -> None:
    log("========== ragpulse RAG 端到端测试 ==========")
    log(f"RAG_VERBOSE={is_rag_verbose()} (设为 0/false 可关闭 rag 内部 [tag] 输出)")
    rag_print("test.py main() entered", tag="test")

    path = find_pdf()
    log(f"PDF: {path}")

    log("1/5 抽取 PDF 文本（pypdf，非 deepdoc 视觉管线）...")
    pages = extract_pdf_pages(path)
    log(f"    非空页数: {len(pages)}")
    if not pages:
        raise RuntimeError("PDF 未抽出任何文本（扫描版需 OCR / docling，本脚本仅测纯文本 RAG）")

    chunks_meta = chunk_pages(pages, source_name=path.name)
    log(f"    分块数: {len(chunks_meta)}")
    for i, (tx, meta) in enumerate(chunks_meta[:3]):
        log(f"    chunk[{i}] page={meta.get('page')} len={len(tx)} preview={tx[:60]!r}...")
    if len(chunks_meta) > 3:
        log(f"    ... 其余 {len(chunks_meta) - 3} 块省略预览")

    dept_tag = os.getenv("RAG_TEST_DEPT", "test_rag")
    kb_id = os.getenv("RAG_TEST_KB", "attention")
    user_id = os.getenv("RAG_USER_ID", "test_user")

    log(f"2/5 作用域 dept={dept_tag!r} kb_id={kb_id!r}（可用环境变量 RAG_TEST_DEPT / RAG_TEST_KB 覆盖）")

    from rag.embedding.qwen_embed import QwenTextEmbedding
    from rag.retrieval.chroma_client import ChromaRagStore, collection_name
    from rag.retrieval.json_export import suggest_ingest_export_path, suggest_retrieval_export_path
    from rag.retrieval.rag_retrieval import retrieve_for_query

    texts = [c[0] for c in chunks_meta]
    metadatas = [c[1] for c in chunks_meta]
    ids = [f"p{m['page']}_i{i}" for i, m in enumerate(metadatas)]

    log("3/5 嵌入 + 写入 Chroma（会先写 ingest manifest JSON）...")
    emb = QwenTextEmbedding()
    embeddings = emb.embed_documents(texts)

    col_name = collection_name(dept=dept_tag, kb_id=kb_id)
    store = ChromaRagStore()

    # 可选：清空同名 collection，避免重复 id 报错
    try:
        store._client.delete_collection(col_name)
        log(f"    已删除旧 collection: {col_name}")
    except Exception as e:
        log(f"    （跳过删除旧库）{e!r}")

    ingest_json = suggest_ingest_export_path(col_name)
    store.add(
        collection_name=col_name,
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        export_manifest_path=str(ingest_json),
    )
    log(f"    ingest manifest: {ingest_json}")

    query = os.getenv(
        "RAG_TEST_QUERY",
        "What is the Transformer and scaled dot-product attention?",
    )
    log(f"4/5 检索 query={query!r}")
    retrieve_json = suggest_retrieval_export_path(query)
    hits = retrieve_for_query(
        query=query,
        user_id=user_id,
        dept_tag=dept_tag,
        kb_id=kb_id,
        top_k=int(os.getenv("RAG_TEST_TOPK", "5")),
        export_path=str(retrieve_json),
    )
    log(f"    命中: {len(hits)} 条，已导出: {retrieve_json}")
    for i, h in enumerate(hits):
        doc = h.get("document") or ""
        log(f"    hit[{i}] id={h.get('id')} distance={h.get('distance'):.4f} text={doc[:120]!r}...")

    log("5/5 可选：LLM 根据检索片段总结（需 LLM_API_URL + LLM_API_KEY）")
    llm_url = (os.getenv("LLM_API_URL") or "").strip()
    llm_key = (os.getenv("LLM_API_KEY") or "").strip()
    if not llm_url or not llm_key:
        log("    跳过（未配置 LLM）")
    else:
        from rag.llm.chat_model import chat_completions

        context = "\n\n---\n\n".join(
            (h.get("document") or "")[:2000] for h in hits[:4] if h.get("document")
        )
        messages = [
            {
                "role": "system",
                "content": "你是助手，仅根据给定上下文简要回答，不确定请说不知道。",
            },
            {
                "role": "user",
                "content": f"上下文：\n{context}\n\n问题：{query}",
            },
        ]
        answer = chat_completions(messages=messages)
        log(f"    模型回答（节选）: {answer[:500]!r}...")

    log("========== 完成 ==========")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[test] 失败: {e}", file=sys.stderr, flush=True)
        raise
