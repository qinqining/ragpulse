"""
ragpulse 最小 HTTP 入口：健康检查 + RAG 检索演示（需配置 .env）。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parent
_STATIC = _ROOT / "web" / "static"

app = FastAPI(title="ragpulse", version="0.1.0")

if _STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _form_bool(v: str) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes", "on")


@app.get("/rag/ingest/options")
def rag_ingest_options() -> dict[str, Any]:
    """前端下拉：解析器类型说明。"""
    from rag.ingest import PARSER_CHOICES

    return {"parsers": PARSER_CHOICES}


@app.post("/rag/ingest")
def rag_ingest(
    file: UploadFile = File(...),
    dept_tag: str = Form("default"),
    kb_id: str = Form("default"),
    parser: str = Form("auto"),
    max_chunk_chars: int = Form(1500),
    replace_collection: str = Form("true"),
    export_manifest: str = Form("true"),
) -> dict[str, Any]:
    """
    上传文件 → 按 parser 解析 → 分块 → 嵌入 → 写入 Chroma。
    ``replace_collection=true`` 会先删除同名 collection 再写入（与 test.py 一致）。
    """
    from rag.ingest.service import run_ingest

    name = (file.filename or "upload").strip() or "upload"
    if max_chunk_chars < 200 or max_chunk_chars > 32000:
        raise HTTPException(400, "max_chunk_chars 建议 200~32000")

    suffix = Path(name).suffix or ".bin"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        file.file.close()

    try:
        dept = (dept_tag or os.getenv("RAG_DEPT", "default")).strip() or "default"
        kb = (kb_id or os.getenv("RAG_KB_ID", "default")).strip() or "default"
        return run_ingest(
            file_path=tmp_path,
            original_filename=name,
            dept_tag=dept,
            kb_id=kb,
            parser=parser,
            max_chunk_chars=max_chunk_chars,
            replace_collection=_form_bool(replace_collection),
            export_manifest=_form_bool(export_manifest),
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/")
def index_page():
    """极简 RAG 测试页（web/static/index.html）。"""
    index = _STATIC / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {"detail": "web/static/index.html missing", "docs": "/docs"}


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    # 覆盖 .env 中的 RAG_DEPT / RAG_KB_ID，便于与 test.py 入库一致
    dept_tag: str | None = None
    kb_id: str | None = None
    # 检索结果落盘 JSON；与 auto_export_retrieval 二选一或同时用（显式路径优先）
    export_path: str | None = None
    auto_export_retrieval: bool = False


@app.post("/rag/retrieve")
def rag_retrieve(body: RetrieveRequest) -> dict[str, Any]:
    from rag.retrieval.json_export import suggest_retrieval_export_path
    from rag.retrieval.rag_retrieval import retrieve_for_query

    export_path = body.export_path
    if body.auto_export_retrieval and not export_path:
        export_path = str(suggest_retrieval_export_path(body.query))

    dept = (body.dept_tag or os.getenv("RAG_DEPT", "default")).strip() or "default"
    kb = (body.kb_id or os.getenv("RAG_KB_ID", "default")).strip() or "default"

    hits = retrieve_for_query(
        query=body.query,
        user_id=os.getenv("RAG_USER_ID", "default"),
        dept_tag=dept,
        kb_id=kb,
        top_k=body.top_k,
        export_path=export_path if (body.export_path or body.auto_export_retrieval) else None,
    )
    return {
        "query": body.query,
        "dept_tag": dept,
        "kb_id": kb,
        "hits": hits,
        "export_path": export_path if (body.export_path or body.auto_export_retrieval) else None,
    }


def main() -> None:
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
