from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from typing import Any

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
_STATIC = _ROOT / "web" / "static"
_log = logging.getLogger("main")
load_dotenv(_ROOT / ".env")

app = FastAPI(title="ragpulse", version="0.1.0")

# CORS — allow browser JS from any origin (dev friendly, tighten for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _configure_agent_chat_logging() -> None:
    """保证 /agent/chat 的 INFO 日志能打到终端（根 logger 默认 WARNING 会丢掉）。"""
    lg = logging.getLogger("api.agent_api")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(levelname)s [agent] %(message)s"))
        lg.addHandler(h)
        lg.propagate = False


# Include agent API routes
try:
    from api.agent_api import router as agent_router
    app.include_router(agent_router)
except ImportError:
    _log.warning("Agent API not available - agent module may not be installed")

if _STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

_INGEST_TASKS: dict[str, dict[str, Any]] = {}
_INGEST_LOCK = threading.Lock()


@app.get("/health")
def health() -> dict[str, Any]:
    return health_detail()


def _present(v: str | None) -> bool:
    return bool((v or "").strip())


@app.get("/health/detail")
def health_detail() -> dict[str, Any]:
    emb_url = os.getenv("EMBEDDING_API_URL")
    emb_key = os.getenv("EMBEDDING_API_KEY")
    llm_url = os.getenv("LLM_API_URL")
    llm_key = os.getenv("LLM_API_KEY")
    llm_model = os.getenv("LLM_MODEL")
    vl_model = os.getenv("LLM_VISION_MODEL")
    public_base = os.getenv("RAG_PUBLIC_BASE_URL")

    missing_hard: list[str] = []
    if not _present(emb_url):
        missing_hard.append("EMBEDDING_API_URL")
    if not (_present(emb_key) or _present(llm_key)):
        missing_hard.append("EMBEDDING_API_KEY or LLM_API_KEY")
    if not _present(llm_url):
        missing_hard.append("LLM_API_URL")
    if not _present(llm_key):
        missing_hard.append("LLM_API_KEY")

    # 多模态是可选能力，单独做 warn
    warnings: list[str] = []
    if _present(vl_model) and not _present(public_base):
        warnings.append("LLM_VISION_MODEL set but RAG_PUBLIC_BASE_URL missing; vision QA may skip images")

    return {
        "status": "ok" if not missing_hard else "degraded",
        "missing_hard": missing_hard,
        "warnings": warnings,
        "env_loaded_from": str(_ROOT / ".env"),
        "checks": {
            "EMBEDDING_API_URL": _present(emb_url),
            "EMBEDDING_API_KEY": _present(emb_key),
            "LLM_API_URL": _present(llm_url),
            "LLM_API_KEY": _present(llm_key),
            "LLM_MODEL": _present(llm_model),
            "LLM_VISION_MODEL": _present(vl_model),
            "RAG_PUBLIC_BASE_URL": _present(public_base),
        },
    }


def _form_bool(v: str) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes", "on")


def _now_ts() -> float:
    return time.time()


def _save_upload_to_temp(file: UploadFile, *, fallback_name: str = "upload") -> tuple[Path, str]:
    name = (file.filename or fallback_name).strip() or fallback_name
    suffix = Path(name).suffix or ".bin"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        file.file.close()
    return tmp_path, name


def _run_ingest_sync(
    *,
    file_path: Path,
    original_filename: str,
    dept_tag: str,
    kb_id: str,
    parser: str,
    pdf_doc_type: str,
    replace_collection: bool,
    export_manifest: bool,
    export_chunks_pre_embed: bool,
    extract_pdf_images: bool,
    llm_chunk_summary: bool,
) -> dict[str, Any]:
    from rag.ingest.service import run_ingest

    return run_ingest(
        file_path=file_path,
        original_filename=original_filename,
        dept_tag=dept_tag,
        kb_id=kb_id,
        parser=parser,
        pdf_doc_type=pdf_doc_type,
        replace_collection=replace_collection,
        export_manifest=export_manifest,
        export_chunks_pre_embed=export_chunks_pre_embed,
        extract_pdf_images=extract_pdf_images,
        llm_chunk_summary=llm_chunk_summary,
    )


def _task_update(task_id: str, **patch: Any) -> None:
    with _INGEST_LOCK:
        row = _INGEST_TASKS.get(task_id)
        if not row:
            return
        row.update(patch)
        row["updated_at"] = _now_ts()


def _start_ingest_task(*, task_id: str, payload: dict[str, Any]) -> None:
    def worker() -> None:
        t0 = _now_ts()
        _task_update(task_id, status="running", message="run_ingest running")
        try:
            result = _run_ingest_sync(**payload)
            _task_update(
                task_id,
                status="succeeded",
                message="run_ingest finished",
                result=result,
                elapsed_sec=round(_now_ts() - t0, 3),
            )
        except ValueError as e:
            _task_update(
                task_id,
                status="failed",
                error_type="ValueError",
                error=str(e),
                elapsed_sec=round(_now_ts() - t0, 3),
            )
        except RuntimeError as e:
            _task_update(
                task_id,
                status="failed",
                error_type="RuntimeError",
                error=str(e),
                elapsed_sec=round(_now_ts() - t0, 3),
            )
        except Exception as e:
            _log.error("ingest async worker crashed: %s", e, exc_info=True)
            _task_update(
                task_id,
                status="failed",
                error_type=type(e).__name__,
                error=str(e),
                elapsed_sec=round(_now_ts() - t0, 3),
            )
        finally:
            try:
                payload["file_path"].unlink(missing_ok=True)
            except Exception:
                pass

    th = threading.Thread(target=worker, name=f"ingest-task-{task_id[:8]}", daemon=True)
    th.start()


@app.get("/rag/ingest/options")
def rag_ingest_options() -> dict[str, Any]:
    """前端下拉：解析器类型说明。"""
    from rag.ingest import PARSER_CHOICES

    return {"parsers": PARSER_CHOICES}


@app.get("/rag/kbs")
def rag_kbs() -> dict[str, Any]:
    """
    列出 Chroma 中已存在的知识库（collection）。

    返回结构：
    - `collections`: [{dept_tag, kb_id, kind, collection_name}, ...]
    """
    from rag.retrieval.chroma_client import ChromaRagStore, parse_collection_name

    store = ChromaRagStore()
    cols = []
    for c in store._client.list_collections():
        parsed = parse_collection_name(c.name)
        if not parsed:
            continue
        cols.append(parsed)

    # 稳定排序：dept -> kb -> kind
    cols.sort(key=lambda x: (x.get("dept_tag", ""), x.get("kb_id", ""), x.get("kind", "")))
    return {"collections": cols}


@app.delete("/rag/kbs/{dept_tag}/{kb_id}")
def rag_kbs_delete(dept_tag: str, kb_id: str, kind: str = "default") -> dict[str, Any]:
    """删除指定 collection（按 dept_tag + kb_id + kind 定位）。"""
    from rag.retrieval.chroma_client import ChromaRagStore

    store = ChromaRagStore()
    ok = store.delete_collection(dept=dept_tag, kb_id=kb_id, kind=kind)
    if ok:
        return {"ok": True, "dept_tag": dept_tag, "kb_id": kb_id, "kind": kind}
    raise HTTPException(status_code=404, detail=f"Collection not found or delete failed: {dept_tag}/{kb_id}/{kind}")


@app.post("/rag/ingest")
def rag_ingest(
    file: UploadFile = File(...),
    dept_tag: str = Form("default"),
    kb_id: str = Form("default"),
    parser: str = Form("auto"),
    pdf_doc_type: str = Form("auto"),
    replace_collection: str = Form("true"),
    export_manifest: str = Form("true"),
    export_chunks_pre_embed: str = Form("true"),
    extract_pdf_images: str = Form("true"),
    llm_chunk_summary: str = Form("false"),
) -> dict[str, Any]:
    """
    上传文件 → 按 parser 解析 → 分块 → 嵌入 → 写入 Chroma。
    ``replace_collection=true`` 会先删除同名 collection 再写入（与 test.py 一致）。
    """
    name = (file.filename or "upload").strip() or "upload"
    tmp_path, name = _save_upload_to_temp(file)

    try:
        dept = (dept_tag or os.getenv("RAG_DEPT", "default")).strip() or "default"
        kb = (kb_id or os.getenv("RAG_KB_ID", "default")).strip() or "default"
        return _run_ingest_sync(
            file_path=tmp_path,
            original_filename=name,
            dept_tag=dept,
            kb_id=kb,
            parser=parser,
            pdf_doc_type=pdf_doc_type,
            replace_collection=_form_bool(replace_collection),
            export_manifest=_form_bool(export_manifest),
            export_chunks_pre_embed=_form_bool(export_chunks_pre_embed),
            extract_pdf_images=_form_bool(extract_pdf_images),
            llm_chunk_summary=_form_bool(llm_chunk_summary),
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        # 503 常见原因：DashScope 嵌入 key/额度、Chroma 写入失败等（HF 镜像成功 ≠ 整链路成功）
        _log.error("run_ingest RuntimeError → HTTP 503: %s", e, exc_info=True)
        raise HTTPException(503, str(e)) from e
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/rag/ingest/async")
def rag_ingest_async(
    file: UploadFile = File(...),
    dept_tag: str = Form("default"),
    kb_id: str = Form("default"),
    parser: str = Form("auto"),
    pdf_doc_type: str = Form("auto"),
    replace_collection: str = Form("true"),
    export_manifest: str = Form("true"),
    export_chunks_pre_embed: str = Form("true"),
    extract_pdf_images: str = Form("true"),
    llm_chunk_summary: str = Form("false"),
) -> dict[str, Any]:
    """
    异步入库：立即返回 task_id，客户端轮询 ``GET /rag/ingest/tasks/{task_id}`` 查看状态。
    """
    name = (file.filename or "upload").strip() or "upload"
    tmp_path, name = _save_upload_to_temp(file)

    dept = (dept_tag or os.getenv("RAG_DEPT", "default")).strip() or "default"
    kb = (kb_id or os.getenv("RAG_KB_ID", "default")).strip() or "default"

    task_id = uuid.uuid4().hex
    with _INGEST_LOCK:
        _INGEST_TASKS[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "message": "task queued",
            "created_at": _now_ts(),
            "updated_at": _now_ts(),
            "elapsed_sec": 0.0,
            "result": None,
            "error": None,
            "error_type": None,
            "params": {
                "original_filename": name,
                "dept_tag": dept,
                "kb_id": kb,
                "parser": parser,
                "pdf_doc_type": pdf_doc_type,
                "replace_collection": _form_bool(replace_collection),
                "export_manifest": _form_bool(export_manifest),
                "export_chunks_pre_embed": _form_bool(export_chunks_pre_embed),
                "extract_pdf_images": _form_bool(extract_pdf_images),
                "llm_chunk_summary": _form_bool(llm_chunk_summary),
            },
        }

    _start_ingest_task(
        task_id=task_id,
        payload={
            "file_path": tmp_path,
            "original_filename": name,
            "dept_tag": dept,
            "kb_id": kb,
            "parser": parser,
            "pdf_doc_type": pdf_doc_type,
            "replace_collection": _form_bool(replace_collection),
            "export_manifest": _form_bool(export_manifest),
            "export_chunks_pre_embed": _form_bool(export_chunks_pre_embed),
            "extract_pdf_images": _form_bool(extract_pdf_images),
            "llm_chunk_summary": _form_bool(llm_chunk_summary),
        },
    )
    return {"ok": True, "task_id": task_id, "status": "queued"}


class UrlIngestRequest(BaseModel):
    """网页 URL 入库请求体。"""
    url: str
    dept_tag: str = "default"
    kb_id: str = "default"
    parser: str = "url"
    replace_collection: bool = False
    export_manifest: bool = True
    export_chunks_pre_embed: bool = True
    llm_chunk_summary: bool = False


def _run_ingest_url_sync(
    *,
    url: str,
    dept_tag: str,
    kb_id: str,
    parser: str,
    replace_collection: bool,
    export_manifest: bool,
    export_chunks_pre_embed: bool,
    llm_chunk_summary: bool,
) -> dict[str, Any]:
    """
    将 URL 内容写入临时文件，然后走标准 run_ingest 链路。
    Trafilatura 清洗函数：输入 = 原始 HTML，纯 HTML 清洗库。
    """
    import tempfile

    from rag.ingest.service import run_ingest

    # URL 内容写入临时文件（parsers._extract_url 读取此文件）
    with tempfile.NamedTemporaryFile(delete=False, suffix=".url", mode="w", encoding="utf-8") as tmp:
        tmp.write(url)
        tmp_path = Path(tmp.name)

    original_filename = f"url_{url[:80]}"
    try:
        return run_ingest(
            file_path=tmp_path,
            original_filename=original_filename,
            dept_tag=dept_tag,
            kb_id=kb_id,
            parser=parser,
            pdf_doc_type="naive",
            replace_collection=replace_collection,
            export_manifest=export_manifest,
            export_chunks_pre_embed=export_chunks_pre_embed,
            extract_pdf_images=False,
            llm_chunk_summary=llm_chunk_summary,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def _start_url_ingest_task(*, task_id: str, payload: dict[str, Any]) -> None:
    def worker() -> None:
        t0 = _now_ts()
        _task_update(task_id, status="running", message="url ingest running")
        try:
            result = _run_ingest_url_sync(**payload)
            _task_update(
                task_id,
                status="succeeded",
                message="url ingest finished",
                result=result,
                elapsed_sec=round(_now_ts() - t0, 3),
            )
        except ValueError as e:
            _task_update(
                task_id,
                status="failed",
                error_type="ValueError",
                error=str(e),
                elapsed_sec=round(_now_ts() - t0, 3),
            )
        except RuntimeError as e:
            _task_update(
                task_id,
                status="failed",
                error_type="RuntimeError",
                error=str(e),
                elapsed_sec=round(_now_ts() - t0, 3),
            )
        except Exception as e:
            _log.error("url ingest worker crashed: %s", e, exc_info=True)
            _task_update(
                task_id,
                status="failed",
                error_type=type(e).__name__,
                error=str(e),
                elapsed_sec=round(_now_ts() - t0, 3),
            )

    th = threading.Thread(target=worker, name=f"url-ingest-{task_id[:8]}", daemon=True)
    th.start()


@app.post("/rag/ingest/url")
def rag_ingest_url(body: UrlIngestRequest = ...) -> dict[str, Any]:
    """
    网页 URL 入库（同步）：抓取 → Trafilatura 清洗 → 分块 → 嵌入 → 写入 Chroma。
    """
    dept = (body.dept_tag or os.getenv("RAG_DEPT", "default")).strip() or "default"
    kb = (body.kb_id or os.getenv("RAG_KB_ID", "default")).strip() or "default"
    try:
        return _run_ingest_url_sync(
            url=body.url,
            dept_tag=dept,
            kb_id=kb,
            parser="url",
            replace_collection=body.replace_collection,
            export_manifest=body.export_manifest,
            export_chunks_pre_embed=body.export_chunks_pre_embed,
            llm_chunk_summary=body.llm_chunk_summary,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        _log.error("run_ingest_url RuntimeError → HTTP 503: %s", e, exc_info=True)
        raise HTTPException(503, str(e)) from e


@app.post("/rag/ingest/url/async")
def rag_ingest_url_async(body: UrlIngestRequest) -> dict[str, Any]:
    """
    网页 URL 入库（异步）：立即返回 task_id，客户端轮询 GET /rag/ingest/tasks/{task_id} 查看状态。
    """
    if not body.url or not body.url.strip():
        raise HTTPException(400, "url cannot be empty")
    if not body.url.startswith(("http://", "https://")):
        raise HTTPException(400, "url must start with http:// or https://")

    dept = (body.dept_tag or os.getenv("RAG_DEPT", "default")).strip() or "default"
    kb = (body.kb_id or os.getenv("RAG_KB_ID", "default")).strip() or "default"

    task_id = uuid.uuid4().hex
    with _INGEST_LOCK:
        _INGEST_TASKS[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "message": "url task queued",
            "created_at": _now_ts(),
            "updated_at": _now_ts(),
            "elapsed_sec": 0.0,
            "result": None,
            "error": None,
            "error_type": None,
            "params": {
                "original_filename": f"url_{body.url[:80]}",
                "dept_tag": dept,
                "kb_id": kb,
                "parser": "url",
                "replace_collection": body.replace_collection,
                "export_manifest": body.export_manifest,
                "export_chunks_pre_embed": body.export_chunks_pre_embed,
                "llm_chunk_summary": body.llm_chunk_summary,
            },
        }

    _start_url_ingest_task(
        task_id=task_id,
        payload={
            "url": body.url,
            "dept_tag": dept,
            "kb_id": kb,
            "parser": "url",
            "replace_collection": body.replace_collection,
            "export_manifest": body.export_manifest,
            "export_chunks_pre_embed": body.export_chunks_pre_embed,
            "llm_chunk_summary": body.llm_chunk_summary,
        },
    )
    return {"ok": True, "task_id": task_id, "status": "queued"}


@app.get("/rag/ingest/tasks/{task_id}")
def rag_ingest_task_status(task_id: str) -> dict[str, Any]:
    with _INGEST_LOCK:
        row = _INGEST_TASKS.get(task_id)
        if not row:
            raise HTTPException(404, f"task_id not found: {task_id}")
        out = dict(row)
    if out.get("status") in ("queued", "running"):
        out["elapsed_sec"] = round(_now_ts() - float(out.get("created_at") or _now_ts()), 3)
    return out


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


class RagQaRequest(BaseModel):
    """检索 + LLM 回答（整条 RAG 演示）；多模态需 ``RAG_PUBLIC_BASE_URL`` + 视觉模型。"""

    query: str
    top_k: int = 5
    dept_tag: str | None = None
    kb_id: str | None = None
    # 是否把命中里的图片以 image_url 发给 LLM（需公网可访问的 RAG_PUBLIC_BASE_URL）
    use_vision: bool = False
    vision_max_images: int = 4


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


@app.post("/rag/qa")
def rag_qa(body: RagQaRequest) -> dict[str, Any]:
    """
    **检索 → LLM 回答**，便于在网页 / 飞书 / 微信服务端复现整条 RAG。

    - 纯文本：配置 ``LLM_API_URL`` + ``LLM_API_KEY`` 即可。
    - 多模态：``use_vision=true`` 时需 ``RAG_PUBLIC_BASE_URL``（如 ``https://api.example.com``）使云厂商能拉取
      ``/static/images/...``；并建议 ``LLM_VISION_MODEL=qwen-vl-plus`` 等视觉模型。
    """
    from rag.retrieval.rag_qa import run_rag_qa

    dept = (body.dept_tag or os.getenv("RAG_DEPT", "default")).strip() or "default"
    kb = (body.kb_id or os.getenv("RAG_KB_ID", "default")).strip() or "default"
    try:
        return run_rag_qa(
            query=body.query,
            dept_tag=dept,
            kb_id=kb,
            top_k=body.top_k,
            use_vision=body.use_vision,
            vision_max_images=body.vision_max_images,
            user_id=os.getenv("RAG_USER_ID", "default"),
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        _log.error("run_rag_qa RuntimeError: %s", e, exc_info=True)
        raise HTTPException(503, str(e)) from e


def main() -> None:
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
