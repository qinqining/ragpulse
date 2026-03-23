"""
从 PDF **内嵌图像流**导出到 ``web/static/images/``，供入库 chunk 的 metadata 引用。

- deepdoc / pypdf 文本解析仍只产出正文；本模块对**同一 PDF 文件**另做一轮 ``pypdf`` 按页抽图。
- ``image_uri`` 存 **JSON 字符串**（``list[str]``），元素均为 **站点相对路径**，例如 ``/static/images/<uuid>.png``，
  **不含协议与域名**，由前端或网关自行拼接。
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# 与 main.py 中 ``app.mount("/static", StaticFiles(..., web/static))`` 一致
IMAGE_URI_PREFIX = "/static/images"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def images_output_dir() -> Path:
    d = repo_root() / "web" / "static" / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


def extract_pdf_embedded_images(
    pdf_path: Path,
    *,
    max_images_per_page: int | None = None,
) -> dict[int, list[str]]:
    """
    按 PDF **物理页码**（1-based）提取内嵌位图。

    :return: ``{ 页码: [ "/static/images/xxx.png", ... ] }``
    """
    try:
        max_n = int(os.getenv("RAG_PDF_MAX_IMAGES_PER_PAGE", "32") or "32")
    except ValueError:
        max_n = 32
    if max_images_per_page is not None:
        max_n = max_images_per_page

    from pypdf import PdfReader

    out: dict[int, list[str]] = {}
    img_dir = images_output_dir()
    reader = PdfReader(str(pdf_path))
    for i, page in enumerate(reader.pages):
        page_num = i + 1
        uris: list[str] = []
        images = getattr(page, "images", None) or []
        for img in images[:max_n]:
            try:
                data = getattr(img, "data", None)
                if not data:
                    continue
                uid = uuid.uuid4().hex
                fname = f"{uid}.png"
                dest = img_dir / fname
                try:
                    from PIL import Image

                    im = Image.open(BytesIO(data))
                    if im.mode in ("RGBA", "LA", "P"):
                        im = im.convert("RGBA")
                    else:
                        im = im.convert("RGB")
                    im.save(dest, format="PNG")
                except Exception as pil_e:
                    _log.debug("PIL save failed, raw bytes: %s", pil_e)
                    ext = "bin"
                    name = getattr(img, "name", None) or ""
                    if isinstance(name, str) and "." in name:
                        ext = name.rsplit(".", 1)[-1].lower()[:8] or "bin"
                    fname = f"{uid}.{ext}"
                    dest = img_dir / fname
                    dest.write_bytes(data)
                rel = f"{IMAGE_URI_PREFIX}/{fname}".replace("\\", "/")
                uris.append(rel)
            except Exception as e:
                _log.warning("pdf page %s image skip: %s", page_num, e)
        if uris:
            out[page_num] = uris
    return out


def attach_image_uris_to_metadatas(
    metadatas: list[dict[str, Any]],
    page_to_uris: dict[int, list[str]],
) -> None:
    """
    按 chunk 已有 ``metadata["page"]`` 绑定该页全部图片 URI，写入 ``image_uri``（JSON 字符串列表）。

    无图则不写 ``image_uri``。
    """
    for meta in metadatas:
        try:
            pg = int(meta.get("page", 1) or 1)
        except (TypeError, ValueError):
            pg = 1
        uris = page_to_uris.get(pg)
        if not uris:
            continue
        meta["image_uri"] = json.dumps(uris, ensure_ascii=False)


def image_uri_list_from_metadata(meta: dict[str, Any]) -> list[str]:
    """解析入库时写入的 ``image_uri`` 字段，供检索端/前端使用。"""
    raw = meta.get("image_uri")
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    s = str(raw).strip()
    if not s:
        return []
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(x) for x in v]
        return [str(v)]
    except json.JSONDecodeError:
        return [s]
