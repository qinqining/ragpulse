"""图像惰性合并与打开（RAGFlow rag.utils.lazy_image 的最小子集）。"""

from __future__ import annotations

import io
from typing import Any

from PIL import Image


class LazyDocxImage:
    @staticmethod
    def merge(a, b):
        if a and not b:
            return a
        if b and not a:
            return b
        return b or a


def ensure_pil_image(img: Any) -> Image.Image | None:
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(bytes(img)))
        except Exception:
            return None
    if isinstance(img, str):
        try:
            return Image.open(img)
        except Exception:
            return None
    return None


def is_image_like(obj: Any) -> bool:
    if obj is None:
        return False
    if isinstance(obj, (Image.Image, LazyDocxImage)):
        return True
    return hasattr(obj, "size") and hasattr(obj, "tobytes")


def open_image_for_processing(obj: Any, allow_bytes: bool = True) -> tuple[Image.Image | None, bool]:
    """
    返回 (image, close_after)。
    ``close_after`` 为 True 时调用方应在用完后 ``img.close()``。
    """
    if obj is None:
        return None, False
    if isinstance(obj, Image.Image):
        return obj, False
    if allow_bytes and isinstance(obj, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(bytes(obj))), True
        except Exception:
            return None, False
    if isinstance(obj, str):
        try:
            return Image.open(obj), True
        except Exception:
            return None, False
    pil = ensure_pil_image(obj)
    return pil, pil is not None and pil is not obj
