#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

# ragpulse: 去除 infinity 依赖，使用 jieba + 可选 OpenCC。

from __future__ import annotations

import logging
import re

try:
    import jieba
    import jieba.posseg as pseg

    _HAS_JIEBA = True
except ImportError:
    jieba = None  # type: ignore
    pseg = None  # type: ignore
    _HAS_JIEBA = False

try:
    from opencc import OpenCC

    _opencc_t2s = OpenCC("t2s")
except Exception:
    _opencc_t2s = None


def strQ2B(ustring: str) -> str:
    """全角转半角（常见标点与空格）。"""
    if not ustring:
        return ""
    out = []
    for c in ustring:
        o = ord(c)
        if o == 0x3000:
            out.append(" ")
        elif 0xFF01 <= o <= 0xFF5E:
            out.append(chr(o - 0xFEE0))
        else:
            out.append(c)
    return "".join(out)


def tradi2simp(s: str) -> str:
    if not s:
        return ""
    if _opencc_t2s is None:
        return s
    try:
        return _opencc_t2s.convert(s)
    except Exception:
        return s


def is_chinese(s: str) -> bool:
    return bool(s) and all("\u4e00" <= ch <= "\u9fff" for ch in s if not ch.isspace())


def is_number(s: str) -> bool:
    return bool(s) and bool(re.fullmatch(r"[0-9]+\.?[0-9]*", s))


def is_alphabet(s: str) -> bool:
    return bool(s) and bool(re.fullmatch(r"[a-zA-Z]+", s))


def naive_qie(txt: str) -> list[str]:
    if not txt:
        return []
    if _HAS_JIEBA:
        return list(jieba.cut(txt))
    return re.findall(r"[\w]+|[^\w\s]", txt, flags=re.UNICODE)


def _freq_from_jieba(word: str) -> int:
    if not _HAS_JIEBA or not word:
        return 10
    try:
        from jieba import dt

        v = int(dt.FREQ.get(word, 0) or 0)
        return max(1, v)
    except Exception:
        return 10


class RagTokenizer:
    def tokenize(self, line: str) -> str:
        from common import settings

        line = line or ""
        if getattr(settings, "DOC_ENGINE_INFINITY", False):
            return line
        if _HAS_JIEBA:
            return " ".join(jieba.cut(line))
        return " ".join(naive_qie(line))

    def fine_grained_tokenize(self, tks: str) -> str:
        from common import settings

        if getattr(settings, "DOC_ENGINE_INFINITY", False):
            return tks or ""
        if not tks:
            return ""
        parts: list[str] = []
        for w in tks.split():
            if w and all("\u4e00" <= c <= "\u9fff" for c in w):
                parts.extend(list(w))
            else:
                parts.append(w)
        return " ".join(parts)

    def tag(self, t: str) -> str:
        if not t:
            return "x"
        if _HAS_JIEBA:
            try:
                for w, f in pseg.cut(t):
                    if w == t:
                        return f or "x"
            except Exception:
                logging.debug("jieba.posseg failed for %r", t, exc_info=True)
        if is_number(t):
            return "m"
        if is_alphabet(t):
            return "eng"
        if is_chinese(t):
            return "n"
        return "x"

    def freq(self, t: str) -> int:
        return _freq_from_jieba(t or "")


tokenizer = RagTokenizer()
tokenize = tokenizer.tokenize
fine_grained_tokenize = tokenizer.fine_grained_tokenize
tag = tokenizer.tag
freq = tokenizer.freq
