"""Token 计数（无 tiktoken 时用启发式，避免强依赖）。"""


def num_tokens_from_string(s: str) -> int:
    if not s:
        return 0
    # 中英混合粗略估计：约 2 字符 ~ 1 token
    return max(1, len(s) // 2)
