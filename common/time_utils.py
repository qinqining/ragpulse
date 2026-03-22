import time
from datetime import datetime


def current_timestamp() -> int:
    return int(time.time() * 1000)


def date_string_to_timestamp(s: str) -> int:
    try:
        return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        return 0
