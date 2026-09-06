"""断点续跑辅助：记录本次失败/跳过的股票，供下次续接。
"""
from __future__ import annotations

import json

from sps.paths import DATA_DIR

LOG_PATH = DATA_DIR / "runs" / "scan_failures.json"

_FAILED_SKIP = {"already_done", "filtered_financial"}


def record_push(failed_key: str, reason: str) -> None:
    """将 symbol 推入下次续跑队列（失败/跳过）。"""
    p = LOG_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            data = {}
    data[failed_key] = reason
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def record_already_done(sym: str) -> None:
    record_push(sym, "already_done")


def get_resume_queue() -> list[str]:
    """返回上次未完成/失败的股票列表（续跑用）。"""
    p = LOG_PATH
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    return [k for k, v in data.items() if v != "already_done"]


def clear_resume_queue() -> None:
    p = LOG_PATH
    if p.exists():
        p.unlink()
