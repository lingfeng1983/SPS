"""用户自定义策略的保存/读取。存 data/runs/my_strategies.json"""
from __future__ import annotations

import json
from datetime import date

from sps.paths import DATA_DIR

STRAT_FILE = DATA_DIR / "runs" / "my_strategies.json"


def _load() -> list[dict]:
    if not STRAT_FILE.exists():
        return []
    try:
        return json.loads(STRAT_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []


def _save(items: list[dict]) -> None:
    STRAT_FILE.parent.mkdir(parents=True, exist_ok=True)
    STRAT_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=1),
                          encoding="utf-8")


def list_strategies() -> list[dict]:
    return _load()


def save_strategy(name: str, conditions: dict, stop_pct: float = 7.0) -> dict:
    items = [s for s in _load() if s["name"] != name]   # 同名覆盖
    rec = {"name": name, "conditions": conditions,
           "stop_pct": stop_pct, "created": str(date.today())}
    items.insert(0, rec)
    _save(items[:50])   # 最多50个
    return rec


def delete_strategy(name: str) -> bool:
    items = _load()
    n = len(items)
    items = [s for s in items if s["name"] != name]
    _save(items)
    return len(items) < n
