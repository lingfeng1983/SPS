"""事件模型：规格书第七节输出契约。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict


def make_event_id(symbol: str, pattern: str, signal_date: str, rule_version: str) -> str:
    raw = f"{symbol}|{pattern}|{signal_date}|{rule_version}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class Event:
    symbol: str
    pattern: str
    status: str  # candidate | confirmed | invalidated
    pattern_start: str | None
    pattern_end: str | None
    signal_date: str | None
    available_at: str
    rule_version: str
    params_hash: str = ""
    key_levels: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    co_patterns: list = field(default_factory=list)
    data_quality_flags: list = field(default_factory=list)
    duplicate: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.signal_date:
            d["event_id"] = make_event_id(
                self.symbol, self.pattern, self.signal_date, self.rule_version)
        return d


def params_hash(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
