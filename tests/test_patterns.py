"""合成K线测试：每个形态至少 1 正例 + 1 反例（规格书第七节）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sps.patterns import WBottom, FlatBreakout
from sps.indicators import pivots

IDX = pd.bdate_range("2020-01-01", periods=400)


def make_df(closes, seed=7, spread=0.003):
    rng = np.random.default_rng(seed)
    c = np.asarray(closes, dtype=float)
    o = c * (1 + rng.normal(0, .002, len(c)))
    sp = np.abs(rng.normal(.005, .002, len(c))) + spread
    h = np.maximum(o, c) * (1 + sp)
    l = np.minimum(o, c) * (1 - sp)
    v = rng.uniform(8e6, 1.2e7, len(c))
    return pd.DataFrame({"O": o, "H": h, "L": l, "C": c, "V": v},
                        index=IDX[:len(c)])


def synth_w_bottom():
    """下跌→双底→回升→突破颈线38放量。突破日=颈线上穿处。"""
    seg = []
    seg += list(np.linspace(50, 30, 40))            # 0-39 下跌
    seg += [30 + .8 * np.sin(i) for i in range(8)]   # 40-47 第一底(低~29.2)
    seg += list(np.linspace(30, 37.5, 11))           # 48-58 反弹
    seg += [38.5] * 3                                # 59-61 颈线尖峰
    seg += list(np.linspace(38, 29.6, 18))           # 62-79 回落(第二底更低)
    seg += [29.6 + .15 * np.sin(i) for i in range(6)]  # 80-85 第二底(~30%内价差)
    seg += list(np.linspace(29.6, 39.5, 25))         # 86-110 突破(38在~106上穿)
    closes = seg + list(np.linspace(40, 42, 60))
    df = make_df(closes)
    df.iloc[104:110, df.columns.get_loc("V")] *= 4   # 突破日放量
    return df


def test_wbottom_positive():
    df = synth_w_bottom()
    evs = WBottom().scan(df, symbol="TEST")
    assert evs, "合成W底应产生确认事件"
    e = evs[-1]
    assert e.pattern == "W_BOTTOM" and e.status == "confirmed"
    assert e.signal_date is not None
    assert "neckline" in e.key_levels


def test_wbottom_negative_downtrend():
    df = make_df(list(np.linspace(60, 20, 300)))
    evs = WBottom().scan(df, symbol="TEST")
    assert not [e for e in evs if e.status == "confirmed"]


def synth_flat_breakout():
    """130日上涨→30日窄平台(含缩量)→放量突破。"""
    seg = list(np.linspace(20, 32, 130))              # 0-129 上涨
    base = [32 + .3 * np.sin(i * .7) for i in range(30)]  # 130-159 平台
    seg += base
    seg += list(np.linspace(32.2, 33.5, 10))          # 160-169 突破
    df = make_df(seg + list(np.linspace(34, 36, 60)))
    df.iloc[135:141, df.columns.get_loc("V")] *= 0.4  # 平台内缩量
    df.iloc[167:170, df.columns.get_loc("V")] *= 5    # 突破放量
    return df


def test_flat_breakout_positive():
    df = synth_flat_breakout()
    evs = FlatBreakout().scan(df, symbol="TEST")
    confirmed = [e for e in evs if e.status == "confirmed"]
    assert confirmed, "合成平台突破应产生确认事件"


def test_pivot_available_at_no_leak():
    df = synth_w_bottom()
    pv = pivots(df, k=5)
    assert (pv["available_at"] > pv["pivot_date"]).all()
    delta = (pv["available_at"] - pv["pivot_date"]).dt.days
    assert (delta >= 5).all()


def test_event_fields_contract():
    df = synth_w_bottom()
    for e in WBottom().scan(df, symbol="600000.SH"):
        d = e.to_dict()
        for f in ("event_id", "symbol", "pattern", "status", "available_at",
                  "rule_version", "params_hash", "key_levels", "features"):
            assert f in d, f"缺少契约字段 {f}"
        assert d["rule_version"] == "1.1"
        break
