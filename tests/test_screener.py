"""screener 指标与组合的单元测试（合成K线）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from sps.screener import INDICATORS, _rps_series, screen, entry_and_stop


def make_df(symbol: str = "T1", n: int = 300,
            drift: float = 0.004, vol_noise: float = 0.012,
            base: float = 10.0) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(symbol)) % 2**32)
    ret = rng.normal(drift, vol_noise, n)
    c = base * np.exp(np.cumsum(ret))
    o = c * (1 + rng.normal(0, 0.004, n))
    h = np.maximum(o, c) * (1 + abs(rng.normal(0, 0.006, n)))
    l = np.minimum(o, c) * (1 - abs(rng.normal(0, 0.006, n)))
    v = rng.lognormal(14, 0.4, n) * (1 + (ret > 0) * 0.5)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"O": o, "H": h, "L": l, "C": c, "V": v}, index=idx)


def test_indicator_registered():
    assert len(INDICATORS) == 18
    expected = {"rps50", "above_ma", "near_high", "vol_ratio", "up_days",
                "turnover", "gain_today", "pullback_stable", "box_amp",
                "vol_narrow",
                "mom_win", "rsv_pos", "ma_align", "macd_cross",
                "new_high_cnt", "ma_spread", "amp20", "yang_streak"}
    assert expected <= set(INDICATORS.keys())


def test_each_indicator_runs():
    df = make_df("X1")
    for name, meta in INDICATORS.items():
        s = meta["fn"](df, meta["default"], rps=None)
        if s is None:
            continue
        assert len(s) == len(df), f"{name} length mismatch"
        # 无未来数据：最后一天只依赖历史 → 序列非全False即可跑通
        assert s.dtype == bool


def test_rps_cross_section():
    # 用确定性数据：S5 涨幅严格最大 → RPS 应为最高百分位
    n = 120
    idx = pd.bdate_range("2024-01-01", periods=n)
    dfs = {}
    for i in range(6):
        c = 10 * (1 + 0.01 * i) ** np.arange(n)
        dfs[f"S{i}"] = pd.DataFrame(
            {"O": c, "H": c, "L": c, "C": c, "V": np.ones(n)}, index=idx)
    wide = pd.DataFrame({s: d["C"] for s, d in dfs.items()})
    rps = _rps_series(wide, 50)
    last = rps.iloc[-1]
    assert last["S5"] == 1.0          # 涨幅最大 → 百分位100%
    assert last["S0"] == pytest.approx(1 / 6)  # 涨幅最小 → 最低档


def test_screen_trigger_and_near():
    strong = make_df("STRONG", drift=0.006, vol_noise=0.010)
    weak = make_df("WEAK", drift=-0.002, vol_noise=0.02)
    daily = {"STRONG": strong, "WEAK": weak}
    wide = pd.DataFrame({s: d["C"] for s, d in daily.items()})
    conds = {"above_ma": 20, "vol_ratio": 1.2}
    out = screen(daily, wide, conds)
    all_recs = out["triggered"] + out["near"]
    syms = {r["symbol"] for r in all_recs}
    assert syms <= {"STRONG", "WEAK"}
    for r in all_recs:
        assert 0 <= r["score"] <= 100
        assert r["status"] in ("triggered", "near")


def test_entry_and_stop():
    df = make_df("E1")
    pos = len(df) - 10
    res = entry_and_stop(df, pos, stop_pct=7.0)
    if res is not None:
        o = float(df["O"].iloc[pos + 1])
        assert abs(res["entry_price"] - round(o, 3)) < 0.01
        assert abs(res["stop_price"] - round(o * 0.93, 3)) < 0.01


def test_entry_and_stop_realtime_pending():
    """信号在最后一根K线：返回参考买点(今日收盘)+pending标记。"""
    df = make_df("RT")
    res = entry_and_stop(df, len(df) - 1, stop_pct=7.0)
    assert res is not None
    assert res["pending"] is True
    c = float(df["C"].iloc[-1])
    assert abs(res["entry_price"] - round(c, 3)) < 0.01
    assert abs(res["stop_price"] - round(c * 0.93, 3)) < 0.01


def test_no_future_leak():
    """指标在 t 日的值不应因 t+1 之后的数据改变。"""
    df = make_df("LEAK", n=200)
    for name, meta in INDICATORS.items():
        s_full = meta["fn"](df, meta["default"], rps=None)
        if s_full is None:
            continue
        cut = 150
        s_trunc = meta["fn"](df.iloc[:cut], meta["default"], rps=None)
        if s_trunc is None:
            continue
        a = s_full.iloc[cut - 1]
        b = s_trunc.iloc[-1]
        assert bool(a) == bool(b), f"{name} 泄漏未来数据"
