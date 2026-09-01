from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sps.patterns import CupHandle, PocketPivot

IDX = pd.bdate_range("2020-01-01", periods=500)


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


def make_df_det(closes, opens=None, highs=None, lows=None, vols=None):
    """Deterministic OHLCV builder."""
    c = np.asarray(closes, dtype=float)
    n = len(c)
    if opens is None:
        opens = c * 1.001
    if highs is None:
        highs = np.maximum(opens, c) * 1.005
    if lows is None:
        lows = np.minimum(opens, c) * 0.995
    if vols is None:
        vols = np.full(n, 1e7)
    return pd.DataFrame({"O": opens, "H": highs, "L": lows, "C": c, "V": vols},
                        index=IDX[:n])


def synth_cup_handle():
    """构造杯柄形态。

    结构（确保 pos_rim > 250 使 cup_win_start >= 0）:
    - 0-250: 缓慢上涨 30→33 (避免前期太低拉低 cup 底)
    - 250-290: 杯左下跌 33→24 (深度 27.3%)
    - 290-305: 杯底横盘 24→25
    - 305-325: 杯右回升 25→33 (rim at pos 325)
    - 326-335: 柄回调 33→31 (深度 6.1%)
    - 336-350: 突破 31→36
    """
    n = 400
    c = np.zeros(n)
    c[:251] = np.linspace(30, 33, 251)
    c[251:291] = np.linspace(33, 24, 40)
    c[291:306] = np.linspace(24, 25, 15)
    c[306:326] = np.linspace(25, 33, 20)
    c[326:336] = np.linspace(33, 31, 10)
    c[336:351] = np.linspace(31, 36, 15)
    c[351:] = np.linspace(36, 38, n - 351)

    df = make_df(c)
    # 突破日放量 (i=340 附近, pos_rim=325, handle len=15)
    df.iloc[338:348, df.columns.get_loc("V")] *= 4.0
    # 柄区量能低
    df.iloc[326:336, df.columns.get_loc("V")] *= 0.4
    return df


def test_cup_handle_positive():
    df = synth_cup_handle()
    evs = CupHandle().scan(df, symbol="TEST")
    assert evs, "合成杯柄应产生确认事件"
    e = evs[-1]
    assert e.pattern == "CUP_HANDLE" and e.status == "confirmed"
    assert e.signal_date is not None


def test_cup_handle_negative_downtrend():
    """持续下跌无法形成杯柄。"""
    df = make_df(list(np.linspace(50, 20, 300)))
    evs = CupHandle().scan(df, symbol="TEST")
    assert not [e for e in evs if e.status == "confirmed"]


def test_cup_handle_negative_no_prior_gain():
    """杯右沿之前没有 30% 上涨 → 无法确认。"""
    n = 400
    c = np.zeros(n)
    c[:251] = np.linspace(30, 28, 251)  # 微跌
    c[251:291] = np.linspace(28, 22, 40)
    c[291:306] = np.linspace(22, 23, 15)
    c[306:326] = np.linspace(23, 28, 20)
    c[326:336] = np.linspace(28, 26, 10)
    c[336:351] = np.linspace(26, 29, 15)
    c[351:] = np.linspace(29, 30, n - 351)
    df = make_df(c)
    df.iloc[338:348, df.columns.get_loc("V")] *= 4.0
    evs = CupHandle().scan(df, symbol="TEST")
    assert not [e for e in evs if e.status == "confirmed"]


def synth_pocket_pivot():
    """构造口袋支点：MA50 上方蓄势 → 放量阳线 2.9% → 收盘近高。

    结构:
    - 0-69: 上涨 20→35 (强趋势，确保 MA50 远低于当前价)
    - 70-99: 横盘 35±0.5 (蓄势)
    - 100-114: 回调 35.5→33 (不破 MA50)
    - 115-118: 微涨 33→34
    - 119: 口袋支点日 (O=34.0, C=35.0, H=35.2, L=33.9, V=2.5e7)
    """
    n = 120
    c = np.zeros(n)
    c[:70] = np.linspace(20, 35, 70)
    c[70:100] = 35 + 0.5 * np.sin(np.arange(30) * 0.5)
    c[100:115] = np.linspace(35.5, 33, 15)
    c[115:119] = np.linspace(33, 34, 4)
    c[119] = 35.0

    o = c.copy()
    h = c.copy()
    l = c.copy()
    v = np.full(n, 1e7)

    o[119] = 34.0
    c[119] = 35.0      # 涨 2.9%
    h[119] = 35.2      # 接近高点
    l[119] = 33.9      # 下影线短
    v[119] = 2.5e7     # 放量

    down_days = [110, 111, 113, 116]
    for j in down_days:
        c[j] = c[j+1] - 0.2
        v[j] = 3e6

    c[118] = 34.0
    o[118] = 33.9

    df = make_df_det(c, o, h, l, v)
    return df, 119


def test_pocket_pivot_positive():
    df, pos = synth_pocket_pivot()
    evs = PocketPivot().scan(df, symbol="TEST")
    confirmed = [e for e in evs if e.status == "confirmed"]
    assert confirmed, f"合成口袋支点应产生确认事件（共 {len(evs)} 个候选）"
    e = confirmed[-1]
    assert e.pattern == "POCKET_PIVOT"
    assert e.signal_date is not None
    assert "entry" in e.key_levels
    assert "stop" in e.key_levels


def test_pocket_pivot_negative_below_ma50():
    """价格持续在 MA50 下方 → 不可能触发口袋支点。"""
    rng = np.random.default_rng(13)
    c = 20 * np.exp(np.cumsum(rng.normal(-0.001, 0.01, 150)))
    df = make_df(c, seed=13)
    evs = PocketPivot().scan(df, symbol="TEST")
    assert not [e for e in evs if e.status == "confirmed"]


def test_pocket_pivot_negative_low_volume():
    """量能不足（没有超过前10日下跌日量）→ 不触发。"""
    df, pos = synth_pocket_pivot()
    df.iloc[pos, df.columns.get_loc("V")] = 1e6
    evs = PocketPivot().scan(df, symbol="TEST")
    confirmed = [e for e in evs if e.status == "confirmed"]
    assert not confirmed
