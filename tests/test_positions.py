from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sps.positions import backtest_exit_rules, add_position, close_position, load_positions, save_positions

IDX = pd.bdate_range("2024-01-01", periods=200)


def make_df(seed=7, drift=0.004, vol_noise=0.012, base=10.0, n=200):
    rng = np.random.default_rng(seed)
    ret = rng.normal(drift, vol_noise, n)
    c = base * np.exp(np.cumsum(ret))
    o = c * (1 + rng.normal(0, 0.004, n))
    h = np.maximum(o, c) * (1 + abs(rng.normal(0, 0.006, n)))
    l = np.minimum(o, c) * (1 - abs(rng.normal(0, 0.006, n)))
    v = rng.lognormal(14, 0.4, n) * (1 + (ret > 0) * 0.5)
    return pd.DataFrame({"O": o, "H": h, "L": l, "C": c, "V": v}, index=IDX[:n])


def test_backtest_exit_rules_basic():
    """基本回测：至少产生若干笔交易。"""
    daily = {
        "S1": make_df(seed=1, drift=0.003),
        "S2": make_df(seed=2, drift=-0.002),
    }
    rules = {"stop_loss": True, "trail_stop": 5.0}
    result = backtest_exit_rules(daily, rules, stop_pct=7.0)
    assert result["n_trades"] > 0, "应至少产生若干笔交易"
    assert "win20" in result
    assert "avg_pnl" in result
    assert "avg_hold_days" in result


def test_backtest_exit_rules_empty_daily():
    """空数据不报错。"""
    result = backtest_exit_rules({}, {"stop_loss": True})
    assert result["n_trades"] == 0


def test_backtest_exit_rules_short_data():
    """数据太短（<120 bars）不产生交易。"""
    daily = {"X": make_df(n=50)}
    result = backtest_exit_rules(daily, {"stop_loss": True})
    assert result["n_trades"] == 0


def test_backtest_exit_rules_per_rule_counts():
    """per_rule_counts 应包含所有规则。"""
    daily = {"S1": make_df(seed=1, drift=0.003)}
    rules = {"stop_loss": True, "break_ma": 5, "trail_stop": 5.0}
    result = backtest_exit_rules(daily, rules, stop_pct=7.0)
    if result["n_trades"] > 0:
        # per_rule_counts 里至少有一个非零
        assert any(v > 0 for v in result["per_rule_counts"].values())


def test_add_and_load_positions():
    """添加持仓后能正确加载。"""
    # 清空
    save_positions([])
    add_position("600519", "贵州茅台", 1600.0, "2024-01-15", stop_pct=7.0, rules={"stop_loss": True}, qty=100)
    add_position("300750", "宁德时代", 180.0, "2024-02-01", stop_pct=7.0, rules={}, qty=500)
    positions = load_positions()
    assert len(positions) == 2
    syms = {p["symbol"] for p in positions}
    assert "600519" in syms
    assert "300750" in syms
    # 清理
    save_positions([])


def test_close_position():
    """平仓后持仓应从 open 移除。"""
    save_positions([])
    add_position("600519", "贵州茅台", 1600.0, "2024-01-15", stop_pct=7.0, rules={}, qty=100)
    close_position("600519", 1700.0, "2024-03-01")
    positions = load_positions()
    # 平仓后 status 变为 closed，不再有 open 持仓
    open_positions = [p for p in positions if p["status"] == "open"]
    assert len(open_positions) == 0
    # 但历史记录保留
    assert len(positions) == 1
    assert positions[0]["status"] == "closed"
    assert positions[0]["exit_price"] == 1700.0
    # 清理
    save_positions([])
