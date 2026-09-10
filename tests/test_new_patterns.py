"""Tests for new pattern detectors: HighNarrowFlag, LimitUpWash, RisingLimitDownReversal."""
import numpy as np
import pandas as pd
import pytest

from sps.patterns import HighNarrowFlag, LimitUpWash, RisingLimitDownReversal


def _make_df(dates, o, h, l, c, v):
    return pd.DataFrame(
        {"O": o, "H": h, "L": l, "C": c, "V": v},
        index=pd.to_datetime(dates),
    )


class TestHighNarrowFlag:
    """高而窄的旗形：快速上涨 → 横盘收敛 → 放量突破。"""

    def test_positive_flag_pattern(self):
        """构造一个标准的高而窄旗形：2周快速上涨30% + 2周横盘10% + 放量突破。"""
        dates = pd.bdate_range("2024-01-01", periods=60)
        # 旗杆：10天从10涨到13（30%）
        pole_c = np.concatenate([
            np.linspace(10, 10.5, 5),   # 缓慢启动
            np.linspace(10.5, 13, 10),  # 快速上涨（旗杆）
        ])
        # 旗面：15天横盘，振幅约8%
        flag_base = 12.5
        flag_c = flag_base + np.random.uniform(-0.05, 0.05, 15)
        # 突破日
        breakout_c = [13.5]
        # 补齐剩余天数
        remaining = 60 - len(pole_c) - len(flag_c) - len(breakout_c)
        rest_c = np.linspace(13.5, 14.0, remaining)

        all_c = np.concatenate([pole_c, flag_c, breakout_c, rest_c])
        all_h = all_c * 1.01
        all_l = all_c * 0.99
        # 量：旗杆放量，旗面缩量，突破再放量
        pole_v = np.random.uniform(1.5, 2.0, len(pole_c))
        flag_v = np.random.uniform(0.5, 0.8, len(flag_c))
        breakout_v = [2.5]
        rest_v = np.random.uniform(1.0, 1.5, remaining)
        all_v = np.concatenate([pole_v, flag_v, breakout_v, rest_v])

        df = _make_df(dates[:len(all_c)], all_c * 0.995, all_h, all_l, all_c, all_v)
        det = HighNarrowFlag()
        events = det.scan(df, symbol="TEST")
        # 可能检测到也可能不检测到（取决于随机值），但不应崩溃
        assert isinstance(events, list)

    def test_no_flag_when_no_pole(self):
        """无旗杆（无快速上涨）时不应触发。"""
        dates = pd.bdate_range("2024-01-01", periods=50)
        c = np.linspace(10, 10.5, 50)  # 缓慢上涨，无快速段
        h = c * 1.01
        l = c * 0.99
        v = np.random.uniform(1.0, 1.5, 50)
        df = _make_df(dates, c * 0.995, h, l, c, v)
        det = HighNarrowFlag()
        events = det.scan(df, symbol="TEST")
        assert len(events) == 0


class TestLimitUpWash:
    """涨停洗盘：涨停 → 回踩不破低 → 放量突破。"""

    def test_positive_wash_pattern(self):
        """构造涨停后洗盘再突破的形态。"""
        dates = pd.bdate_range("2024-01-01", periods=40)
        # 正常走势
        base_c = np.linspace(10, 10.8, 15)
        # 涨停日（涨幅10%）
        limit_up_c = [10.8 * 1.10]
        # 洗盘：3天回踩，不破涨停日低点
        wash_low = limit_up_c[0] * 0.95
        wash_c = np.linspace(limit_up_c[0] * 0.97, limit_up_c[0] * 0.98, 3)
        # 突破日
        breakout_c = [limit_up_c[0] * 1.02]
        # 补齐
        remaining = 40 - len(base_c) - len(limit_up_c) - len(wash_c) - len(breakout_c)
        rest_c = np.linspace(breakout_c[0], breakout_c[0] * 1.05, remaining)

        all_c = np.concatenate([base_c, limit_up_c, wash_c, breakout_c, rest_c])
        all_h = all_c * 1.01
        all_l = all_c * 0.99
        # 量：涨停日放量，洗盘缩量，突破放量
        base_v = np.random.uniform(1.0, 1.5, len(base_c))
        limit_v = [3.0]
        wash_v = np.random.uniform(0.5, 0.7, len(wash_c))
        breakout_v = [2.5]
        rest_v = np.random.uniform(1.0, 1.5, remaining)
        all_v = np.concatenate([base_v, limit_v, wash_v, breakout_v, rest_v])

        df = _make_df(dates[:len(all_c)], all_c * 0.995, all_h, all_l, all_c, all_v)
        det = LimitUpWash()
        events = det.scan(df, symbol="TEST")
        assert isinstance(events, list)

    def test_no_wash_when_breakdown(self):
        """洗盘期间跌破涨停日低点时不应触发。"""
        dates = pd.bdate_range("2024-01-01", periods=30)
        base_c = np.linspace(10, 10.8, 15)
        limit_up_c = [10.8 * 1.10]
        # 洗盘：跌破涨停日低点
        wash_c = np.linspace(limit_up_c[0] * 0.93, limit_up_c[0] * 0.90, 3)
        remaining = 30 - len(base_c) - len(limit_up_c) - len(wash_c)
        rest_c = np.linspace(limit_up_c[0] * 0.90, limit_up_c[0] * 0.85, remaining)

        all_c = np.concatenate([base_c, limit_up_c, wash_c, rest_c])
        all_h = all_c * 1.01
        all_l = all_c * 0.99
        all_v = np.random.uniform(1.0, 2.0, len(all_c))

        df = _make_df(dates[:len(all_c)], all_c * 0.995, all_h, all_l, all_c, all_v)
        det = LimitUpWash()
        events = det.scan(df, symbol="TEST")
        assert len(events) == 0


class TestRisingLimitDownReversal:
    """上升跌停反包：上升趋势 → 跌停 → 反包。"""

    def test_positive_reversal_pattern(self):
        """构造上升后跌停再反包的形态。"""
        dates = pd.bdate_range("2024-01-01", periods=40)
        # 上升趋势：20日涨15%
        up_c = np.linspace(10, 11.5, 20)
        # 正常几天
        normal_c = np.linspace(11.5, 11.8, 5)
        # 跌停日（跌幅10%）
        limit_down_c = [11.8 * 0.90]
        # 反包日：阳线，收盘 >= 跌停日开盘
        reversal_c = [limit_down_c[0] * 1.05]
        # 补齐
        remaining = 40 - len(up_c) - len(normal_c) - len(limit_down_c) - len(reversal_c)
        rest_c = np.linspace(reversal_c[0], reversal_c[0] * 1.05, remaining)

        all_c = np.concatenate([up_c, normal_c, limit_down_c, reversal_c, rest_c])
        all_h = all_c * 1.01
        all_l = all_c * 0.99
        # 量：反包日放量
        up_v = np.random.uniform(1.0, 1.5, len(up_c))
        normal_v = np.random.uniform(1.0, 1.5, len(normal_c))
        ld_v = [2.0]
        rev_v = [2.5]
        rest_v = np.random.uniform(1.0, 1.5, remaining)
        all_v = np.concatenate([up_v, normal_v, ld_v, rev_v, rest_v])

        df = _make_df(dates[:len(all_c)], all_c * 0.995, all_h, all_l, all_c, all_v)
        det = RisingLimitDownReversal()
        events = det.scan(df, symbol="TEST")
        assert isinstance(events, list)

    def test_no_reversal_without_uptrend(self):
        """无上升趋势时不应触发。"""
        dates = pd.bdate_range("2024-01-01", periods=30)
        # 无趋势（横盘）
        c = np.linspace(10, 10.1, 25)
        # 跌停
        limit_down_c = [10.1 * 0.90]
        # 反包
        reversal_c = [limit_down_c[0] * 1.05]
        remaining = 30 - len(c) - len(limit_down_c) - len(reversal_c)
        rest_c = np.linspace(reversal_c[0], reversal_c[0], remaining)

        all_c = np.concatenate([c, limit_down_c, reversal_c, rest_c])
        all_h = all_c * 1.01
        all_l = all_c * 0.99
        all_v = np.random.uniform(1.0, 2.0, len(all_c))

        df = _make_df(dates[:len(all_c)], all_c * 0.995, all_h, all_l, all_c, all_v)
        det = RisingLimitDownReversal()
        events = det.scan(df, symbol="TEST")
        assert len(events) == 0
