"""形态识别引擎。

设计：每个形态是一个 Detector，scan(df) 返回 Event 列表。
所有检测器遵守：
- Pivot 只用 PivotView.as_of(t)（防 p+k 前泄漏）
- 确认条件依赖 t 日收盘/量 → signal_date=t，可执行进场为 t+1 开盘（统计层处理）
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .events import Event, params_hash
from .indicators import (PivotView, is_uptrend, volume_ratio,
                         volume_shrink_regression, wilder_atr)


def wilder_atr_series(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ATR — 唯一入口，所有形态必须调用此函数，禁止内联重算。"""
    return wilder_atr(df, n)


def sma_series(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()

RULE_VERSION = "1.1"


@dataclass
class DetectorBase:
    name: str = "BASE"
    params: dict = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        self.phash = params_hash({**self.__dict__, "params": self.params})

    def _event(self, df, status, pat_start, pat_end, sig_date, avail, **kw):
        return Event(
            symbol=kw.pop("symbol", ""), pattern=self.name, status=status,
            pattern_start=str(pat_start.date()) if pat_start is not None else None,
            pattern_end=str(pat_end.date()) if pat_end is not None else None,
            signal_date=str(sig_date.date()) if sig_date is not None else None,
            available_at=str(avail.date()), rule_version=RULE_VERSION,
            params_hash=self.phash, **kw)


# ================================================================ W_BOTTOM

@dataclass
class WBottom(DetectorBase):
    name: str = "W_BOTTOM"
    max_diff_pct: float = 0.03      # 两低价差
    min_gap_days: int = 15
    max_gap_days: int = 90
    min_bounce_pct: float = 0.15
    vr_break: float = 1.8

    def scan(self, df: pd.DataFrame, symbol: str = "") -> list[Event]:
        k = self.params.get("k", 5)
        pv = PivotView(df, k)
        # ATR 唯一入口：禁止内联重算（规格书 0.5 节纪律）
        atr = wilder_atr_series(df, n=14)
        events = []
        lows_all = pv.pv[pv.pv["kind"] == "low"]
        n = len(df)
        # 遍历确认候选日：从第2个低点可用后开始
        # min_gap_days=两底最小间隔, k=Pivot延迟, 5=第二底构筑最小窗口
        for i in range(self.min_gap_days + k + 5, n):
            t = df.index[i]
            lows = pv.as_of(t)
            lows = lows[lows["kind"] == "low"]
            if len(lows) < 2:
                continue
            l2_row = lows.iloc[-1]
            l1_rows = lows.iloc[:-1]
            # 找 L1：距 L2 间隔在区间内的最近低点
            pos_l2 = df.index.get_loc(l2_row["pivot_date"])
            for _, l1 in l1_rows.iloc[::-1].iterrows():
                pos_l1 = df.index.get_loc(l1["pivot_date"])
                T = pos_l2 - pos_l1
                if not (self.min_gap_days <= T <= self.max_gap_days):
                    continue
                L1v, L2v = float(df.loc[l1["pivot_date"], "L"]), float(df.loc[l2_row["pivot_date"], "L"])
                if abs(L1v - L2v) / ((L1v + L2v) / 2) > self.max_diff_pct:
                    continue
                # 中间反弹高点 M：两低点之间的已确认摆动高点
                highs = pv.as_of(t)
                highs = highs[(highs["kind"] == "high") &
                              (highs["pivot_date"] > l1["pivot_date"]) &
                              (highs["pivot_date"] < l2_row["pivot_date"])]
                if highs.empty:
                    continue
                m_date = highs.sort_values("pivot_date").iloc[-1]["pivot_date"]
                M = float(df.loc[m_date, "H"])
                m_pos = df.index.get_loc(m_date)
                bounce = M / min(L1v, L2v) - 1
                if bounce < self.min_bounce_pct or M - min(L1v, L2v) < 2 * atr.iloc[m_pos]:
                    continue
                # 第二底构筑>=5日：±3日窗口内至少5日收盘处于 L2+1ATR 内（非插针）
                win = df["C"].iloc[max(0, pos_l2 - 3):pos_l2 + 4]
                if (win <= L2v + float(atr.iloc[min(pos_l2, n - 1)])).sum() < 5:
                    continue
                # 第二底缩量：±2日日均量低于第一底同口径
                v_l2 = float(df["V"].iloc[max(0, pos_l2 - 2):pos_l2 + 3].mean())
                v_l1 = float(df["V"].iloc[max(0, pos_l1 - 2):pos_l1 + 3].mean())
                if v_l1 <= 0 or v_l2 >= v_l1:
                    continue
                stronger = bool(L2v > L1v)
                feats = {"bounce": round(bounce, 4), "gap_days": T,
                         "right_higher": stronger}
                avail = l2_row["available_at"]
                if i < df.index.get_loc(avail):
                    continue  # 低点尚未确认
                # 确认：突破颈线 M 且放量
                if df["C"].iloc[i] > M:
                    vr = volume_ratio(df, i)
                    if vr is not None and vr >= self.vr_break:
                        stop = float(L2v - 1 * atr.iloc[i])
                        events.append(self._event(
                            df, "confirmed", l1["pivot_date"], t, t, t,
                            key_levels={"neckline": round(M, 3),
                                        "invalidation": round(M - 1 * atr.iloc[i], 3),
                                        "stop": round(stop, 3)},
                            features=feats, symbol=symbol))
                        break  # 该确认日只报一次
                break  # 已找到有效结构配对，更老的 L1 不再考察
        return events


# ================================================================ FLAT_BREAKOUT

@dataclass
class FlatBreakout(DetectorBase):
    name: str = "FLAT_BREAKOUT"
    min_len: int = 20          # 平台最短持续交易日
    max_len: int = 60          # 平台最长持续交易日
    max_range_pct: float = 0.15  # 平台内最大振幅（高-低）/低
    vr_break: float = 2.0      # 突破日量比下限
    rps_min: float | None = None  # RPS 由扫描层注入，单票模式跳过
    # 以下魔法数字提成命名参数（原硬编码在 scan 内）
    min_lookback: int = 130    # 最少历史K线根数（含250日新高回看+MA20预热）
    pre_rally_pct: float = 0.25  # 平台前上涨段最低涨幅
    pre_lookback: int = 120    # 平台前上涨段回看日数
    pullback_max: float = -0.25  # 平台内相对前段高点的最大回撤
    pullback_lookback: int = 60  # 回撤计算时向前看多少日
    confirm_window: int = 15   # 平台结束后确认突破的最大等待日
    vol_shrink_ratio: float = 0.8  # 平台内缩量阈值（MA5/MA20）

    def scan(self, df: pd.DataFrame, symbol: str = "") -> list[Event]:
        """先固定平台区间 [s, s+len)，再在其后 confirm_window 日内找放量突破日。"""
        C, H, L, V = df["C"], df["H"], df["L"], df["V"]
        vol_ma20 = V.shift(1).rolling(20).mean()
        vol_ma5 = V.shift(1).rolling(5).mean()
        n = len(df)
        events = []
        for s in range(self.min_lookback, n - self.min_len - 2):
            for length in (self.min_len, self.min_len + 10, self.max_len):
                e0 = s + length
                if e0 >= n:
                    continue
                seg = df.iloc[s:e0]
                hi, lo = float(seg["H"].max()), float(seg["L"].min())
                rng = (hi - lo) / lo
                if rng > self.max_range_pct:
                    continue
                # 平台内缩量回调出现过
                if not (vol_ma5.iloc[s:e0] < self.vol_shrink_ratio * vol_ma20.iloc[s:e0]).any():
                    continue
                # 平台前上涨段 >= pre_rally_pct（pre_lookback 日回看）
                pre = C.iloc[e0 - 1] / C.iloc[max(0, e0 - self.pre_lookback - 1):e0 - 1].min() - 1
                if pre < self.pre_rally_pct:
                    continue
                # 平台内回撤不超过前段涨幅 pullback_max
                if float(seg["L"].min()) / C.iloc[max(0, s - self.pullback_lookback):s].max() - 1 < self.pullback_max:
                    continue
                upper = hi * 1.001 + 0.1 * (hi - lo) * 0.5
                # 确认窗口：平台结束后 confirm_window 个交易日内首次放量收盘上穿
                for t in range(e0, min(e0 + self.confirm_window, n)):
                    if C.iloc[t] <= upper:
                        continue
                    vr = volume_ratio(df, t)
                    if vr is None or vr < self.vr_break:
                        continue
                    mid = (hi + lo) / 2
                    events.append(self._event(
                        df, "confirmed", df.index[s], df.index[t],
                        df.index[t], df.index[t],
                        key_levels={"breakout": round(upper, 3),
                                    "mid": round(mid, 3),
                                    "stop": round(float(C.iloc[t]) * 0.93, 3)},
                        features={"range_pct": round(rng, 4),
                                  "length": length,
                                  "pre_gain": round(float(pre), 3)},
                        symbol=symbol))
                    break  # 该平台只报首个确认日
                break  # 找到更长平台即停（优先长平台）
        return events


# ================================================================ CUP_HANDLE

@dataclass
class CupHandle(DetectorBase):
    """杯柄形态（规格书1.4，参数按欧奈尔）。"""
    name: str = "CUP_HANDLE"
    min_cup_days: int = 35
    max_cup_days: int = 250
    min_depth: float = 0.12
    max_depth: float = 0.33
    min_prior_gain: float = 0.30
    # 魔法数字提成命名参数
    min_lookback: int = 20      # 最少历史K线（含Pivot延迟+确认窗口）
    max_handle_days: int = 25   # 柄区最大日数（预筛选，精确检查用 min_handle_days）
    min_handle_days: int = 5    # 柄区最短日数
    max_handle_depth: float = 0.12  # 柄区最大回调深度
    vr_breakout: float = 2.0    # 突破日量比下限
    handle_vol_mult: float = 1.4    # 突破日量/柄区均量下限
    stop_pct: float = 0.07      # 止损比例
    prior_lo_lookback: tuple = (20, 120)  # 左侧涨幅回看窗口（近端, 远端）

    def scan(self, df: pd.DataFrame, symbol: str = "") -> list[Event]:
        k = self.params.get("k", 8)   # 长基底用大k
        pv = PivotView(df, k)
        atr = wilder_atr_series(df)
        n = len(df)
        events = []
        highs_all = pv.pv[pv.pv["kind"] == "high"]
        for i in range(self.min_cup_days + 2 * k + self.min_lookback, n):
            t = df.index[i]
            vis = pv.as_of(t)
            hs = vis[vis["kind"] == "high"]
            if hs.empty:
                continue
            # 杯右沿 = 最近已确认高点
            rim_row = hs.iloc[-1]
            pos_rim = df.index.get_loc(rim_row["pivot_date"])
            if i - pos_rim < k or i - pos_rim > self.max_handle_days:
                continue  # 右沿之后需在柄区
            H_top = float(df.loc[rim_row["pivot_date"], "H"])
            # 左侧涨幅：H_top 前 20~120 日内最低点起涨 >=30%
            lo_win = df["L"].iloc[max(0, pos_rim - self.prior_lo_lookback[1]):max(1, pos_rim - self.prior_lo_lookback[0])]
            if lo_win.empty:
                continue
            prior_lo = float(lo_win.min())
            if H_top / prior_lo - 1 < self.min_prior_gain:
                continue
            # 向前找杯底：右沿之前 min~max 日内的最低低点
            cup_win_start = pos_rim - self.max_cup_days
            cup_win_end = pos_rim - 10
            if cup_win_start < 0:
                continue
            seg = df["L"].iloc[cup_win_start:cup_win_end]
            if seg.empty:
                continue
            bottom_px = float(seg.min())
            depth = (H_top - bottom_px) / H_top
            if not (self.min_depth <= depth <= self.max_depth):
                continue
            cup_len = int((df.index[cup_win_end] - df.index[cup_win_start]).days * 0.7)
            # 柄区：右沿之后至今，回调深度<=杯深一半且<=max_handle_depth
            handle = df.iloc[pos_rim:i]
            if len(handle) < self.min_handle_days or len(handle) > self.max_handle_days:
                continue
            h_low = float(handle["L"].min())
            h_depth = (H_top - h_low) / H_top
            if h_depth > depth / 2 or h_depth > self.max_handle_depth:
                continue
            if h_low < (bottom_px + H_top) / 2:
                continue  # 柄低于杯体中点 → 失败形态
            # 确认：收盘突破柄区最高价 + 双重量能条件
            buy_pt = float(handle["H"].max())
            if df["C"].iloc[i] > buy_pt:
                vr = volume_ratio(df, i)
                v_handle_base = float(handle["V"].mean())
                if (vr is not None and vr >= self.vr_breakout and v_handle_base > 0
                        and float(df["V"].iloc[i]) >= self.handle_vol_mult * v_handle_base):
                    events.append(self._event(
                        df, "confirmed", df.index[max(0, cup_win_start)], t,
                        t, t,
                        key_levels={"buy_point": round(buy_pt, 3),
                                    "stop": round(buy_pt * (1 - self.stop_pct), 3)},
                        features={"depth": round(depth, 3),
                                  "handle_depth": round(h_depth, 3),
                                  "prior_gain": round(float(H_top / prior_lo - 1), 3)},
                        symbol=symbol))
        return events


# ================================================================ POCKET_PIVOT

@dataclass
class PocketPivot(DetectorBase):
    """口袋支点（规格书3.1，Morales/Kacher 定义）。"""
    name: str = "POCKET_PIVOT"
    min_gain: float = 0.02
    max_ext_ma10: float = 0.05
    # 魔法数字提成命名参数
    min_lookback: int = 70     # 最少历史K线（含MA50预热+蓄势回看）
    down_vol_lookback: int = 10  # 下跌日量能回看窗口
    high_close_ratio: float = 0.4  # (H-C)/(H-L) 最大允许值
    range_lookback: int = 65   # 蓄势回看窗口
    max_range: float = 0.25    # 蓄势区间最大振幅
    max_pullback: float = 0.15 # 上升趋势中最大回调
    pullback_lookback: int = 20  # 回调计算窗口
    stop_pct: float = 0.07     # 止损比例

    def scan(self, df: pd.DataFrame, symbol: str = "") -> list[Event]:
        n = len(df)
        ma50 = sma_series(df["C"], 50)
        ma10 = sma_series(df["C"], 10)
        events = []
        for i in range(self.min_lookback, n):
            o, c, h, l = (float(df["O"].iloc[i]), float(df["C"].iloc[i]),
                          float(df["H"].iloc[i]), float(df["L"].iloc[i]))
            prev_c = float(df["C"].iloc[i - 1])
            # 当日阳线且涨幅>=min_gain
            if not (c > o and c / prev_c - 1 >= self.min_gain):
                continue
            # 核心条件：当日量 > 前 down_vol_lookback 日所有下跌日最大量
            down_vols = [float(df["V"].iloc[j]) for j in range(i - self.down_vol_lookback, i)
                         if float(df["C"].iloc[j]) < float(df["C"].iloc[j - 1])]
            if not down_vols:
                continue
            if float(df["V"].iloc[i]) <= max(down_vols):
                continue
            # 收盘接近日内高点：(H-C)/(H-L)<=high_close_ratio
            if h > l and (h - c) / (h - l) > self.high_close_ratio:
                continue
            # 前提过滤：C>MA50；距MA10<=max_ext_ma10
            m50, m10 = ma50.iloc[i], ma10.iloc[i]
            if pd.isna(m50) or c <= m50:
                continue
            if pd.isna(m10) or abs(c / m10 - 1) > self.max_ext_ma10:
                continue
            # 蓄势前提：此前 range_lookback 日区间振幅<=max_range 或上升趋势中回调<=max_pullback
            win = df["C"].iloc[i - self.range_lookback:i]
            rng = float(win.max()) / float(win.min()) - 1
            pullback = 1 - c / float(df["C"].iloc[i - self.pullback_lookback:i].max())
            if not (rng <= self.max_range or (c > m50 and pullback <= self.max_pullback)):
                continue
            events.append(self._event(
                df, "confirmed", df.index[i - self.down_vol_lookback], df.index[i],
                df.index[i], df.index[i],
                key_levels={"entry": round(c, 3),
                            "stop": round(c * (1 - self.stop_pct), 3),
                            "ma10": round(float(m10), 3)},
                features={"gain": round(float(c / prev_c - 1), 4),
                          "range_65d": round(rng, 3)},
                symbol=symbol))
        return events


ALL_DETECTORS = [WBottom, FlatBreakout, CupHandle, PocketPivot]
