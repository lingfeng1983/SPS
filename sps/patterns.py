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


# ================================================================

@dataclass
class WBottom(DetectorBase):
    name: str = "W_BOTTOM"
    max_diff_pct: float = 0.03
    min_gap_days: int = 15
    max_gap_days: int = 90
    min_bounce_pct: float = 0.15
    vr_break: float = 1.8

    def scan(self, df: pd.DataFrame, symbol: str = "") -> list[Event]:
        k = self.params.get("k", 5)
        pv = PivotView(df, k)
        atr = wilder_atr_series(df, n=14)
        events = []
        n = len(df)
        for i in range(self.min_gap_days + k + 5, n):
            t = df.index[i]
            lows = pv.as_of(t)
            lows = lows[lows["kind"] == "low"]
            if len(lows) < 2:
                continue
            l2_row = lows.iloc[-1]
            l1_rows = lows.iloc[:-1]
            pos_l2 = df.index.get_loc(l2_row["pivot_date"])
            for _, l1 in l1_rows.iloc[::-1].iterrows():
                pos_l1 = df.index.get_loc(l1["pivot_date"])
                T = pos_l2 - pos_l1
                if not (self.min_gap_days <= T <= self.max_gap_days):
                    continue
                L1v, L2v = float(df.loc[l1["pivot_date"], "L"]), float(df.loc[l2_row["pivot_date"], "L"])
                if abs(L1v - L2v) / ((L1v + L2v) / 2) > self.max_diff_pct:
                    continue
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
                win = df["C"].iloc[max(0, pos_l2 - 3):pos_l2 + 4]
                if (win <= L2v + float(atr.iloc[min(pos_l2, n - 1)])).sum() < 5:
                    continue
                v_l2 = float(df["V"].iloc[max(0, pos_l2 - 2):pos_l2 + 3].mean())
                v_l1 = float(df["V"].iloc[max(0, pos_l1 - 2):pos_l1 + 3].mean())
                if v_l1 <= 0 or v_l2 >= v_l1:
                    continue
                stronger = bool(L2v > L1v)
                feats = {"bounce": round(bounce, 4), "gap_days": T,
                         "right_higher": stronger}
                avail = l2_row["available_at"]
                if i < df.index.get_loc(avail):
                    continue
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
                        break
                break
        return events


# ================================================================

@dataclass
class FlatBreakout(DetectorBase):
    name: str = "FLAT_BREAKOUT"
    min_len: int = 20
    max_len: int = 60
    max_range_pct: float = 0.15
    vr_break: float = 2.0
    rps_min: float | None = None
    min_lookback: int = 130
    pre_rally_pct: float = 0.25
    pre_lookback: int = 120
    pullback_max: float = -0.25
    pullback_lookback: int = 60
    confirm_window: int = 15
    vol_shrink_ratio: float = 0.8

    def scan(self, df: pd.DataFrame, symbol: str = "") -> list[Event]:
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
                if not (vol_ma5.iloc[s:e0] < self.vol_shrink_ratio * vol_ma20.iloc[s:e0]).any():
                    continue
                pre = C.iloc[e0 - 1] / C.iloc[max(0, e0 - self.pre_lookback - 1):e0 - 1].min() - 1
                if pre < self.pre_rally_pct:
                    continue
                if float(seg["L"].min()) / C.iloc[max(0, s - self.pullback_lookback):s].max() - 1 < self.pullback_max:
                    continue
                upper = hi * 1.001 + 0.1 * (hi - lo) * 0.5
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
                                  "prior_rally": round(pre, 4)},
                        symbol=symbol))
                    break
        return events


# ================================================================

@dataclass
class CupHandle(DetectorBase):
    name: str = "CUP_HANDLE"
    min_cup_days: int = 30
    max_cup_days: int = 120
    max_depth: float = 0.30
    min_prior_gain: float = 0.30
    prior_lookback: int = 250
    max_handle_days: int = 21
    max_handle_depth: float = 0.15
    vr_break: float = 1.5
    stop_pct: float = 0.07

    def scan(self, df: pd.DataFrame, symbol: str = "") -> list[Event]:
        k = self.params.get("k", 8)
        pv = PivotView(df, k)
        atr = wilder_atr_series(df, n=14)
        events = []
        n = len(df)
        for i in range(self.prior_lookback + k, n):
            t = df.index[i]
            highs = pv.as_of(t)
            highs = highs[highs["kind"] == "high"]
            if len(highs) < 2:
                continue
            H_top_row = highs.iloc[-1]
            H_top = float(df.loc[H_top_row["pivot_date"], "H"])
            H_pos = df.index.get_loc(H_top_row["pivot_date"])
            prior_lo = float(df["L"].iloc[max(0, H_pos - self.prior_lookback):H_pos].min())
            if prior_lo <= 0:
                continue
            prior_gain = H_top / prior_lo - 1
            if prior_gain < self.min_prior_gain:
                continue
            lows = pv.as_of(t)
            lows = lows[(lows["kind"] == "low") & (lows["pivot_date"] < H_top_row["pivot_date"])]
            if lows.empty:
                continue
            cup_bottom_row = lows.iloc[-1]
            cup_bottom = float(df.loc[cup_bottom_row["pivot_date"], "L"])
            cup_start = H_top_row["pivot_date"]
            cup_win_start = df.index.get_loc(cup_start)
            cup_win_end = df.index.get_loc(cup_bottom_row["pivot_date"])
            cup_days = cup_win_end - cup_win_start
            if not (self.min_cup_days <= cup_days <= self.max_cup_days):
                continue
            depth = 1 - cup_bottom / H_top
            if depth > self.max_depth:
                continue
            handle_start_pos = cup_win_end
            handle_end_pos = min(handle_start_pos + self.max_handle_days, n - 1)
            if handle_start_pos >= handle_end_pos:
                continue
            handle_high = float(df["H"].iloc[handle_start_pos:handle_end_pos + 1].max())
            handle_low = float(df["L"].iloc[handle_start_pos:handle_end_pos + 1].min())
            h_depth = 1 - handle_low / handle_high if handle_high > 0 else 1.0
            if h_depth > self.max_handle_depth:
                continue
            buy_pt = handle_high
            for j in range(handle_start_pos, min(handle_end_pos + 5, n)):
                if df["C"].iloc[j] > buy_pt:
                    vr = volume_ratio(df, j)
                    if vr is not None and vr >= self.vr_break:
                        events.append(self._event(
                            df, "confirmed", df.index[max(0, cup_win_start)], t,
                            t, t,
                            key_levels={"buy_point": round(buy_pt, 3),
                                        "stop": round(buy_pt * (1 - self.stop_pct), 3)},
                            features={"depth": round(depth, 3),
                                      "handle_depth": round(h_depth, 3),
                                      "prior_gain": round(float(H_top / prior_lo - 1), 3)},
                            symbol=symbol))
                        break
        return events


# ================================================================

@dataclass
class PocketPivot(DetectorBase):
    """口袋支点（规格书3.1，Morales/Kacher 定义）。"""
    name: str = "POCKET_PIVOT"
    min_gain: float = 0.02
    max_ext_ma10: float = 0.05
    min_lookback: int = 70
    down_vol_lookback: int = 10
    high_close_ratio: float = 0.4
    range_lookback: int = 65
    max_range: float = 0.25
    max_pullback: float = 0.15
    pullback_lookback: int = 20
    stop_pct: float = 0.07

    def scan(self, df: pd.DataFrame, symbol: str = "") -> list[Event]:
        n = len(df)
        ma50 = sma_series(df["C"], 50)
        ma10 = sma_series(df["C"], 10)
        events = []
        for i in range(self.min_lookback, n):
            o, c, h, l = (float(df["O"].iloc[i]), float(df["C"].iloc[i]),
                          float(df["H"].iloc[i]), float(df["L"].iloc[i]))
            prev_c = float(df["C"].iloc[i - 1])
            if not (c > o and c / prev_c - 1 >= self.min_gain):
                continue
            down_vols = [float(df["V"].iloc[j]) for j in range(i - self.down_vol_lookback, i)
                         if float(df["C"].iloc[j]) < float(df["C"].iloc[j - 1])]
            if not down_vols:
                continue
            if float(df["V"].iloc[i]) <= max(down_vols):
                continue
            if h > l and (h - c) / (h - l) > self.high_close_ratio:
                continue
            m50, m10 = ma50.iloc[i], ma10.iloc[i]
            if pd.isna(m50) or c <= m50:
                continue
            if pd.isna(m10) or abs(c / m10 - 1) > self.max_ext_ma10:
                continue
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


# ================================================================
# 新增形态：高而窄的旗形（欧奈尔《笑傲股市》）
# ================================================================

@dataclass
class HighNarrowFlag(DetectorBase):
    """高而窄的旗形（High and Narrow Flag，欧奈尔经典形态）。

    结构：
    1. 旗杆：快速上涨（>=25%），发生在 2-4 周内，伴随放量
    2. 旗面：横盘收敛，振幅收窄（<=15%），持续 2-4 周，量能萎缩
    3. 突破：放量突破旗面上沿
    """
    name: str = "HIGH_NARROW_FLAG"
    # 旗杆参数
    pole_min_gain: float = 0.25       # 旗杆最低涨幅
    pole_max_days: int = 25           # 旗杆最长交易日
    pole_min_days: int = 5            # 旗杆最短交易日
    # 旗面参数
    flag_min_days: int = 10           # 旗面最短交易日
    flag_max_days: int = 30           # 旗面最长交易日
    flag_max_range: float = 0.15      # 旗面最大振幅
    flag_max_pullback: float = 0.12   # 旗面相对旗杆高点的最大回撤
    # 突破参数
    vr_break: float = 1.5             # 突破日量比
    stop_pct: float = 0.07            # 止损比例

    def scan(self, df: pd.DataFrame, symbol: str = "") -> list[Event]:
        n = len(df)
        if n < 60:
            return []
        C, H, L, V = df["C"], df["H"], df["L"], df["V"]
        vol_ma5 = V.shift(1).rolling(5).mean()
        events = []

        for i in range(40, n):
            # 1. 找旗杆：从当前日回溯，找快速上涨段
            pole_end_price = C.iloc[i]
            # 向前找旗杆起点：涨幅 >= pole_min_gain，且天数在范围内
            pole_start_pos = None
            for j in range(max(0, i - self.pole_max_days), i - self.pole_min_days + 1):
                gain = pole_end_price / C.iloc[j] - 1
                if gain >= self.pole_min_gain:
                    pole_start_pos = j
                    break
            if pole_start_pos is None:
                continue

            pole_days = i - pole_start_pos
            if not (self.pole_min_days <= pole_days <= self.pole_max_days):
                continue

            # 旗杆高点
            pole_high = float(H.iloc[pole_start_pos:i + 1].max())

            # 2. 找旗面：旗杆高点之后的横盘收敛段
            # 旗面起点 = 旗杆高点位置
            flag_start_pos = H.iloc[pole_start_pos:i + 1].idxmax()
            flag_start_idx = df.index.get_loc(flag_start_pos) if isinstance(flag_start_pos, pd.Timestamp) else pole_start_pos

            # 旗面终点 = 当前日（候选突破日）
            flag_end_pos = i
            flag_days = flag_end_pos - flag_start_idx
            if not (self.flag_min_days <= flag_days <= self.flag_max_days):
                continue

            # 旗面振幅检查
            flag_high = float(H.iloc[flag_start_idx:flag_end_pos + 1].max())
            flag_low = float(L.iloc[flag_start_idx:flag_end_pos + 1].min())
            if flag_low <= 0:
                continue
            flag_range = (flag_high - flag_low) / flag_low
            if flag_range > self.flag_max_range:
                continue

            # 旗面相对旗杆高点的回撤
            pullback = 1 - flag_low / pole_high
            if pullback > self.flag_max_pullback:
                continue

            # 旗面量能萎缩
            if flag_days >= 5:
                flag_vol = float(V.iloc[flag_start_idx:flag_end_pos + 1].mean())
                pole_vol = float(V.iloc[pole_start_idx:flag_start_idx + 1].mean())
                if pole_vol <= 0 or flag_vol > pole_vol * 0.8:
                    continue

            # 3. 突破确认：收盘突破旗面上沿 + 放量
            flag_upper = flag_high * 1.001
            if C.iloc[i] <= flag_upper:
                continue
            vr = volume_ratio(df, i)
            if vr is None or vr < self.vr_break:
                continue

            # 确认突破
            buy_point = C.iloc[i]
            stop_price = buy_point * (1 - self.stop_pct)
            events.append(self._event(
                df, "confirmed", df.index[flag_start_idx], df.index[i],
                df.index[i], df.index[i],
                key_levels={"buy_point": round(buy_point, 3),
                            "stop": round(stop_price, 3),
                            "flag_high": round(flag_high, 3)},
                features={"pole_gain": round(pole_end_price / C.iloc[pole_start_idx] - 1, 4),
                          "flag_range": round(flag_range, 4),
                          "flag_days": flag_days,
                          "pullback": round(pullback, 4)},
                symbol=symbol))
        return events


# ================================================================
# 新增形态：涨停洗盘（A股特色）
# ================================================================

@dataclass
class LimitUpWash(DetectorBase):
    """涨停洗盘（A股特色形态）。

    结构：
    1. 涨停日：收盘涨幅 >= 9.8%（近似涨停）
    2. 洗盘：涨停后 2-10 日内，价格回踩但不破涨停日低点，量能萎缩
    3. 确认：放量突破涨停日高点
    """
    name: str = "LIMIT_UP_WASH"
    limit_up_pct: float = 0.098     # 涨停阈值
    wash_min_days: int = 2          # 洗盘最短天数
    wash_max_days: int = 10         # 洗盘最长天数
    wash_max_range: float = 0.10    # 洗盘期间最大振幅
    vr_break: float = 1.5           # 突破日量比
    stop_pct: float = 0.07          # 止损比例

    def scan(self, df: pd.DataFrame, symbol: str = "") -> list[Event]:
        n = len(df)
        if n < 20:
            return []
        C, H, L, V = df["C"], df["H"], df["L"], df["V"]
        events = []

        for i in range(10, n):
            # 1. 找涨停日
            prev_c = C.iloc[i - 1]
            if prev_c <= 0:
                continue
            gain = C.iloc[i] / prev_c - 1
            if gain < self.limit_up_pct:
                continue

            limit_up_high = H.iloc[i]
            limit_up_low = L.iloc[i]
            limit_up_vol = V.iloc[i]
            limit_up_date = df.index[i]

            # 2. 洗盘窗口：涨停后 wash_min_days 到 wash_max_days
            wash_end = min(i + self.wash_max_days, n - 1)
            wash_start = i + self.wash_min_days
            if wash_start > wash_end:
                continue

            # 检查洗盘期间是否破涨停日低点
            wash_low = L.iloc[i + 1:wash_end + 1].min()
            if wash_low < limit_up_low * 0.98:  # 允许2%缓冲
                continue

            # 洗盘振幅检查
            wash_high = H.iloc[i + 1:wash_end + 1].max()
            if limit_up_low > 0:
                wash_range = (wash_high - wash_low) / limit_up_low
                if wash_range > self.wash_max_range:
                    continue

            # 洗盘量能萎缩
            wash_vol = V.iloc[i + 1:wash_end + 1].mean()
            if limit_up_vol > 0 and wash_vol > limit_up_vol * 0.7:
                continue

            # 3. 突破确认：洗盘后放量突破涨停日高点
            for j in range(wash_start, min(wash_end + 5, n)):
                if C.iloc[j] > limit_up_high:
                    vr = volume_ratio(df, j)
                    if vr is not None and vr >= self.vr_break:
                        buy_point = C.iloc[j]
                        stop_price = limit_up_low * 0.98
                        events.append(self._event(
                            df, "confirmed", limit_up_date, df.index[j],
                            df.index[j], df.index[j],
                            key_levels={"buy_point": round(buy_point, 3),
                                        "stop": round(stop_price, 3),
                                        "limit_up_high": round(limit_up_high, 3)},
                            features={"limit_up_gain": round(gain, 4),
                                      "wash_days": j - i,
                                      "wash_range": round(wash_range, 4)},
                            symbol=symbol))
                        break
        return events


# ================================================================
# 新增形态：上升跌停反包（极端情绪修复）
# ================================================================

@dataclass
class RisingLimitDownReversal(DetectorBase):
    """上升跌停反包（极端情绪修复形态）。

    结构：
    1. 上升趋势：近 20 日涨幅 >= 10%
    2. 跌停日：收盘跌幅 >= 9.8%（近似跌停）
    3. 反包：次日或隔日，收盘 >= 开盘且收盘接近跌停日高点
    4. 确认：反包日量能放大
    """
    name: str = "RISING_LIMIT_DOWN_REVERSAL"
    uptrend_min_gain: float = 0.10   # 上升趋势最低涨幅
    uptrend_lookback: int = 20       # 趋势回看天数
    limit_down_pct: float = 0.098    # 跌停阈值
    reversal_window: int = 3         # 反包窗口（跌停后几天内）
    vr_min: float = 1.2              # 反包日最低量比
    stop_pct: float = 0.07           # 止损比例

    def scan(self, df: pd.DataFrame, symbol: str = "") -> list[Event]:
        n = len(df)
        if n < 30:
            return []
        C, H, L, O, V = df["C"], df["H"], df["L"], df["O"], df["V"]
        events = []

        for i in range(self.uptrend_lookback + 1, n):
            # 1. 检查上升趋势
            trend_start = C.iloc[i - self.uptrend_lookback]
            if trend_start <= 0:
                continue
            trend_gain = C.iloc[i - 1] / trend_start - 1
            if trend_gain < self.uptrend_min_gain:
                continue

            # 2. 找跌停日
            prev_c = C.iloc[i - 1]
            if prev_c <= 0:
                continue
            drop = C.iloc[i] / prev_c - 1
            if drop > -self.limit_down_pct:
                continue

            limit_down_open = O.iloc[i]
            limit_down_high = H.iloc[i]
            limit_down_low = L.iloc[i]
            limit_down_date = df.index[i]

            # 3. 反包窗口：跌停后 reversal_window 日内
            for j in range(i + 1, min(i + 1 + self.reversal_window, n)):
                # 反包条件：阳线（收盘 >= 开盘）且收盘接近跌停日高点
                if C.iloc[j] < O.iloc[j]:
                    continue
                # 收盘 >= 跌停日开盘（强反包）或 >= 跌停日最高（超强反包）
                if C.iloc[j] < limit_down_high * 0.97:
                    continue

                # 量能检查
                vr = volume_ratio(df, j)
                if vr is None or vr < self.vr_min:
                    continue

                # 确认反包
                buy_point = C.iloc[j]
                stop_price = limit_down_low * 0.98
                events.append(self._event(
                    df, "confirmed", limit_down_date, df.index[j],
                    df.index[j], df.index[j],
                    key_levels={"buy_point": round(buy_point, 3),
                                "stop": round(stop_price, 3),
                                "limit_down_high": round(limit_down_high, 3)},
                    features={"trend_gain": round(trend_gain, 4),
                              "limit_down_drop": round(drop, 4),
                              "reversal_strength": round(C.iloc[j] / limit_down_high - 1, 4)},
                    symbol=symbol))
                break
        return events


ALL_DETECTORS = [WBottom, FlatBreakout, CupHandle, PocketPivot,
                 HighNarrowFlag, LimitUpWash, RisingLimitDownReversal]
