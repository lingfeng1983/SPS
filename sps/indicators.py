"""技术指标与通用词典（规格书 0.2 / 0.5 节）。

无未来数据原则：
- 所有指标只用 t 及以前数据
- Pivot 可用时点 = p + k
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def wilder_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ATR(14)。"""
    h, l, c = df["H"], df["L"], df["C"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()


def natr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return wilder_atr(df, n) / df["C"]


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


# ---------------------------------------------------------------- Pivots

def pivots(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """返回已确认摆动点表：index=pivot日, kind=high/low, available_at=确认日(p+k)。

    摆动高点：H 高于前后各 k 根的最高价。注意确认日 p+k 收盘后才可用。
    """
    H, L = df["H"].values, df["L"].values
    n = len(df)
    out = []
    for i in range(k, n - k):
        if H[i] > H[i - k:i].max() and H[i] > H[i + 1:i + k + 1].max():
            out.append((df.index[i], "high", df.index[i + k]))
        if L[i] < L[i - k:i].min() and L[i] < L[i + 1:i + k + 1].min():
            out.append((df.index[i], "low", df.index[i + k]))
    return pd.DataFrame(out, columns=["pivot_date", "kind", "available_at"])


class PivotView:
    """按"当日视角"访问摆动点：只暴露 available_at <= t 的点。

    这是防未来数据泄漏的关键封装。
    """

    def __init__(self, df: pd.DataFrame, k: int = 5):
        self.k = k
        self.pv = pivots(df, k)

    def as_of(self, t: pd.Timestamp) -> pd.DataFrame:
        return self.pv[self.pv["available_at"] <= t]


# ---------------------------------------------------------------- 0.5 词典

def is_uptrend(df: pd.DataFrame, t: int) -> bool:
    """C>MA60 且 MA60[t]>MA60[t-20]，t 为整数位置。"""
    if t < 81:
        return False
    c = df["C"].iloc[t]
    ma60 = sma(df["C"], 60)
    return bool(c > ma60.iloc[t] and ma60.iloc[t] > ma60.iloc[t - 20])


def is_downtrend(df: pd.DataFrame, t: int) -> bool:
    if t < 81:
        return False
    c = df["C"].iloc[t]
    ma60 = sma(df["C"], 60)
    return bool(c < ma60.iloc[t] and ma60.iloc[t] < ma60.iloc[t - 20])


def big_bull_body(df: pd.DataFrame, t: int) -> bool:
    """大阳线：实体>=前20日实体绝对值中位数1.5倍 且涨幅>=2%。"""
    if t < 21:
        return False
    o, c = df["O"].iloc[t], df["C"].iloc[t]
    body = (c - o)
    prev = (df["C"].iloc[t - 20:t] - df["O"].iloc[t - 20:t]).abs().median()
    chg = c / df["C"].iloc[t - 1] - 1
    return bool(c > o and prev > 0 and body >= 1.5 * prev and chg >= 0.02)


def volume_ratio(df: pd.DataFrame, t: int, n: int = 20) -> float | None:
    """VR20(t) = V[t]/Vol_MA(n,t-1)，分母不含当日。数据不足返回 None。"""
    if t < n or n == 0:
        return None
    base = df["V"].iloc[t - n:t].mean()
    if base <= 0:
        return None
    return float(df["V"].iloc[t] / base)


def volume_shrink_regression(v: pd.Series) -> bool:
    """量能逐步萎缩：log(V) 时间回归斜率<0，且末5日均量<=首5日80%。"""
    v = v.dropna()
    if len(v) < 10 or (v <= 0).any():
        return False
    x = np.arange(len(v))
    slope = np.polyfit(x, np.log(v.values), 1)[0]
    return bool(slope < 0 and v.iloc[-5:].mean() <= 0.8 * v.iloc[:5].mean())


def rps(all_close_wide: pd.DataFrame, t_idx: pd.Timestamp, n: int) -> pd.Series | None:
    """全市场 RPS(n)：n 日涨幅百分位。all_close_wide 为宽表(index=date, cols=symbol)。
    使用截至 t_idx 的可见股票池（列在该日前已有 >=n 行数据的股票）。
    返回 symbol->percentile。实现于扫描层调用。"""
    w = all_close_wide.loc[:t_idx]
    if len(w) < n + 1:
        return None
    ret = w.iloc[-1] / w.iloc[-1 - n] - 1
    valid = w.count() > n  # 该日前已有足够历史
    ret = ret[valid]
    if ret.empty:
        return None
    return ret.rank(pct=True)


def true_gap_up(unadj: pd.DataFrame, t: int, pct: float = 0.02) -> bool:
    """不复权真实向上跳空：L[t] > H[t-1]*(1+pct)。unadj 为不复权OHLC。"""
    if t < 1:
        return False
    return bool(unadj["L"].iloc[t] > unadj["H"].iloc[t - 1] * (1 + pct))
