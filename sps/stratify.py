"""RPS 与牛熊分层的形态统计（B阶段：让"什么条件下有效"显形）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


def market_regime(index_close: pd.Series) -> pd.DataFrame:
    """按规格书五节：牛市=C>MA200且MA200上行；熊市=反向；其余=震荡。
    返回 date->regime 表。"""
    ma200 = index_close.rolling(200, min_periods=200).mean()
    up = (index_close > ma200) & (ma200 > ma200.shift(20))
    dn = (index_close < ma200) & (ma200 < ma200.shift(20))
    regime = pd.Series("range", index=index_close.index)
    regime[up] = "bull"
    regime[dn] = "bear"
    return regime.to_frame("regime")


def attach_regime(events: list[dict], regime: pd.DataFrame) -> None:
    for e in events:
        sd = e.get("signal_date")
        if not sd:
            continue
        ts = pd.Timestamp(sd)
        if ts in regime.index:
            e["regime"] = regime.loc[ts, "regime"]
        else:
            pos = regime.index.searchsorted(ts)
            if 0 < pos <= len(regime):
                e["regime"] = regime.iloc[min(pos, len(regime)) - 1]["regime"]


def rps_series(df: pd.DataFrame, n: int = 50) -> pd.Series:
    """单票 RPS(n) 的近似：n日收益率序列（横截面百分位由调用方在全样本上算）。
    这里先返回每交易日的 n 日涨幅，供跨股票拼接。"""
    C = df["C"]
    return C / C.shift(n) - 1


def build_rps_table(data: dict[str, pd.DataFrame], n: int = 50) -> pd.DataFrame:
    """全样本 RPS：对每个交易日做横截面百分位排名。
    返回 wide 表 (date × symbol)。只用各票自身可见历史（min_periods=n）。"""
    rets = {}
    for sym, df in data.items():
        r = df["C"] / df["C"].shift(n) - 1
        rets[sym] = r
    wide = pd.DataFrame(rets)
    # 横截面百分位（0~1），每日独立
    return wide.rank(axis=1, pct=True)


def stratified_report(events: list[dict], stat_rows: list[dict]) -> pd.DataFrame:
    """按 regime 分层的形态统计表。"""
    return aggregate_by_field(events, stat_rows, "regime")


ENV_FIT = {
    # 平台突破是牛市/震荡品种，需强相对强度；熊市即便强势也补跌
    "FLAT_BREAKOUT": lambda reg, rps: (
        "good" if (reg in ("bull", "range") and rps is not None and rps >= 0.85)
        else "bad" if reg == "bear" else "neutral"),
    # W底是底部反转品种，熊市/震荡更优
    "W_BOTTOM": lambda reg, rps: "good" if reg in ("bear", "range") else "bad",
}


def aggregate_by_field(events: list[dict], stat_rows: list[dict],
                       field: str, horizons=(5, 10, 20, 60)) -> pd.DataFrame:
    """按任意 event 字段（regime / env_fit / rps_bucket ...）做形态分层统计。"""
    recs = []
    by_pat: dict = {}
    for r in stat_rows:
        ev = next((e for e in events
                   if e.get("signal_date") == r["signal_date"]
                   and e.get("symbol") == r["symbol"]), None)
        key = ev.get(field, "?") if ev else "?"
        by_pat.setdefault(r["pattern"], []).append((r, key))
    for pat, items in by_pat.items():
        for val in sorted({v for _, v in items}):
            sub = [r for r, v in items if v == val]
            for h in horizons:
                rs = [r["fwd"][h]["ret"] for r in sub
                      if r.get("fwd") and r["fwd"].get(h)]
                if len(rs) < 10:
                    continue
                arr = np.array(rs)
                wins, losses = arr[arr > 0], arr[arr <= 0]
                recs.append({
                    "pattern": pat, field: val, "horizon": f"{h}d",
                    "n": len(arr),
                    "win_rate": round(float((arr > 0).mean()), 3),
                    "mean_ret": round(float(arr.mean()), 4),
                    "median_ret": round(float(np.median(arr)), 4),
                    "pl_ratio": (round(float(wins.mean() / abs(losses.mean())), 2)
                                 if len(wins) and len(losses)
                                 and losses.mean() != 0 else None),
                })
    return pd.DataFrame(recs)
