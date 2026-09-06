"""统一统计口径（规格书第五节）：胜率、收益分布、MAE/MFE、止损模拟。

进场价 A = 信号日后首个可成交日开盘价（主口径）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def entry_price(df: pd.DataFrame, signal_pos: int, max_defer: int = 5):
    """信号日后首个可成交日开盘价。返回 (entry_price, entry_pos) 或 (None, None)。
    可成交 = 非停牌且有成交；一字涨停(开=收=高=低 且涨幅>9.8%)视为不可成交简化处理。
    """
    n = len(df)
    for j in range(signal_pos + 1, min(signal_pos + 1 + max_defer, n)):
        o = float(df["O"].iloc[j])
        c_prev = float(df["C"].iloc[signal_pos])
        v = float(df["V"].iloc[j])
        if v <= 0:            # 停牌
            continue
        if (o >= df["H"].iloc[j] and df["L"].iloc[j] >= o * 0.999
                and o > c_prev * 1.098):   # 一字板近似
            continue
        return o, j
    return None, None


def forward_stats(df: pd.DataFrame, entry_pos: int, A: float,
                  horizons=(5, 10, 20, 60)) -> dict | None:
    """从实际进场日起的 n 日统计。"""
    n_max = len(df)
    res = {}
    for h in horizons:
        end = entry_pos + h
        if end >= n_max:
            res[h] = None
            continue
        seg_c = df["C"].iloc[entry_pos + 1:end + 1]
        seg_l = df["L"].iloc[entry_pos + 1:end + 1]
        seg_h = df["H"].iloc[entry_pos + 1:end + 1]
        if len(seg_c) == 0:
            res[h] = None
            continue
        ret = float(seg_c.iloc[-1]) / A - 1
        mae = float(seg_l.min()) / A - 1
        mfe = float(seg_h.max()) / A - 1
        res[h] = {"ret": ret, "mae": mae, "mfe": mfe}
    return res


def stop_prices(A: float, levels=(-0.05, -0.07, -0.10)) -> dict:
    """返回进场时即可确定的固定止损价，不混入未来退出结果。"""
    return {lv: A * (1 + lv) for lv in levels}


def stop_loss_sim(df: pd.DataFrame, entry_pos: int, A: float,
                  levels=(-0.05, -0.07, -0.10)) -> dict:
    """兼容旧调用；新代码应使用语义明确的 :func:`stop_prices`。"""
    return stop_prices(A, levels)


def stop_exit_sim(df: pd.DataFrame, entry_pos: int, A: float,
                  levels=(-0.05, -0.07, -0.10)) -> dict:
    """模拟各档止损的实际退出价；未触发时返回样本末日收盘价。"""
    out = {lv: None for lv in levels}
    n = len(df)
    for lv in levels:
        stop_px = A * (1 + lv)
        exit_px, exited = None, False
        for j in range(entry_pos + 1, n):
            o, l = float(df["O"].iloc[j]), float(df["L"].iloc[j])
            if l <= stop_px:
                exit_px = min(o, stop_px) if o < stop_px else stop_px
                exited = True
                break
            # 未触发止损则持有到期末（此处模拟持有至数据末尾）
        if not exited:
            exit_px = float(df["C"].iloc[-1])
        out[lv] = exit_px if exit_px else None  # 存实际退出价，非比例
    return out


def aggregate(rows: list[dict], horizons=(5, 10, 20, 60)) -> pd.DataFrame:
    """把每事件的前向统计聚合成形态级报告表。"""
    recs = []
    for h in horizons:
        rs = [r["fwd"][h]["ret"] for r in rows if r.get("fwd") and r["fwd"][h]]
        maes = [r["fwd"][h]["mae"] for r in rows if r.get("fwd") and r["fwd"][h]]
        mfes = [r["fwd"][h]["mfe"] for r in rows if r.get("fwd") and r["fwd"][h]]
        if not rs:
            continue
        arr = np.array(rs)
        wins = arr[arr > 0]
        losses = arr[arr <= 0]
        recs.append({
            "horizon": f"{h}d",
            "n": len(arr),
            "win_rate": round(float((arr > 0).mean()), 4),
            "mean_ret": round(float(arr.mean()), 4),
            "median_ret": round(float(np.median(arr)), 4),
            "p25": round(float(np.percentile(arr, 25)), 4),
            "p75": round(float(np.percentile(arr, 75)), 4),
            "avg_mae": round(float(np.mean(maes)), 4),
            "avg_mfe": round(float(np.mean(mfes)), 4),
            "pl_ratio": (round(float(wins.mean() / abs(losses.mean())), 3)
                         if len(wins) and len(losses) and losses.mean() != 0 else None),
        })
    return pd.DataFrame(recs)
