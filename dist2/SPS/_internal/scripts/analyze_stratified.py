"""B阶段分析：读取 events.jsonl，做牛熊分层统计 + RPS 分层，输出对比报告。

用法：
  .venv/Scripts/python scripts/analyze_stratified.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from sps.data import get_index, get_daily, DATA_DIR
from sps.stratify import (build_rps_table, market_regime, rps_series,
                          stratified_report)

RUN_DIR = DATA_DIR / "runs"
DAILY_DIR = DATA_DIR / "daily"


def load_events() -> list[dict]:
    return [json.loads(l) for l in open(RUN_DIR / "events.jsonl", encoding="utf-8")]


def rebuild_stat_rows(events: list[dict]) -> list[dict]:
    """从缓存日线重建每个事件的前向统计（与 run_scan 相同口径）。"""
    rows = []
    cache = {}
    horizons = (5, 10, 20, 60)
    for e in events:
        if e.get("duplicate") or not e.get("signal_date"):
            continue
        sym = e["symbol"]
        f = next(DAILY_DIR.glob(f"{sym}_*.parquet"), None)
        if f is None:
            continue
        if sym not in cache:
            try:
                cache[sym] = pd.read_parquet(f)
            except Exception:  # noqa: BLE001
                cache[sym] = None
        df = cache[sym]
        if df is None or len(df) < 70:
            continue
        ts = pd.Timestamp(e["signal_date"])
        if ts not in df.index:
            continue
        pos = df.index.get_loc(ts)
        # 次日开盘进场（可执行口径）
        ep = None
        epos = None
        for j in range(pos + 1, min(pos + 6, len(df))):
            o = float(df["O"].iloc[j])
            if float(df["V"].iloc[j]) > 0 and not (
                    o >= df["H"].iloc[j] and o > float(df["C"].iloc[pos]) * 1.098):
                ep, epos = o, j
                break
        if ep is None:
            continue
        fwd = {}
        for h in horizons:
            end = epos + h
            if end >= len(df):
                fwd[h] = None
                continue
            seg_c = df["C"].iloc[epos + 1:end + 1]
            seg_l = df["L"].iloc[epos + 1:end + 1]
            seg_h = df["H"].iloc[epos + 1:end + 1]
            if len(seg_c) == 0:
                fwd[h] = None
                continue
            fwd[h] = {"ret": float(seg_c.iloc[-1]) / ep - 1,
                      "mae": float(seg_l.min()) / ep - 1,
                      "mfe": float(seg_h.max()) / ep - 1}
        rows.append({"pattern": e["pattern"], "symbol": sym,
                     "signal_date": e["signal_date"], "fwd": fwd,
                     "entry": ep})
    return rows


def main():
    events = load_events()
    print(f"loaded {len(events)} events")
    idx = get_index("000300", start="20180101")
    regime = market_regime(idx["C"])
    from sps.stratify import attach_regime
    attach_regime(events, regime)

    stat_rows = rebuild_stat_rows(events)
    print(f"rebuilt stats for {len(stat_rows)} executable events")

    # ---- 牛熊分层 ----
    tbl = stratified_report(events, stat_rows)
    print("\n===== 沪深300牛熊分层的形态统计（可执行口径）=====")
    print(tbl.to_string(index=False))
    tbl.to_csv(RUN_DIR / "stats_stratified.csv", index=False)

    # ---- RPS 分层 ----
    # 用已缓存的日线重建全样本 RPS(50)
    data = {}
    for f in DAILY_DIR.glob("*.parquet"):
        sym = f.name.split("_")[0]
        if sym.startswith("index"):
            continue
        try:
            data[sym] = pd.read_parquet(f)
        except Exception:  # noqa: BLE001
            pass
    if data:
        wide_ret = pd.DataFrame({s: rps_series(df, 50) for s, df in data.items()})
        rps_pct = wide_ret.rank(axis=1, pct=True)
        recs2 = []
        by_pat_rps = {}
        for r in stat_rows:
            ev = next((e for e in events
                       if e.get("signal_date") == r["signal_date"]
                       and e.get("symbol") == r["symbol"]), None)
            if not ev:
                continue
            ts = pd.Timestamp(r["signal_date"])
            if ts not in rps_pct.index or r["symbol"] not in rps_pct.columns:
                continue
            v = rps_pct.loc[ts, r["symbol"]]
            if pd.isna(v):
                continue
            bucket = "RPS>=85" if v >= .85 else ("RPS 50-85" if v >= .5
                                                else "RPS<50")
            by_pat_rps.setdefault(r["pattern"], []).append((r, bucket))
        for pat, items in by_pat_rps.items():
            for bucket in ("RPS>=85", "RPS 50-85", "RPS<50"):
                sub = [r for r, b in items if b == bucket]
                for h in (10, 20, 60):
                    rs = [r["fwd"][h]["ret"] for r in sub
                          if r.get("fwd") and r["fwd"].get(h)]
                    if len(rs) < 15:
                        continue
                    arr = np.array(rs)
                    wins, losses = arr[arr > 0], arr[arr <= 0]
                    recs2.append({
                        "pattern": pat, "rps_bucket": bucket,
                        "horizon": f"{h}d", "n": len(arr),
                        "win_rate": round(float((arr > 0).mean()), 3),
                        "mean_ret": round(float(arr.mean()), 4),
                        "median_ret": round(float(np.median(arr)), 4),
                        "pl_ratio": (round(float(wins.mean() /
                                                  abs(losses.mean())), 2)
                                     if len(wins) and len(losses) and
                                     losses.mean() != 0 else None),
                    })
        tbl2 = pd.DataFrame(recs2)
        print("\n===== 信号日 RPS(50) 分层的形态统计 =====")
        print(tbl2.to_string(index=False))
        tbl2.to_csv(RUN_DIR / "stats_by_rps.csv", index=False)


if __name__ == "__main__":
    main()
