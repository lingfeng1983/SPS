"""交叉过滤验证：regime × RPS 组合后的形态统计（B阶段收尾）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from sps.data import get_index, DATA_DIR
from sps.stratify import attach_regime, build_rps_table, rps_series

RUN_DIR = DATA_DIR / "runs"
DAILY_DIR = DATA_DIR / "daily"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_stratified import load_events, rebuild_stat_rows  # noqa: E402


def main():
    events = load_events()
    idx = get_index("000300", start="20180101")
    regime = market_regime(idx["C"])
    attach_regime(events, regime)

    data = {}
    for f in DAILY_DIR.glob("*.parquet"):
        sym = f.name.split("_")[0]
        if not sym.startswith("index"):
            try:
                data[sym] = pd.read_parquet(f)
            except Exception:  # noqa: BLE001
                pass
    wide = pd.DataFrame({s: rps_series(df, 50) for s, df in data.items()})
    rps_pct = wide.rank(axis=1, pct=True)

    stat_rows = rebuild_stat_rows(events)
    recs = []
    combos = [
        ("FLAT_BREAKOUT", "bull", .85), ("FLAT_BREAKOUT", "bull", .50),
        ("FLAT_BREAKOUT", "bear", .85), ("FLAT_BREAKOUT", None, .85),
        ("W_BOTTOM", "bear", None), ("W_BOTTOM", "bull", None),
    ]
    for pat, reg, rmin in combos:
        sub = []
        for e in events:
            if e.get("pattern") != pat or e.get("duplicate") \
                    or not e.get("signal_date"):
                continue
            if reg and e.get("regime") != reg:
                continue
            ts = pd.Timestamp(e["signal_date"])
            if rmin is not None:
                if ts not in rps_pct.index or e["symbol"] not in rps_pct.columns:
                    continue
                v = rps_pct.loc[ts, e["symbol"]]
                if pd.isna(v) or v < rmin:
                    continue
            sr = next((r for r in stat_rows if r["symbol"] == e["symbol"]
                       and r["signal_date"] == e["signal_date"]), None)
            if sr:
                sub.append(sr)
        for h in (10, 20, 60):
            rs = [r["fwd"][h]["ret"] for r in sub
                  if r.get("fwd") and r["fwd"].get(h)]
            if len(rs) < 15:
                continue
            arr = np.array(rs)
            wins, losses = arr[arr > 0], arr[arr <= 0]
            recs.append({
                "pattern": pat,
                "filter": f"regime={reg or 'ANY'}"
                          + (f", RPS>={int(rmin*100)}" if rmin else ""),
                "horizon": f"{h}d", "n": len(arr),
                "win_rate": round(float((arr > 0).mean()), 3),
                "mean_ret": round(float(arr.mean()), 4),
                "median_ret": round(float(np.median(arr)), 4),
                "pl": (round(float(wins.mean() / abs(losses.mean())), 2)
                       if len(wins) and len(losses) else None),
            })
    tbl = pd.DataFrame(recs)
    print("===== 组合过滤：形态 × 大盘环境 × RPS =====")
    print(tbl.to_string(index=False))
    tbl.to_csv(RUN_DIR / "stats_cross_filter.csv", index=False)


from sps.stratify import market_regime  # noqa: E402

if __name__ == "__main__":
    main()
