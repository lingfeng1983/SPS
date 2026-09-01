"""参数回测：对每个指标的参数网格，在全部缓存股票上计算历史信号的前向表现。

输出 data/runs/param_stats.json：
{indicator: {param_str: {n_signals, win5, win10, win20, avg20}}}

口径与规格书一致：
- 信号日 t 收盘确认 → 进场 t+1 开盘价（无次日则跳过该样本）
- 胜率 = 前向收益>0 占比；avg = 平均收益
运行：python scripts/backtest_params.py [--max-stocks N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from sps.data import DATA_DIR
from sps.positions import COMMISSION_RATE, STAMP_TAX, SLIPPAGE
from sps.screener import INDICATORS, _rps_series

RUN_DIR = DATA_DIR / "runs"
OUT = RUN_DIR / "param_stats.json"
OOS_DAYS = 120          # 样本外验证窗口：最近120个交易日
COST_PER_TRADE = COMMISSION_RATE * 2 + STAMP_TAX + SLIPPAGE * 2   # 单边合计~0.68%

# 每个指标的参数网格（覆盖常用档位；用户自定义值会映射到最近档位展示）
GRIDS = {
    "rps50": [0.7, 0.8, 0.85, 0.9, 0.95],
    "above_ma": [10, 20, 30, 60],
    "near_high": [5, 10, 15, 25],
    "vol_ratio": [1.2, 1.5, 2.0, 2.5],
    "turnover": [[3, 15], [5, 20], [8, 30]],
    "up_days": [2, 3, 4],
    "gain_today": [2, 3, 5, 7],
    "pullback_stable": [2, 3, 5],
    "box_amp": [8, 12, 15, 20],
    "vol_narrow": [2.5, 3.0, 3.5, 4.5],
    # 全量因子
    "mom_win": [[20, 10], [20, 15], [20, 20], [60, 20], [60, 30]],
    "rsv_pos": [[60, 50], [60, 80], [60, 95], [120, 80], [250, 90]],
    "ma_align": [2, 3, 5, 8],
    "macd_cross": [3, 5, 8],
    "new_high_cnt": [1, 2, 4],
    "ma_spread": [[-2, 5], [0, 8], [0, 12], [3, 15]],
    "amp20": [3.5, 5.0, 7.0],
    "yang_streak": [2, 3, 5],
}
HORIZONS = (5, 10, 20)


def load_daily(max_stocks: int | None) -> tuple[dict, pd.DataFrame | None]:
    daily = {}
    files = sorted(DATA_DIR.glob("daily/*.parquet"))
    if max_stocks:
        files = files[:max_stocks]
    for f in files:
        sym = f.stem.split("_")[0]
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if isinstance(df.index, pd.DatetimeIndex):
            base = df
        elif "date" in df.columns:
            base = df.set_index(pd.to_datetime(df["date"]))
        else:
            continue
        if {"O", "H", "L", "C", "V"} <= set(base.columns):
            d = base[["O", "H", "L", "C", "V"]].dropna()
            if len(d) >= 120:
                daily[sym] = d
    wide = pd.DataFrame({s: d["C"] for s, d in daily.items()}) if daily else None
    return daily, wide


def forward_returns(df: pd.DataFrame, sig_pos: int) -> dict | None:
    """t+1 开盘进场，各周期收益。样本不足返回 None。"""
    n = len(df)
    epos = sig_pos + 1
    if epos >= n:
        return None
    A = float(df["O"].iloc[epos])
    if A <= 0:
        return None
    out = {}
    for h in HORIZONS:
        i = epos + h - 1          # 持有h个交易日
        if i < n:
            out[h] = float(df["C"].iloc[i]) / A - 1
    return out


def backtest(daily: dict, wide: pd.DataFrame | None) -> dict:
    rps = _rps_series(wide, 50) if wide is not None else None
    stats: dict[str, dict[str, dict]] = {}
    # 全市场统一的样本内/外切分日（按最新K线日往前推 OOS_DAYS 个交易日）
    latest = max(df.index[-1] for df in daily.values())
    cut = latest - pd.Timedelta(days=int(OOS_DAYS * 1.5))   # 120交易日≈180自然日
    print(f"样本外切分日: {cut.date()} (之后 {OOS_DAYS} 交易日为样本外)")

    for name, meta in INDICATORS.items():
        grid = GRIDS.get(name)
        if not grid:
            continue
        stats[name] = {}
        for p in grid:
            rets = {h: [] for h in HORIZONS}          # 全样本(扣成本后)
            rets_is = {h: [] for h in HORIZONS}       # 样本内
            rets_oos = {h: [] for h in HORIZONS}      # 样本外
            n_sig = 0
            for sym, df in daily.items():
                df.attrs["symbol"] = sym
                try:
                    series = meta["fn"](df, p, rps=rps)
                except Exception:
                    continue
                if series is None or series.empty:
                    continue
                # 信号去重：连续True只取第一天
                sig = series & ~series.shift(1).fillna(False).astype(bool)
                positions = np.where(sig.values)[0]
                for pos in positions[-60:]:     # 每股最多取最近60个信号，控时长
                    fr = forward_returns(df, int(pos))
                    if not fr:
                        continue
                    # 交易成本：以20日持有为主口径扣除（进出各计一次滑点）
                    costed = {h: v - COST_PER_TRADE for h, v in fr.items()}
                    n_sig += 1
                    in_oos = df.index[int(pos)] >= cut
                    for h, v in costed.items():
                        rets[h].append(v)
                        (rets_oos if in_oos else rets_is)[h].append(v)
            rec = {"n_signals": n_sig}
            for h in HORIZONS:
                arr = np.array(rets[h])
                if len(arr):
                    rec[f"win{h}"] = round(float((arr > 0).mean()) * 100, 1)
                    if h == 20:
                        rec["avg20"] = round(float(arr.mean()) * 100, 2)
                else:
                    rec[f"win{h}"] = None
            # 样本外验证：样本内/外 20日胜率对比
            a_is, a_oos = np.array(rets_is[20]), np.array(rets_oos[20])
            if len(a_is) >= 30 and len(a_oos) >= 15:
                rec["oos_win20"] = round(float((a_oos > 0).mean()) * 100, 1)
                rec["is_win20"] = rec.get("win20")
                rec["oos_n"] = int(len(a_oos))
                # 内外差距 >8 个百分点 → 标记衰减
                rec["oos_decay"] = bool(
                    rec["is_win20"] is not None
                    and rec["is_win20"] - rec["oos_win20"] > 8)
            stats[name][str(p)] = rec
        print(f"[done] {name}: {len(stats[name])} 档参数")

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-stocks", type=int, default=None)
    args = ap.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    daily, wide = load_daily(args.max_stocks)
    print(f"载入 {len(daily)} 只股票缓存")
    if not daily:
        print("无缓存数据，先运行 run_scan.py")
        return
    stats = backtest(daily, wide)
    meta = {"generated": str(pd.Timestamp.now().date()),
            "stocks": len(daily), "stats": stats,
            "oos_days": OOS_DAYS,
            "cost_model": f"每笔往返扣 {COST_PER_TRADE*100:.2f}%（佣金万{COMMISSION_RATE*1e4:.0f}双边+印花税千{STAMP_TAX*1e3:.1f}+滑点千{SLIPPAGE*1e3:.0f}双边）"}
    OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n✅ 参数回测完成 → {OUT}")


if __name__ == "__main__":
    main()
