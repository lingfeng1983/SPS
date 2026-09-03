"""全市场/样本扫描运行器（C阶段：内嵌环境适配漏斗，直接产出标的清单）。

用法：
  python scripts/run_scan.py --max-stocks 120
  python scripts/run_scan.py --symbols 600519,300750
  python scripts/run_scan.py --recent 20        # 只看最近N天内触发的标的
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from sps.data import (DATA_DIR, get_all_symbols, get_daily, get_index)
from sps.fundamental import check_redlines, get_fundamental
from sps.patterns import ALL_DETECTORS
from sps.stats import aggregate, entry_price, forward_stats, stop_loss_sim
from sps.stratify import (ENV_FIT, attach_regime, build_rps_table,
                          market_regime, rps_series)

OUT_DIR = DATA_DIR / "runs"


def dedup_events(events: list["dict"], df: pd.DataFrame,
                 window_days: int = 20) -> list[dict]:
    """同股票同标签在 window 个交易日内只保留首次事件（规格书四.6）。"""
    idx_map = {d: i for i, d in enumerate(df.index)}
    kept, out = {}, []
    for e in sorted(events, key=lambda x: x.get("signal_date") or ""):
        sd = e.get("signal_date")
        if not sd:
            continue
        key = (e["symbol"], e["pattern"])
        pos = idx_map.get(pd.Timestamp(sd))
        last = kept.get(key)
        if last is not None and pos is not None and pos - last <= window_days:
            e["duplicate"] = True
        else:
            kept[key] = pos
            e["duplicate"] = False
        out.append(e)
    return out


def score_event(e: dict, rps_pct: pd.DataFrame) -> None:
    """给事件附加 env_fit / rps / score（0~100 环境适配分）。"""
    reg = e.get("regime", "?")
    sym, sd = e["symbol"], e.get("signal_date")
    rps = None
    if sd and sym in rps_pct.columns:
        try:
            rps = float(rps_pct.loc[pd.Timestamp(sd), sym])
        except (KeyError, ValueError):
            rps = None
    e["rps50"] = round(rps, 3) if rps is not None and not pd.isna(rps) else None
    fit_fn = ENV_FIT.get(e["pattern"])
    e["env_fit"] = fit_fn(reg, rps) if fit_fn else "neutral"
    # 评分：good=100, neutral=50, bad=0；再叠加 RPS 强度（越强越加）
    base = {"good": 100, "neutral": 50, "bad": 0}.get(e["env_fit"], 50)
    rps_bonus = 0
    if rps is not None:
        rps_bonus = int(min(rps, 1.0) * 30)  # 最多+30
    e["score"] = base + rps_bonus


def run(symbols: list[str], start: str = "20190101",
        recent_days: int | None = None,
        kind_map: dict[str, str] | None = None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events_all, stat_rows = [], []
    total = len(symbols)

    # 基准与 RPS 表（供环境评分用）
    idx = get_index("000300", start="20180101")
    regime = market_regime(idx["C"])

    print(f"[progress] 0/{total}")

    # ====================================================
    # 第一步：批量获取日线（HiThink 并发 + akshare 回退）
    # ====================================================
    from sps.data import HIT_HINK_AVAILABLE, batch_get_daily

    def _on_progress(done, tot):
        pct = round(done / tot * 100)
        print(f"[progress] {done}/{tot} ({pct}%)")

    if HIT_HINK_AVAILABLE:
        print(f"  使用 HiThink Finance-API 并发拉取（线程数=5）...")
    else:
        print(f"  HiThink 不可用，回退到 akshare 串行拉取...")

    # 批量获取：线程数 5（HiThink 限速比东财宽松）
    n_cached = len([s for s in symbols if (DATA_DIR / "daily" / f"{s.replace('.','_')}_{start}_latest.parquet").exists()
                    or (DATA_DIR / "daily" / f"{s.replace('.','_')}_{start}_latest_akshare.parquet").exists()])
    if n_cached == total:
        print(f"[progress] {total}/{total} (100%)")
        print(f"  所有 {total} 只股票日线缓存已存在，无需下载")
        daily = {}
        for sym in symbols:
            cache_file = DATA_DIR / "daily" / f"{sym.replace('.','_')}_{start}_latest.parquet"
            akshare_cache = DATA_DIR / "daily" / f"{sym.replace('.','_')}_{start}_latest_akshare.parquet"
            if cache_file.exists():
                daily[sym] = pd.read_parquet(cache_file)
            elif akshare_cache.exists():
                daily[sym] = pd.read_parquet(akshare_cache)
    else:
        daily = batch_get_daily(symbols, start=start, end=None,
                                max_workers=5, on_progress=_on_progress)
        print(f"\n  成功拉取 {len(daily)}/{total} 只股票日线")

    # ====================================================
    # 第二步：形态检测（保持不变，逐只跑检测器）
    # ====================================================
    for i, sym in enumerate(symbols):
        df = daily.get(sym)
        if df is None or len(df) < 60:
            continue
        kind = (kind_map or {}).get(sym, "stock")
        if kind == "stock":
            fu = get_fundamental(sym)
            ok, hits = check_redlines(sym, fundamental=fu)
            if not ok:
                continue
        sym_events = []
        for det_cls in ALL_DETECTORS:
            det = det_cls()
            try:
                evs = det.scan(df, symbol=sym)
            except Exception as e:
                print(f"  [warn] {det_cls.__name__}.scan({sym}) 失败: {e}")
                continue
            sym_events.extend(e.to_dict() for e in evs)
        sym_events = dedup_events(sym_events, df)
        attach_regime(sym_events, regime)
        for d in sym_events:
            if d.get("duplicate") or not d.get("signal_date"):
                events_all.append(d)
                continue
            pos = df.index.get_loc(pd.Timestamp(d["signal_date"]))
            A, epos = entry_price(df, pos)
            if A is not None:
                fwd = forward_stats(df, epos, A)
                stops = stop_loss_sim(df, epos, A)
                d["entry"] = {"price": round(A, 3),
                              "date": str(df.index[epos].date()),
                              "stops": {str(k): round(v, 4)
                                        for k, v in stops.items()}}
                stat_rows.append({"pattern": d["pattern"], "symbol": sym,
                                  "signal_date": d["signal_date"], "fwd": fwd,
                                  "entry": A})
            events_all.append(d)
        if (i + 1) % 100 == 0:
            print(f"  已检测 {i+1}/{total}, 事件累计 {len(events_all)}")

    # RPS 表（基于本次全部已加载股票）
    data = {}
    for d in events_all:
        if d["symbol"] not in data:
            f = [x for x in DATA_DIR.glob(f"daily/{d['symbol']}_*.parquet")]
            if f:
                data[d["symbol"]] = pd.read_parquet(f[0])
    wide = pd.DataFrame({s: rps_series(df, 50) for s, df in data.items()})
    rps_pct = wide.rank(axis=1, pct=True)

    for e in events_all:
        if e.get("signal_date") and not e.get("duplicate"):
            score_event(e, rps_pct)

    # 写事件
    ev_path = OUT_DIR / "events.jsonl"
    with open(ev_path, "w", encoding="utf-8") as f:
        for d in events_all:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # 形态统计
    if stat_rows:
        tbl = aggregate(stat_rows)
        print("\n===== 全样本形态统计（可执行口径）=====")
        print(tbl.to_string(index=False))

    # 候选标的清单（已确认 + 未重复 + 评分排序）
    cands = [e for e in events_all if e.get("status") == "confirmed"
             and not e.get("duplicate") and e.get("signal_date")]
    cands.sort(key=lambda e: (e.get("score", 0), e.get("rps50") or 0),
               reverse=True)
    if recent_days:
        cut = pd.Timestamp(date.today()) - timedelta(days=recent_days)
        cands = [e for e in cands if pd.Timestamp(e["signal_date"]) >= cut]

    # 候选标的清单（结构形态 only，POCKET_PIVOT 作为佐证不再竞争排序）
    struct = [e for e in events_all if e.get("status") == "confirmed"
              and not e.get("duplicate")
              and e.get("signal_date")
              and e["pattern"] not in ("POCKET_PIVOT",)]
    struct.sort(key=lambda e: (e.get("score", 0), e.get("rps50") or 0),
                reverse=True)
    if recent_days:
        cut = pd.Timestamp(date.today()) - timedelta(days=recent_days)
        struct = [e for e in struct
                  if pd.Timestamp(e["signal_date"]) >= cut]
    # 口袋支点也列个清单（用于佐证叠加搜索）
    pp = [e for e in events_all if e.get("status") == "confirmed"
          and not e.get("duplicate") and e["pattern"] == "POCKET_PIVOT"
          and e.get("signal_date")]
    pp.sort(key=lambda e: (e.get("rps50") or 0), reverse=True)

    print(f"\n===== 候选标的清单（结构形态 only，共 {len(struct)} 个）=====")
    print(f"基本面红线过滤：淘汰 {fund_filtered} 只")
    print(f"{'分数':>4} {'标的':<8} {'形态':<16} {'信号日':<11} {'RPS50':>6} "
          f"{'适配':<8} {'进场价':>9} {'-7%止损':>9} {'20日中位':>8}")
    for e in struct[:60]:
        en = e.get("entry", {})
        stops = en.get("stops", {})
        stop7 = stops.get("-0.07")
        med20 = None
        if e.get("signal_date"):
            sr = next((r for r in stat_rows
                       if r["symbol"] == e["symbol"]
                       and r["signal_date"] == e["signal_date"]), None)
            if sr and sr["fwd"].get(20):
                med20 = sr["fwd"][20]["ret"]
        med_s = f"{med20*100:+.1f}%" if med20 is not None else "-"
        print(f"{e.get('score',0):>4} {e['symbol']:<8} {e['pattern']:<16} "
              f"{e['signal_date']:<11} {str(e.get('rps50')):>6} "
              f"{e.get('env_fit',''):<8} {str(en.get('price','')):>9} "
              f"{str(round(stop7,2) if stop7 else '-'):>9} {med_s:>8}")

    if pp:
        print(f"\n===== 口袋支点佐证清单（共 {len(pp)} 个，单列不参与结构形态排序）=====")
        print(f"{'RPS50':>6} {'标的':<8} {'信号日':<11} {'进场价':>9}")
        for e in pp[:40]:
            en = e.get("entry", {})
            print(f"{str(e.get('rps50')):>6} {e['symbol']:<8} "
                  f"{e['signal_date']:<11} {str(en.get('price','')):>9}")
    print(f"{'分数':>4} {'标的':<8} {'形态':<16} {'信号日':<11} {'RPS50':>6} "
          f"{'适配':<8} {'进场价':>9} {'-7%止损':>9} {'20日中位':>8}")
    for e in cands[:60]:
        en = e.get("entry", {})
        stops = en.get("stops", {})
        stop7 = stops.get("-0.07")
        # 20日中位收益来自 fwd 统计
        med20 = None
        if e.get("signal_date"):
            sr = next((r for r in stat_rows
                       if r["symbol"] == e["symbol"]
                       and r["signal_date"] == e["signal_date"]), None)
            if sr and sr["fwd"].get(20):
                med20 = sr["fwd"][20]["ret"]
        med_s = f"{med20*100:+.1f}%" if med20 is not None else "-"
        print(f"{e.get('score',0):>4} {e['symbol']:<8} {e['pattern']:<16} "
              f"{e['signal_date']:<11} {str(e.get('rps50')):>6} "
              f"{e.get('env_fit',''):<8} {str(en.get('price','')):>9} "
              f"{str(round(stop7,2) if stop7 else '-'):>9} {med_s:>8}")

    # 清单落盘
    (OUT_DIR / "candidates.json").write_text(
        json.dumps(cands, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n候选清单 -> {OUT_DIR / 'candidates.json'}")
    return cands


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default="")
    ap.add_argument("--max-stocks", type=int, default=None)
    ap.add_argument("--recent", type=int, default=None)
    args = ap.parse_args()
    kind_map: dict[str, str] = {}
    if args.symbols:
        syms = [s.strip() for s in args.symbols.split(",")]
    else:
        uni = get_all_symbols(include_etf=False, exclude_st_bj=True)
        syms = uni["symbol"].tolist()
        kind_map = dict(zip(uni["symbol"], uni.get("kind", "stock")))
        n_st = len(uni)
        if args.max_stocks:
            syms = syms[:args.max_stocks]
        print(f"universe: {n_st} 只（已排除 ST/退市/北交所/新三板，不含ETF）")
    run(syms, recent_days=args.recent, kind_map=kind_map)
