"""全市场/样本扫描运行器（C阶段：内嵌环境适配漏斗，直接产出标的清单）。

用法：
  python scripts/run_scan.py --max-stocks 120
  python scripts/run_scan.py --symbols 600519,300750
  python scripts/run_scan.py --recent 20        # 只看最近N天内触发的标的
  python scripts/run_scan.py --data-only        # 只更新行情数据，不跑形态检测
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from sps.data import (DATA_DIR, get_all_symbols, get_daily, get_index)
from sps.fundamental import check_redlines, get_fundamental
from sps.patterns import ALL_DETECTORS
from sps.candidates import STOP_CONTRACT, write_candidate_artifacts
from sps.stats import (aggregate, entry_price, forward_stats, stop_exit_sim,
                       stop_prices)
from sps.stratify import (ENV_FIT, attach_regime, build_rps_table,
                          market_regime, rps_series)

OUT_DIR = DATA_DIR / "runs"

# 并行检测的共享状态（由 _shard_init 注入子进程）
_G: dict = {}


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
    # 次新股上市不足 N 日时横截面分位为 NaN，视同无 RPS 数据
    if rps is not None and pd.isna(rps):
        rps = None
    e["rps50"] = round(rps, 3) if rps is not None else None
    fit_fn = ENV_FIT.get(e["pattern"])
    e["env_fit"] = fit_fn(reg, rps) if fit_fn else "neutral"
    # 评分：good=100, neutral=50, bad=0；再叠加 RPS 强度（越强越加）
    base = {"good": 100, "neutral": 50, "bad": 0}.get(e["env_fit"], 50)
    rps_bonus = int(min(rps, 1.0) * 30) if rps is not None else 0  # 最多+30
    e["score"] = base + rps_bonus


def _detect_one(sym: str, df: pd.DataFrame | None, regime: pd.DataFrame,
                kind_map: dict[str, str] | None) -> tuple[list, list, dict, list]:
    """单票形态检测（纯函数，串行/并行两条路径共用同一实现）。

    返回 (events, stat_rows, counters, warns)；counters 用于聚合覆盖元数据。
    """
    counters = {"loaded": 0, "insufficient_history": 0, "fund_filtered": 0,
                "processed": 0, "detector_failures": 0}
    warns: list[str] = []
    events, stat_rows = [], []
    if df is None:
        return events, stat_rows, counters, warns
    counters["loaded"] = 1
    if len(df) < 60:
        counters["insufficient_history"] = 1
        return events, stat_rows, counters, warns
    kind = (kind_map or {}).get(sym, "stock")
    if kind == "stock":
        fu = get_fundamental(sym)
        ok, hits = check_redlines(sym, fundamental=fu)
        if not ok:
            counters["fund_filtered"] = 1
            return events, stat_rows, counters, warns
    counters["processed"] = 1
    sym_events = []
    for det_cls in ALL_DETECTORS:
        det = det_cls()
        try:
            evs = det.scan(df, symbol=sym)
        except Exception as e:  # noqa: BLE001
            counters["detector_failures"] += 1
            warns.append(f"[warn] {det_cls.__name__}.scan({sym}) 失败: {e}")
            continue
        sym_events.extend(e.to_dict() for e in evs)
    sym_events = dedup_events(sym_events, df)
    attach_regime(sym_events, regime)
    for d in sym_events:
        if d.get("duplicate") or not d.get("signal_date"):
            events.append(d)
            continue
        pos = df.index.get_loc(pd.Timestamp(d["signal_date"]))
        A, epos = entry_price(df, pos)
        if A is not None:
            fwd = forward_stats(df, epos, A)
            stops = stop_prices(A)
            stop_exits = stop_exit_sim(df, epos, A)
            d["entry"] = {"price": round(A, 3),
                          "date": str(df.index[epos].date()),
                          "stops": {str(k): round(v, 4)
                                    for k, v in stops.items()},
                          "stop_contract": STOP_CONTRACT,
                          "stop_exits": {str(k): round(v, 4)
                                         for k, v in stop_exits.items()}}
            stat_rows.append({"pattern": d["pattern"], "symbol": sym,
                              "signal_date": d["signal_date"], "fwd": fwd,
                              "entry": A})
        events.append(d)
    return events, stat_rows, counters, warns


def _shard_init(regime_df, kind_map, data_dir: str) -> None:
    """子进程初始化：注入共享状态（只随进程 pickle 一次，不随任务重复）。"""
    _G["regime"] = regime_df
    _G["kind_map"] = kind_map
    _G["daily_dir"] = Path(data_dir) / "daily"


def _detect_shard(syms: list[str]):
    """子进程任务：自行读取本分片的日线缓存并逐票检测，只回传小结果。"""
    events, stat_rows, warns = [], [], []
    total_cnt = {"n_syms": len(syms), "loaded": 0, "insufficient_history": 0,
                 "fund_filtered": 0, "processed": 0, "detector_failures": 0}
    for sym in syms:
        df = None
        try:
            files = list(_G["daily_dir"].glob(f"{sym}_*.parquet"))
            if files:
                # 按「数据最后日期」选最新文件（mtime 会选中过期残留）
                try:
                    from sps.health import _parquet_last_date
                    pick = max(files, key=_parquet_last_date)
                except Exception:
                    pick = max(files, key=lambda p: p.stat().st_mtime)
                candidate = pd.read_parquet(pick)
                if isinstance(candidate.index, pd.DatetimeIndex) and \
                        {"O", "H", "L", "C", "V"} <= set(candidate.columns):
                    df = candidate
                elif "date" in candidate.columns:
                    cand = candidate.set_index(pd.to_datetime(candidate["date"]))
                    if {"O", "H", "L", "C", "V"} <= set(cand.columns):
                        df = cand[["O", "H", "L", "C", "V"]]
        except Exception:
            df = None
        evs, srows, cnt, w = _detect_one(sym, df, _G["regime"], _G["kind_map"])
        events.extend(evs)
        stat_rows.extend(srows)
        warns.extend(w)
        for k in ("loaded", "insufficient_history", "fund_filtered",
                  "processed", "detector_failures"):
            total_cnt[k] += cnt[k]
    return events, stat_rows, total_cnt, warns


def _scan_workers() -> int:
    """并行检测进程数：默认 CPU 核数（上限 8），SPS_SCAN_WORKERS 覆盖，0=串行。"""
    env = os.environ.get("SPS_SCAN_WORKERS")
    if env is not None:
        try:
            return max(0, int(env))
        except ValueError:
            pass
    return min(os.cpu_count() or 2, 8)


def run(symbols: list[str], start: str = "20190101",
        recent_days: int | None = None,
        kind_map: dict[str, str] | None = None,
        data_only: bool = False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events_all, stat_rows = [], []
    fund_filtered = 0
    insufficient_history = 0
    processed_symbols = 0
    detector_failures = 0
    failed_load = 0
    total = len(symbols)

    print(f"[progress] 0/{total}")

    # ====================================================
    # 第一步：批量获取日线（快照快路径 + HiThink 并发 + akshare 回退）
    # ====================================================
    from sps.data import batch_get_daily, hithink_available

    def _on_progress(done, tot):
        pct = round(done / tot * 100)
        print(f"[progress] {done}/{tot} ({pct}%)")

    if hithink_available():
        print("  使用 HiThink Finance-API 拉取（快照快路径 + 并发补漏）...")
    else:
        print("  HiThink 不可用，回退到 akshare 串行拉取...")

    # 扫描前强制刷新：缓存只差一天用全市场快照秒级补齐，缺历史的票才逐票拉。
    daily = batch_get_daily(symbols, start=start, end=None, max_workers=5,
                            on_progress=_on_progress, refresh=True)
    print(f"\n  成功加载 {len(daily)}/{total} 只股票日线")

    if data_only:
        print(f"[完成] 行情数据已更新（{len(daily)}/{total} 只），未跑形态扫描")
        return []

    # 基准与牛熊分层（供环境评分用）
    idx = get_index("000300", start="20180101")
    regime = market_regime(idx["C"])

    # ====================================================
    # 第二步：形态检测（多进程分片并行，失败回退单进程）
    # ====================================================
    workers = _scan_workers()
    env_override = os.environ.get("SPS_SCAN_WORKERS") is not None
    ran_parallel = False
    loaded_total = 0
    if workers >= 2 and (env_override or total > 80):
        shard_size = max(20, total // (workers * 6))
        shards = [symbols[i:i + shard_size] for i in range(0, total, shard_size)]
        try:
            import multiprocessing as mp
            loaded_total = 0
            done_syms = 0
            with mp.Pool(workers, initializer=_shard_init,
                         initargs=(regime, kind_map or {}, str(DATA_DIR))) as pool:
                for evs, srows, cnt, warns in pool.imap_unordered(
                        _detect_shard, shards):
                    events_all.extend(evs)
                    stat_rows.extend(srows)
                    fund_filtered += cnt["fund_filtered"]
                    insufficient_history += cnt["insufficient_history"]
                    processed_symbols += cnt["processed"]
                    detector_failures += cnt["detector_failures"]
                    loaded_total += cnt["loaded"]
                    for w in warns:
                        print(f"  {w}")
                    done_syms += cnt["n_syms"]
                    pct = round(min(done_syms, total) / total * 100)
                    print(f"[progress] {min(done_syms, total)}/{total} ({pct}%)")
                    print(f"  已检测 {done_syms}/{total}, 事件累计 {len(events_all)}")
            failed_load = total - loaded_total
            ran_parallel = True
        except Exception as e:  # noqa: BLE001
            # 回退单进程前必须清空并行阶段已收集的部分结果，否则事件翻倍
            print(f"[warn] 并行检测不可用（{e}），回退单进程")
            events_all.clear()
            stat_rows.clear()
            fund_filtered = insufficient_history = 0
            processed_symbols = detector_failures = loaded_total = 0

    if not ran_parallel:
        for i, sym in enumerate(symbols):
            evs, srows, cnt, warns = _detect_one(
                sym, daily.get(sym), regime, kind_map)
            events_all.extend(evs)
            stat_rows.extend(srows)
            fund_filtered += cnt["fund_filtered"]
            insufficient_history += cnt["insufficient_history"]
            processed_symbols += cnt["processed"]
            detector_failures += cnt["detector_failures"]
            for w in warns:
                print(f"  {w}")
            if (i + 1) % 100 == 0:
                print(f"  已检测 {i+1}/{total}, 事件累计 {len(events_all)}")
        failed_load = max(total - len(daily), 0)

    # RPS 表（基于本次全部已加载股票）
    from sps.health import _parquet_last_date
    data = {}
    for d in events_all:
        if d["symbol"] not in data:
            f = list(DATA_DIR.glob(f"daily/{d['symbol']}_*.parquet"))
            if f:
                try:
                    pick = max(f, key=_parquet_last_date)
                except Exception:
                    pick = f[0]
                data[d["symbol"]] = pd.read_parquet(pick)
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

    # 清单与覆盖元数据绑定落盘，读端会用哈希拒绝不匹配的半成品。
    failed_load = max(total - len(daily), 0)
    meta = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "requested_symbols": total,
        "loaded_symbols": len(daily),
        "failed_load_symbols": failed_load,
        "insufficient_history_symbols": insufficient_history,
        "filtered_symbols": fund_filtered,
        "processed_symbols": processed_symbols,
        "detector_failures": detector_failures,
        "accounted_symbols": (failed_load + insufficient_history
                              + fund_filtered + processed_symbols),
        "event_count": len(events_all),
        "candidate_count": len(cands),
        "candidate_symbols": len({e["symbol"] for e in cands}),
        "latest_signal_date": max(
            (e.get("signal_date") or "" for e in cands), default=""
        ),
    }
    write_candidate_artifacts(OUT_DIR / "candidates.json", cands, meta)
    print(f"\n候选清单 -> {OUT_DIR / 'candidates.json'}")
    return cands


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # Windows 多进程子进程引导
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default="")
    ap.add_argument("--max-stocks", type=int, default=None)
    ap.add_argument("--recent", type=int, default=None)
    ap.add_argument("--data-only", action="store_true",
                    help="只更新行情数据，不跑形态检测")
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
    run(syms, recent_days=args.recent, kind_map=kind_map,
        data_only=args.data_only)
