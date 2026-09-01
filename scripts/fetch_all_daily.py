"""全市场日线补拉（可断点续传，已有缓存自动跳过）。

用法：
  python scripts/fetch_all_daily.py            # 全 universe
  python scripts/fetch_all_daily.py --workers 6

完成后自动重跑 backtest_params.py 刷新 param_stats.json。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from sps.data import DATA_DIR, DAILY_DIR, get_all_symbols, get_daily

FAIL_LOG = DATA_DIR / "runs" / "fetch_failures.json"


def have_cached(sym: str) -> bool:
    return any(DAILY_DIR.glob(f"{sym}_qfq_*.parquet"))


def fetch_one(sym: str, kind: str, start: str) -> tuple[str, bool, str]:
    if have_cached(sym):
        return sym, True, "cached"
    try:
        get_daily(sym, start=start, kind=kind)
        return sym, True, "ok"
    except Exception as e:  # noqa: BLE001
        return sym, False, str(e)[:120]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--start", default="20150101")
    args = ap.parse_args()

    uni = get_all_symbols(trade_date="2026-08-26")  # 复用已有列表快照
    print(f"universe: {len(uni)}", flush=True)

    kind_map = uni.set_index("symbol")["kind"].to_dict() if "kind" in uni.columns else {}
    todo = [(s, kind_map.get(s, "stock")) for s in uni["symbol"].astype(str)]
    done = fail = 0
    failures: dict[str, str] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, s, k, args.start): s for s, k in todo}
        for fut in as_completed(futs):
            sym, ok, msg = fut.result()
            done += 1
            if not ok:
                fail += 1
                failures[sym] = msg
            if done % 100 == 0 or done == len(futs):
                el = time.time() - t0
                print(f"[{done}/{len(futs)}] fail={fail} "
                      f"elapsed={el/60:.1f}min eta={el/done*(len(futs)-done)/60:.1f}min",
                      flush=True)
    FAIL_LOG.write_text(json.dumps(failures, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"DONE total={len(futs)} fail={fail} -> {FAIL_LOG}", flush=True)


if __name__ == "__main__":
    main()
