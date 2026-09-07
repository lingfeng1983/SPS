"""每日持仓监控脚本：收盘后拉取最新日线，跑诊断，输出简洁报告。

用法：
  python scripts/monitor.py                  # 默认：持仓诊断
  python scripts/monitor.py --format json    # JSON 格式（供程序消费）
  python scripts/monitor.py --telegram       # 输出带 Telegram emoji 标记的文本

计划任务（Hermes cronjob）：
  python scripts/monitor.py --telegram
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from sps.data import get_daily, get_index
from sps.positions import run_diagnosis, load_positions


def pull_daily(symbols: list[str], max_workers: int = 5) -> dict[str, pd.DataFrame]:
    """并发拉取日线，带重试。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[str, pd.DataFrame] = {}

    def _fetch_one(sym: str):
        for attempt in range(3):
            try:
                df = get_daily(sym, start="20240101")
                if df is not None and len(df) > 0:
                    return sym, df
            except Exception:
                time.sleep(2 + attempt * 3)
        return sym, None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_fetch_one, s): s for s in symbols}
        for fut in as_completed(futs):
            sym, df = fut.result()
            if df is not None:
                results[sym] = df
    return results


def monitor(format: str = "text", telegram: bool = False) -> str:
    """跑完整持仓监控，返回格式化报告。"""
    positions = load_positions()
    opens = [r for r in positions if r["status"] == "open"]
    if not opens:
        return "⚠️ 无持仓中记录。请先通过 SPS 界面添加持仓。"

    syms = [r["symbol"] for r in opens]

    # 拉取日线
    daily = pull_daily(syms)

    # 跑诊断
    diag = run_diagnosis(daily)

    # 市场基准（沪深300）
    try:
        idx = get_index("000300", start="20240101")
        idx_ret_1d = (idx["C"].iloc[-1] / idx["C"].iloc[-2] - 1) * 100
        idx_ret_20d = (idx["C"].iloc[-1] / idx["C"].iloc[-21] - 1) * 100
        benchmark = f"300指数 当日{idx_ret_1d:+.2f}% · 20日{idx_ret_20d:+.2f}%"
    except Exception:
        benchmark = "基准数据不可用"

    # 汇总统计
    sells = [r for r in diag["results"] if r["action"] in ("sell", "warn_sell")]
    holds = [r for r in diag["results"] if r["action"] == "hold"]
    reds = [r for r in diag["results"] if r.get("health") == "red"]
    yellows = [r for r in diag["results"] if r.get("health") == "yellow"]

    total_pnl = sum(r.get("pnl_pct", 0) for r in diag["results"])

    if format == "json":
        return json.dumps({
            "day": diag["day"],
            "summary": diag["summary"],
            "results": diag["results"],
            "benchmark": benchmark,
            "total_pnl_pct": round(total_pnl, 2),
        }, ensure_ascii=False, indent=2)

    # 文本格式
    lines = []
    header = f"📊 SPS 持仓监控 — {diag['day']}"
    lines.append(header)
    lines.append(f"📈 {benchmark}")
    lines.append(f"持仓 {diag['summary']['open']} | 卖出信号 {len(sells)} | 持有 {len(holds)}")
    lines.append(f"总浮亏合计: {total_pnl:+.1f}%")
    lines.append("")

    if sells:
        lines.append("🔴 卖出/预警信号:")
        for r in sells:
            name = r.get("name", "")
            signals = ", ".join(s["rule"] for s in r.get("signals", []))
            lines.append(
                f"  · {r['symbol']} {name} | 现价 {r.get('price','-')} | "
                f"浮亏 {r.get('pnl_pct','-')}% | {signals}"
            )

    if yellows:
        lines.append("🟡 趋势预警:")
        for r in yellows:
            name = r.get("name", "")
            tip = r.get("health_tip", "")
            lines.append(f"  · {r['symbol']} {name} | 现价 {r.get('price','-')} | {tip}")

    if holds:
        lines.append("🟢 持有中:")
        for r in holds:
            name = r.get("name", "")
            lines.append(f"  · {r['symbol']} {name} | 现价 {r.get('price','-')} | 浮亏 {r.get('pnl_pct','-')}%")

    if not sells and not yellows:
        lines.append("✅ 无卖出信号，持仓健康。")

    report = "\n".join(lines)

    if telegram:
        report = report.replace("⚠️", "⚠️").replace("📊", "📊").replace("🔴", "🔴").replace("🟡", "🟡").replace("🟢", "🟢").replace("✅", "✅").replace("📈", "📈")

    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["text", "json"], default="text")
    ap.add_argument("--telegram", action="store_true", help="输出适配 Telegram 的格式")
    args = ap.parse_args()

    report = monitor(format=args.format, telegram=args.telegram)
    print(report)
