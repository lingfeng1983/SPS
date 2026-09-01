"""为每个候选标的生成可视化 HTML：K线 + 形态标记 + 基本面卡 + 信号详情。

交付：D:\SPS\data\runs\viz\ 下每个候选 <symbol>.html + index.html。

用法：
  python scripts/gen_viz.py --min-score 120 --limit 40   # 前40个优质候选
  python scripts/gen_viz.py --all                          # 全候选
  python scripts/gen_viz.py --symbols 000001,000002       # 指定几只
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from sps.data import DATA_DIR
from sps.fundamental import get_fundamental

RUN_DIR = DATA_DIR / "runs"
VIZ_DIR = RUN_DIR / "viz"
CDN_PLOTLY = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def load_candidates() -> list[dict]:
    p = RUN_DIR / "candidates.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def candlestick_data(symbol: str) -> pd.DataFrame | None:
    """从缓存的日线 Parquet 取出 OHLCV（含 date）。"""
    for f in sorted(DATA_DIR.glob(f"daily/{symbol}_*.parquet"), key=lambda p: p.stat().st_mtime):
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if df.empty:
            continue
        # date 可能来自 index 或独立列 — 避免索引/列二义性
        if isinstance(df.index, pd.DatetimeIndex):
            dates = pd.to_datetime(df.index).values
            base = df
        else:
            if "date" not in df.columns:
                continue
            dates = pd.to_datetime(df["date"]).values
            base = df.drop(columns=[col for col in df.columns if col.lower() == "date"])
        need = ["O", "H", "L", "C", "V"]
        if not set(need).issubset(base.columns):
            continue
        n = min(len(dates), *(len(base[c]) for c in need))
        dates = dates[:n].tolist()
        out = {col: base[col].tolist()[:n] for col in need}
        return {"dates": dates, "O": out["O"], "H": out["H"], "L": out["L"], "C": out["C"], "V": out["V"]}
    return None


def fmt_price(v) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):,.2f}"
    except (ValueError, TypeError):
        return "-"


def row(k, v):
    return f"      <tr><td style='padding-right:12px'><b>{k}</b></td><td>{v}</td></tr>\n"


def make_candidate_html(symbol: str, event: dict, fu: dict) -> str:
    df = candlestick_data(symbol)
    pf = event.get("pattern", "")
    score = event.get("score", 0)
    env_fit = event.get("env_fit", "-")
    regime = event.get("regime", "-")
    rps50 = event.get("rps50")
    sig_date = event.get("signal_date", "-")
    avail = event.get("available_at", "-")
    pat_start = event.get("pattern_start") or "-"
    pat_end = event.get("pattern_end") or "-"
    entry = event.get("entry", {})
    entry_price = entry.get("price")
    entry_date = entry.get("date", "-")
    stops = entry.get("stops", {})
    stop7 = stops.get("-0.07")
    kl = event.get("key_levels", {})

    # 基本面
    name = fu.get("_name", "")
    pe_ttm = fmt_price(fu.get("pe_ttm"))          # 返回-，因为fundamental不含pe
    pb = fmt_price(fu.get("pb"))
    roe = fu.get("roe")
    roe_s = f"{roe:.2f}%" if roe is not None else "-"
    debt = fu.get("debt_ratio")
    debt_s = f"{debt:.2f}%" if debt is not None else "-"
    pyoy = fu.get("profit_yoy")
    pyoy_s = f"{pyoy:+.2f}%" if pyoy is not None else "-"
    ryoy = fu.get("revenue_yoy")
    ryoy_s = f"{ryoy:+.2f}%" if ryoy is not None else "-"
    report_date = fu.get("report_date", "-")

    env_cls = "good" if env_fit == "good" else ("bad" if env_fit == "bad" else "neutral")

    # ---------- K线 HTML ----------
    if df is None or not isinstance(df, dict) or "dates" not in df:
        kline_block = "<div class='warn'>⚠️ 无日线数据（已从 Parquet 缓存读取失败）。</div>"
        shapes = []
        traces_json = "[]"
        layout_json = json.dumps({
            "title": {"text": f"{symbol} {name} 无数据"},
            "xaxis": {"title": "日期"}, "yaxis": {"title": "价格"},
            "height": 300, "margin": {"t": 20}
        })
    else:
        dates = df["dates"]
        dopen = df["O"]; dhigh = df["H"]; dlow = df["L"]; dclose = df["C"]
        # 均线
        ma5 = df["C"].rolling(5, min_periods=5).mean().tolist()
        ma10 = df["C"].rolling(10, min_periods=10).mean().tolist()
        ma20 = df["C"].rolling(20, min_periods=20).mean().tolist()

        # 信号日 / 进场日索引
        sig_idx = None
        ent_idx = None
        if sig_date and sig_date != "-":
            try:
                sig_ts = pd.Timestamp(sig_date)
                if sig_ts in df["date"].values:
                    sig_idx = int(df.index[df["date"] == sig_ts][0])
            except Exception:
                pass
        if entry_date and entry_date != "-":
            try:
                ent_ts = pd.Timestamp(entry_date)
                if ent_ts in df["date"].values:
                    ent_idx = int(df.index[df["date"] == ent_ts][0])
            except Exception:
                pass

        # 形状
        shapes = []
        # 信号日竖线
        if sig_idx is not None:
            shapes.append({
                "type": "line",
                "x0": dates[sig_idx], "x1": dates[sig_idx],
                "y0": 0, "y1": 1, "yref": "paper",
                "line": {"color": "#2a9d8f", "width": 1.8, "dash": "dashdot"},
            })
        # 进场日竖线
        if ent_idx is not None:
            shapes.append({
                "type": "line",
                "x0": dates[ent_idx], "x1": dates[ent_idx],
                "y0": 0, "y1": 1, "yref": "paper",
                "line": {"color": "#555", "width": 1.2, "dash": "dot"},
            })
        # 止损线
        if stop7 is not None:
            shapes.append({
                "type": "line",
                "x0": dates[0], "x1": dates[-1],
                "y0": stop7, "y1": stop7,
                "line": {"color": "#d62828", "width": 1.4, "dash": "dash"},
            })
        # 关键位水平线
        for k, v in kl.items():
            keyname = str(k).lower()
            if keyname in ("breakout", "neckline", "buy_point", "entry"):
                if isinstance(v, (int, float)):
                    shapes.append({
                        "type": "line",
                        "x0": dates[0], "x1": dates[-1],
                        "y0": v, "y1": v,
                        "line": {"color": "#4361ee", "width": 1.2, "dash": "dot"},
                    })

        traces = [{
            "type": "candlestick",
            "x": json.dumps(dates),
            "open": json.dumps(dopen),
            "high": json.dumps(dhigh),
            "low": json.dumps(dlow),
            "close": json.dumps(dclose),
            "increasing": {"line": {"color": "#2a9d8f"}},
            "decreasing": {"line": {"color": "#d62828"}},
            "name": "K线",
        }]

        if any(m is not None for m in ma5):
            traces.append({
                "type": "scatter", "x": json.dumps(dates),
                "y": json.dumps(ma5), "mode": "lines",
                "name": "MA5", "line": {"color": "#f9a03f", "width": 1},
            })
        if any(m is not None for m in ma10):
            traces.append({
                "type": "scatter", "x": json.dumps(dates),
                "y": json.dumps(ma10), "mode": "lines",
                "name": "MA10", "line": {"color": "#4361ee", "width": 1},
            })
        if any(m is not None for m in ma20):
            traces.append({
                "type": "scatter", "x": json.dumps(dates),
                "y": json.dumps(ma20), "mode": "lines",
                "name": "MA20", "line": {"color": "#d62828", "width": 1},
            })

        traces_json = json.dumps(traces, ensure_ascii=False)
        layout = {
            "shapes": shapes,
            "legend": {"orientation": "h", "y": -0.22},
            "xaxis": {
                "title": "日期", "rangeslider": {"visible": False},
                "rangeselector": {"buttons": [
                    {"count": 5, "label": "5日", "step": "day", "stepmode": "backward"},
                    {"count": 20, "label": "20日", "step": "day", "stepmode": "backward"},
                    {"count": 60, "label": "60日", "step": "day", "stepmode": "backward"},
                    {"step": "all", "label": "全体"},
                ]},
            },
            "yaxis": {"title": "价格 (元)"},
            "height": 600,
            "margin": {"l": 50, "r": 20, "t": 30, "b": 75},
        }
        layout_json = json.dumps(layout, ensure_ascii=False)

        kline_block = f"""
<div class="kline-box">
  <div id="kline-{symbol}"></div>
  <script>
    var data = {traces_json};
    var layout = {layout_json};
    var config = {{responsive:true, displayModeBar:false, displaylogo:false}};
    Plotly.newPlot('kline-{symbol}', data, layout, config);
  </script>
  <div class="kline-legend">
    <span><b style="color:#2a9d8f">●</b> 阳线</span>
    <span><b style="color:#d62828">●</b> 阴线</span>
    <span><b style="color:#f9a03f">—</b> MA5</span>
    <span><b style="color:#4361ee">—</b> MA10</span>
    <span><b style="color:#d62828">—</b> MA20</span>
    <span><b style="color:#2a9d8f;border-top:2px dashed">━</b> 信号日</span>
    <span><b style="color:#555;border-top:1px dotted">┄</b> 进场日</span>
    <span><b style="color:#d62828;border-top:2px dashed">━</b> 止损 -7%</span>
    <span><b style="color:#4361ee;border-top:1px dotted">┄</b> 关键位</span>
  </div>
</div>
"""

    # ---------- 信号详情 + 基本面 行生成 ----------
    sig_rows = ""
    sig_rows += row("形态", pf)
    sig_rows += row("信号日", sig_date)
    sig_rows += row("形态起始日", pat_start)
    sig_rows += row("形态结束日", pat_end)
    sig_rows += row("可用确认日", avail)
    sig_rows += row("大盘环境", regime)
    sig_rows += row("RPS(50)", fmt_price(rps50) if isinstance(rps50, (int, float)) else (rps50 if rps50 else "-"))
    sig_rows += row("评分", f"{score} / 100")
    sig_rows += row("环境适配", env_fit)

    entry_rows = ""
    entry_rows += row("进场价格", fmt_price(entry_price) + " 元")
    entry_rows += row("进场日期", entry_date)
    entry_rows += row("进场口径", "信号次日开盘价（可执行）" if entry_date != "-" else "理论价（次日未成交）")
    entry_rows += row("止损 -7%", fmt_price(stop7) + " 元" if stop7 else "-")

    kl_rows = ""
    if kl:
        for k, v in kl.items():
            if isinstance(v, (int, float)):
                kl_rows += row(k.replace("_", " ").title(), fmt_price(v) + " 元")
    if not kl_rows:
        kl_rows = "      <tr><td colspan='2' style='color:#999'>— 该形态无独立关键位 —</td></tr>\n"

    fm_rows = ""
    fm_rows += row("TTM 市盈率", pe_ttm)
    fm_rows += row("市净率", pb)
    fm_rows += row("净资产收益率 ROE", roe_s)
    fm_rows += row("资产负债率", debt_s)
    fm_rows += row("净利润同比增长", pyoy_s)
    fm_rows += row("营业总收入同比增长", ryoy_s)
    fm_rows += row("最近报告期", report_date)

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>{symbol} · {pf} 形态</title>
<script src="{CDN_PLOTLY}"></script>
<style>
  body {{ font-family:Segoe UI, system-ui, sans-serif; background:#f4f5f7;
         color:#1a1a2e; max-width:1020px; margin:20px auto; padding:0 16px; }}
  .header {{ background:#fff; border-bottom:3px solid #4361ee; padding:12px 16px;
            border-radius:0 0 8px 8px; box-shadow:0 1px 3px rgba(0,0,0,.05); }}
  .header h1 {{ margin:0; font-size:21px; display:flex; align-items:center; }}
  .header .sub {{ color:#555; font-size:13px; margin-top:2px; }}
  .badge {{ display:inline-block; padding:2px 10px; border-radius:12px; font-size:12px; color:#fff; }}
  .good {{ background:#2a9d8f; }} .bad {{ background:#d62828; }} .neutral {{ background:#e9a23b; }}
  .score {{ float:right; font-size:22px; font-weight:800; color:#4361ee; margin-right:10px; }}
  .section {{ background:#fff; border:1px solid #e5e5ea; border-radius:8px;
             margin:11px 0; padding:13px 16px; box-shadow:0 1px 2px rgba(0,0,0,.04); }}
  .section h2 {{ font-size:14px; margin:0 0 9px 0; color:#3a0ca3;
                 border-bottom:1px solid #eee; padding-bottom:6px; }}
  table {{ border-collapse:collapse; width:auto; font-size:13px; background:#fafafa; }}
  td {{ padding:3px 8px; vertical-align:top; }}
  .grid > .section {{ margin-top:0; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:0 20px; }}
  .warn {{ color:#b00000; background:#ffeaea; padding:10px 12px; border-radius:6px;
          border:1px solid #ffcccc; font-size:13px; }}
  .kline-box {{ background:#fff; border:1px solid #e5e5ea; border-radius:8px;
               padding:8px; box-shadow:0 1px 2px rgba(0,0,0,.04); }}
  .kline-legend {{ font-size:12px; color:#555; margin-top:6px; text-align:right;
                  display:flex; flex-wrap:wrap; gap:10px; }}
  .kline-legend span {{ padding:2px 7px; border-radius:4px; background:#f5f5f5; }}
  a {{ color:#4361ee; text-decoration:none; font-weight:600; }}
</style>
</head>
<body>
<div class="header">
  <h1>{symbol} {name}<span class="badge {env_cls}" style="margin-left:8px">{env_fit}</span></h1>
  <div class="sub">形态 · {pf} · 环境 {regime} · 评分 {score}/100</div>
  <div class="score">{score}</div>
  <div style="clear:both"></div>
</div>

<div class="section">
  <h2>📈 日 K 线图表（含均线 + 信号/进场/止损标记）</h2>
  {kline_block}
</div>

<div class="grid">
  <div class="section">
    <h2>📊 基本面数据（最近报告期）</h2>
    <table>{fm_rows}</table>
  </div>

  <div class="section">
    <h2>🔍 形态信号详情</h2>
    <table>{sig_rows}</table>
  </div>
</div>

<div class="section">
  <h2>💰 交易参数（可执行口径）</h2>
  <table>
{entry_rows}
{ kl_rows}
  </table>
</div>

<div class="section" style="text-align:center">
  <a href="index.html">⬅️ 返回候选清单主面板</a>
</div>

</body>
</html>
"""

    return html


def gen_index(candidates: list[dict]) -> str:
    rows = ""
    for i, c in enumerate(candidates, 1):
        score = c.get("score", 0)
        env = c.get("env_fit", "-")
        pf = c.get("pattern", "")
        sym = c.get("symbol", "")
        sig = c.get("signal_date", "")
        rps = c.get("rps50")
        env_cls = "good" if env == "good" else ("bad" if env == "bad" else "neutral")
        bg = "background:#fff" if i % 2 == 0 else "background:#fafafa"
        rows += f"""        <tr style="{bg}">
          <td style="text-align:center">{i}</td>
          <td><a href="{sym}.html">{sym}</a></td>
          <td><span class="badge {env_cls}">{env}</span></td>
          <td style="font-weight:bold;color:#4361ee">{score}</td>
          <td>{pf}</td>
          <td>{sig}</td>
          <td>{fmt_price(rps) if isinstance(rps,(int,float)) else (rps or "-")}</td>
        </tr>
"""
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>SPS 候选标的 — 可视化主面板</title>
<style>
  body {{ font-family:Segoe UI, system-ui, sans-serif; background:#f4f5f7; color:#1a1a2e;
         max-width:1050px; margin:20px auto; padding:0 16px; }}
  h1 {{ border-bottom:3px solid #4361ee; padding-bottom:10px; font-size:22px; }}
  h2 {{ font-size:16px; margin:6px 0 12px; color:#3a0ca3; }}
  p {{ color:#555; font-size:13px; }}
  table {{ border-collapse:collapse; width:100%; background:#fff; font-size:13px;
          box-shadow:0 1px 2px rgba(0,0,0,.04); }}
  th,td {{ border:1px solid #ddd; padding:6px 10px; text-align:left; }}
  th {{ background:#eef2ff; }}
  .controls {{ display:flex; gap:8px; flex-wrap:wrap; margin:10px 0 14px;
              padding:10px; background:#fff; border:1px solid #e5e5ea; border-radius:8px; }}
  .controls a {{ color:#4361ee; text-decoration:none; padding:4px 10px; border-radius:6px;
                border:1px solid #4361ee; }}
  .controls a:hover {{ background:#4361ee; color:#fff; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; color:#fff; margin-left:5px; }}
  .good {{ background:#2a9d8f; }} .bad {{ background:#d62828; }} .neutral {{ background:#e9a03b; }}
  .note {{ font-size:12px; color:#999; margin-top:12px; }}
</style>
</head>
<body>
<h1>📋 SPS 候选标的 — 可视化主面板</h1>
<p>点击任意一行的 <b>标的代码</b> 跳转至单独的详情页：
日 K 线图（含均线、信号日/进场日/止损/关键位标记）、基本面卡片、形态信号详情、交易参数。</p>
<div class="controls">
  <a href="index.html?sort=score">按评分排序</a>
  <a href="index.html?sort=date">按信号日排序</a>
  <a href="index.html?sort=symbol">按代码排序</a>
  <a href="index.html?min=120">评分 ≥ 120</a>
  <a href="index.html?min=100">评分 ≥ 100</a>
  <a href="index.html?min=80">评分 ≥ 80</a>
  <a href="index.html?min=0">全部</a>
</div>
<h2>候选清单（共 {len(candidates)} 个结构形态信号）</h2>
<table>
  <thead><tr>
    <th>#</th><th>标的</th><th>适配</th><th>评分</th><th>形态</th><th>信号日</th><th>RPS50</th>
  </tr></thead>
  <tbody>
{rows}
  </tbody>
</table>
<p class="note">※ 本面板仅列 <b>结构形态信号</b>（W底、平台突破、杯柄、口袋支点）。
口袋支点作为佐证信号另列（参见 run_scan.py 控制台输出）。HTML 详情页的数据来自本地 Parquet 缓存与同花顺财务接口。</p>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="生成候选标的可视化 HTML")
    ap.add_argument("--min-score", type=int, default=0)
    ap.add_argument("--max-score", type=int, default=999)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--symbols", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out) if args.out else VIZ_DIR

    candidates = load_candidates()
    if args.symbols:
        syms = {s.strip() for s in args.symbols.split(",")}
        candidates = [c for c in candidates if c.get("symbol") in syms]

    if not args.all:
        candidates = [c for c in candidates
                     if args.min_score <= c.get("score", 0) <= args.max_score]
        if not args.limit:
            args.limit = 60
    if args.limit and not args.all:
        candidates = candidates[:args.limit]

    candidates.sort(key=lambda c: (c.get("score", 0),
                                    c.get("rps50") or 0, c.get("signal_date") or ""), reverse=True)

    generated = 0
    for i, c in enumerate(candidates, 1):
        symbol = c["symbol"]
        pf = c.get("pattern", "")
        fu = get_fundamental(symbol)
        html = make_candidate_html(symbol, c, fu)
        path = out_dir / f"{symbol}.html"
        path.write_text(html, encoding="utf-8")
        print(f"[{i:>3}/{len(candidates):>3}] {symbol} · {pf} · score={c.get('score')} → {path.name}")
        generated += 1

    idx_html = gen_index(candidates)
    (VIZ_DIR / "index.html").write_text(idx_html, encoding="utf-8")
    print(f"\n✅ 共生成 {generated} 个详情页 + index.html 于 {VIZ_DIR}")


if __name__ == "__main__":
    main()
