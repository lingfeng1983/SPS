"""炫酷可视化仪表盘生成器 v2。

生成两类HTML：
  - index.html    : 暗黑风格股票卡片网格仪表盘，支持客户端即时筛选/排序
  - details/*.html: 每只股票独立详情页（K线大图+均线+信号+基本面+形态键位）

依赖：
  - data/runs/candidates.json        (run_scan.py 输出)
  - data/runs/../daily/<symbol>*.parquet  (K线缓存)
  - data/meta/stock_list_*.parquet   (代码->名称映射)
  - fundamental/<symbol>.json         (同花顺财务缓存或实时拉取)

用法：
  python scripts/gen_dashboard.py --min-score 100 --limit 40
  python scripts/gen_dashboard.py --all
  python scripts/gen_dashboard.py --symbols 600519,000001,000002
  python scripts/gen_dashboard.py --recent 365  # 最近一年信号
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from sps.data import DATA_DIR
from sps.fundamental import get_fundamental

RUN_DIR = DATA_DIR / "runs"
DASH_DIR = RUN_DIR / "dashboard"
DETAIL_DIR = DASH_DIR / "details"
STOCK_LIST_DIR = DATA_DIR / "meta"

CDN_PLOTLY = "https://cdn.plot.ly/plotly-2.35.2.min.js"
CDN_FONT = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"


# ---------------------------------------------------------------- 辅助函数


def load_symbol_names() -> dict[str, str]:
    """读取最新股票列表缓存，返回 symbol -> name 字典。"""
    files = sorted(STOCK_LIST_DIR.glob("stock_list_*.parquet"),
                   key=lambda p: p.stat().st_mtime)
    if not files:
        return {}
    f = files[-1]
    try:
        df = pd.read_parquet(f)
    except Exception:
        return {}
    if "symbol" not in df.columns or "name" not in df.columns:
        return {}
    return dict(zip(df["symbol"].astype(str), df["name"].astype(str)))


def load_candidates() -> list[dict]:
    p = RUN_DIR / "candidates.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def candlestick_raw(symbol: str) -> dict | None:
    """从 Parquet 缓存读取原始 OHLCV + 日期，返回 dict 或 None。"""
    for f in sorted(DATA_DIR.glob(f"daily/{symbol}_*.parquet"),
                   key=lambda p: p.stat().st_mtime):
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if df.empty:
            continue
        # date 优先处理为 index，如有独立 date 列则覆盖
        if isinstance(df.index, pd.DatetimeIndex):
            dates = pd.to_datetime(df.index).values
            base = df
        else:
            if "date" not in df.columns:
                continue
            dates = pd.to_datetime(df["date"]).values
            # 扔掉所有名为 date 的列（不区分大小写）
            drop_cols = [c for c in df.columns if c.lower() == "date"]
            base = df.drop(columns=drop_cols)
        need = ["O", "H", "L", "C", "V"]
        if not set(need).issubset(base.columns):
            continue
        n = min(len(dates), *(len(base[c]) for c in need))
        return {
            "dates": pd.to_datetime(dates[:n]).tolist(),
            "O": base["O"].tolist()[:n],
            "H": base["H"].tolist()[:n],
            "L": base["L"].tolist()[:n],
            "C": base["C"].tolist()[:n],
            "V": base["V"].tolist()[:n],
        }
    return None


def fmt_price(v) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):,.2f}"
    except (ValueError, TypeError):
        return "-"


def fmt_pct(v, sign: bool = True) -> str:
    if v is None:
        return "-"
    try:
        s = f"{float(v):+.2f}" if sign else f"{float(v):.2f}"
        return f"{s}%"
    except (ValueError, TypeError):
        return "-"


def kline_json(symbol: str, event: dict) -> str:
    """生成 Plotly K线图表的data/layout JSON食用串。"""
    raw = candlestick_raw(symbol)
    if raw is None:
        return "{}"

    dates = raw["dates"]
    dopen = raw["O"]
    dhigh = raw["H"]
    dlow = raw["L"]
    dclose = raw["C"]
    dvol = raw["V"]

    n = len(dates)
    if n < 5:
        return "{}"

    # 均线
    def ma(x, w):
        s = []
        for i in range(n):
            if i < w - 1:
                s.append(None)
            else:
                s.append(sum(x[i - w + 1:i + 1]) / w)
        return s

    ma5 = ma(dclose, 5)
    ma10 = ma(dclose, 10)
    ma20 = ma(dclose, 20)

    # 信号日 / 进场日索引 (按日期字符串匹配)
    sig_idx = None
    ent_idx = None
    if event.get("signal_date"):
        try:
            sig_ts = pd.to_datetime(event["signal_date"])
            for i, d in enumerate(dates):
                if pd.Timestamp(d) == sig_ts:
                    sig_idx = i
                    break
        except Exception:
            pass
    if event.get("entry", {}).get("date"):
        try:
            ent_ts = pd.to_datetime(event["entry"]["date"])
            for i, d in enumerate(dates):
                if pd.Timestamp(d) == ent_ts:
                    ent_idx = i
                    break
        except Exception:
            pass

    # 止损线 / 形态键位
    entry = event.get("entry", {})
    stops = entry.get("stops", {})
    stop7 = stops.get("-0.07")
    kl = event.get("key_levels", {})

    shapes = []
    if sig_idx is not None:
        shapes.append({
            "type": "line", "x0": str(dates[sig_idx]), "x1": str(dates[sig_idx]),
            "y0": 0, "y1": 1, "yref": "paper",
            "line": {"color": "#2a9d8f", "width": 2, "dash": "dashdot"},
        })
    if ent_idx is not None:
        shapes.append({
            "type": "line", "x0": str(dates[ent_idx]), "x1": str(dates[ent_idx]),
            "y0": 0, "y1": 1, "yref": "paper",
            "line": {"color": "#ad9300", "width": 1.4, "dash": "dot"},
        })
    if stop7 is not None:
        shapes.append({
            "type": "line", "x0": str(dates[0]), "x1": str(dates[-1]),
            "y0": stop7, "y1": stop7,
            "line": {"color": "#d62828", "width": 1.6, "dash": "dash"},
        })
    for k, v in kl.items():
        if isinstance(v, (int, float)):
            shapes.append({
                "type": "line", "x0": str(dates[0]), "x1": str(dates[-1]),
                "y0": v, "y1": v,
                "line": {"color": "#3a0ca3", "width": 1.2, "dash": "dot"},
            })

    # 布局
    layout = {
        "shapes": shapes,
        "legend": {"orientation": "h", "y": -0.22, "x": 0.5, "xanchor": "center"},
        "xaxis": {
            "title": "日期", "rangeslider": {"visible": False},
            "rangeselector": {
                "buttons": [
                    {"count": 5, "label": "5日", "step": "day", "stepmode": "backward", "color": "#334155"},
                    {"count": 20, "label": "20日", "step": "day", "stepmode": "backward", "color": "#334155"},
                    {"count": 60, "label": "60日", "step": "day", "stepmode": "backward", "color": "#334155"},
                    {"count": 120, "label": "半年", "step": "day", "stepmode": "backward", "color": "#334155"},
                    {"step": "all", "label": "全体", "color": "#334155"},
                ]
            },
            "gridcolor": "#1f2937", "tickfont": {"color": "#94a3b8"},
        },
        "yaxis": {
            "title": "价格 (元)", "gridcolor": "#1f2937",
            "tickfont": {"color": "#94a3b8"},
        },
        "paper_bgcolor": "#0f172a", "plot_bgcolor": "#0f172a",
        "height": 520, "margin": {"l": 50, "r": 15, "t": 20, "b": 70},
    }

    data = [
        {
            "type": "candlestick",
            "x": dates, "open": dopen, "high": dhigh, "low": dlow, "close": dclose,
            "increasing": {"line": {"color": "#22c55e"}},
            "decreasing": {"line": {"color": "#ef4444"}},
            "name": "K线",
        },
        {"type": "scatter", "x": dates, "y": ma5, "mode": "lines", "name": "MA5",
         "line": {"color": "#f59e0b", "width": 1.2}, "connectgaps": False},
        {"type": "scatter", "x": dates, "y": ma10, "mode": "lines", "name": "MA10",
         "line": {"color": "#3b82f6", "width": 1.5}, "connectgaps": False},
        {"type": "scatter", "x": dates, "y": ma20, "mode": "lines", "name": "MA20",
         "line": {"color": "#ef4444", "width": 1.5}, "connectgaps": False},
    ]
    return json.dumps({"data": data, "layout": layout}, ensure_ascii=False,
                      default=str)


# ---------------------------------------------------------------- 详情页


def detail_html(symbol: str, event: dict, fu: dict, name: str) -> str:
    score = event.get("score", 0)
    env_fit = event.get("env_fit", "-")
    regime = event.get("regime", "-")
    rps50 = event.get("rps50")
    sig_date = event.get("signal_date", "-")
    avail = event.get("available_at", "-")
    pf = event.get("pattern", "-")
    entry = event.get("entry", {})
    entry_price = entry.get("price")
    entry_date = entry.get("date", "-")
    stops = entry.get("stops", {})
    stop7 = stops.get("-0.07")
    stop5 = stops.get("-0.05")
    stop10 = stops.get("-0.10")
    kl = event.get("key_levels", {})

    env_cls = "bg-emerald-500" if env_fit == "good" else (
             "bg-red-500" if env_fit == "bad" else "bg-amber-500")
    env_text = env_fit.capitalize() if env_fit else "未知"

    plot_json = kline_json(symbol, event)

    roe = fu.get("roe")
    debt = fu.get("debt_ratio")
    pyoy = fu.get("profit_yoy")
    ryoy = fu.get("revenue_yoy")
    pe_ttm = fu.get("pe_ttm")
    pb = fu.get("pb")
    report_date = fu.get("report_date", "-")

    fm_rows = ""
    fm_rows += "<tr><td class='lbl'>TTM 市盈率</td><td>" + (
        fmt_price(pe_ttm) if pe_ttm is not None else "N/A") + "</td></tr>"
    fm_rows += "<tr><td class='lbl'>市净率 PB</td><td>" + fmt_price(pb) + "</td></tr>"
    fm_rows += "<tr><td class='lbl'>净资产收益率 (ROE)</td><td>" + (
        f"{roe:.2f}%" if roe is not None else "N/A") + "</td></tr>"
    fm_rows += "<tr><td class='lbl'>资产负债率</td><td>" + (
        f"{debt:.2f}%" if debt is not None else "N/A") + "</td></tr>"
    fm_rows += "<tr><td class='lbl'>净利润同比增长</td><td>" + fmt_pct(pyoy, sign=False) + "</td></tr>"
    fm_rows += "<tr><td class='lbl'>营业总收入同比增长</td><td>" + fmt_pct(ryoy, sign=False) + "</td></tr>"
    fm_rows += "<tr><td class='lbl'>最近报告期</td><td>" + (
        report_date if report_date else "N/A") + "</td></tr>"

    sig_rows = ""
    sig_rows += "<tr><td class='lbl'>形态类别</td><td class='pk'>" + str(pf) + "</td></tr>"
    sig_rows += "<tr><td class='lbl'>信号日</td><td>" + str(sig_date) + "</td></tr>"
    sig_rows += "<tr><td class='lbl'>形态起始日</td><td>" + (
        event.get("pattern_start", "-") or "-") + "</td></tr>"
    sig_rows += "<tr><td class='lbl'>形态结束日</td><td>" + (
        event.get("pattern_end", "-") or "-") + "</td></tr>"
    sig_rows += "<tr><td class='lbl'>可用确认日</td><td>" + str(avail) + "</td></tr>"
    sig_rows += "<tr><td class='lbl'>大盘环境</td><td>" + str(regime) + "</td></tr>"
    sig_rows += "<tr><td class='lbl'>RPS(50)</td><td>" + (
        str(rps50) if rps50 is not None else "N/A") + "</td></tr>"
    sig_rows += "<tr><td class='lbl'>环境适配</td><td><span class='badge " + env_cls + "'>" + env_text + "</span></td></tr>"
    sig_rows += "<tr><td class='lbl'>环境评分</td><td class='pk'>" + str(score) + " / 100</td></tr>"

    kl_rows = ""
    if kl:
        for k, v in kl.items():
            if isinstance(v, (int, float)):
                kl_rows += "<tr><td class='lbl'>" + str(k).replace("_", " ").title() + "</td><td class='pk'>" + fmt_price(v) + " 元</td></tr>"
    else:
        kl_rows = "<tr><td colspan='2' class='muted'>— 该形态无独立关键位记录 —</td></tr>"

    entry_rows = ""
    entry_rows += "<tr><td class='lbl'>进场价格</td><td class='pk'>" + fmt_price(entry_price) + " 元</td></tr>"
    entry_rows += "<tr><td class='lbl'>进场日期</td><td>" + (
        str(entry_date) if entry_date != "-" else "未成交") + "</td></tr>"
    entry_rows += "<tr><td class='lbl'>进场口径</td><td>" + (
        "信号次日开盘价（可执行）" if entry_date != "-" else "理论价（次日未成交)") + "</td></tr>"
    entry_rows += "<tr><td class='lbl' style='color:#dc2626'>止损 -7% (欧奈尔)</td><td class='pk'>" + (
        fmt_price(stop7) + " 元" if stop7 else "N/A") + "</td></tr>"
    if stop5:
        entry_rows += "<tr><td class='lbl' style='color:#f59e0b'>止损 -5%</td><td>" + fmt_price(stop5) + " 元</td></tr>"
    else:
        entry_rows += "<tr><td class='lbl'>止损 -5%</td><td class='muted'>-</td></tr>"
    if stop10:
        entry_rows += "<tr><td class='lbl' style='color:#f59e0b'>止损 -10%</td><td>" + fmt_price(stop10) + " 元</td></tr>"
    else:
        entry_rows += "<tr><td class='lbl'>止损 -10%</td><td class='muted'>-</td></tr>"
    entry_rows += kl_rows

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{symbol} · {pf} — 形态详情</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">
<script src="{CDN_PLOTLY}"></script>
<style>
  :root {{
    --bg: #0f172a; --card: #1e293bd; --border: #334155; --muted: #94a3b8;
    --text: #f1f5f9; --accent: #3b82f6; --emerald: #22c55e; --red: #ef4444;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Inter', system-ui, sans-serif; background: var(--bg);
        color: var(--text); min-height: 100vh; padding: 18px 16px 60px; }}
  h1 {{ font-size: 1.55rem; font-weight: 700; letter-spacing: -0.02em; }}
  h2 {{ font-size: 1rem; font-weight: 600; color: #94a3b9; margin-bottom: 14px;
       display: flex; align-items: center; gap: 8px; }}
  .flex {{ display: flex; gap: 16px; }}
  .grow {{ flex: 1; }}
  .grid {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 16px; }}
  @media (max-width: 880px) {{ .grid, .flex {{ flex-direction: column; }} }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px;
          padding: 18px; box-shadow: 0 8px 20px rgba(0,0,0,.25); }}
  .header {{ margin-bottom: 18px; display: flex; align-items: center; justify-content: space-between;
            flex-wrap: wrap; }}
  .header .left {{ display: flex; align-items: center; gap: 14px; }}
  .header .title {{ display: flex; align-items: center; gap: 10px; }}
  .ticker {{ background: #3b82f6; color: #fff; font-weight: 700; padding: 5px 12px;
            border-radius: 8px; font-size: 1.1rem; letter-spacing: 0.02em; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 999px;
            font-size: 0.75rem; font-weight: 600; color: #fff; text-transform: uppercase;
            letter-spacing: 0.03em; }}
  .back {{ color: #94a3b8; text-decoration: none; font-size: 0.9rem; margin-bottom: 12px;
           display: inline-block; }}
  .back:hover {{ color: #f1f5f9; }}
  .plot {{ width: 100%; height: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid rgba(148,163,184,0.1); }}
  td.lbl {{ color: var(--muted); padding-right: 14px; white-space: nowrap; }}
  td.pk {{ font-weight: 600; text-align: right; }}
  tr.muted td {{ color: #64748b; padding-top: 0; }}
  tr.muted {{ border-bottom: none; }}
  .section {{ margin-bottom: 16px; }}
  .footer {{ color: var(--muted); font-size: 0.8rem; text-align: center; margin-top: 30px; }}
  .scale {{ display:inline-flex; align-items:center; gap: 6px; margin-left: 8px; }}
  .tag {{ display:inline-block; padding:2px 8px; border-radius:6px; ... }}
</style>
</head>
<body>

<div class="header">
  <div class="left">
    <a class="back" href="index.html">⬅ 仪表盘</a>
    <div class="title">
      <span class="ticker">{symbol}</span>
      <div>
        <h1>{name}</h1>
        <div style="color:var(--muted); font-size:0.9rem; margin-top:2px;">形态 · {pf} · 大盘环境 {regime}</div>
      </div>
    </div>
  </div>
  <span class="badge {env_cls}">{env_text}</span>
</div>

<div class="grid">
  <div class="card grow">
    <h2>📈 日 K线图 & 均线 & 信号标记</h2>
    <div id="chart" class="plot"></div>
    <div style="margin-top:8px; color:#94a3b8; font-size:0.85rem; line-height:1.6;">
      <b style="color:#22c55e">● 阳线</b>&nbsp;
      <b style="color:#ef4444">● 阴线</b>&nbsp;
      <b style="color:#f59e0b">─ MA5</b>&nbsp;
      <b style="color:#3b82f6">─ MA10</b>&nbsp;
      <b style="color:#ef4444">─ MA20</b>&nbsp;
      <span style="border-left:2px dashed #2a9d8f;padding-left:8px">信号日</span>&nbsp;
      <span style="border-left:1px dotted #ad9300;padding-left:8px">进场日</span>&nbsp;
      <span style="border-left:2px dashed #ef4444;padding-left:8px">止损 -7%</span>&nbsp;
      <span style="border-left:1px dotted #3a0ca3;padding-left:8px">形态键位</span>
    </div>
  </div>

  <div>
    <div class="card section">
      <h2>🔍 形态信号详情</h2>
      <table>{sig_rows}</table>
    </div>

    <div class="card section">
      <h2>💰 交易参数（可执行口径）</h2>
      <table>{entry_rows}</table>
    </div>
  </div>

  <div class="card section">
    <h2>📊 基本面数据（最近报告期）</h2>
    <table>{fm_rows}</table>
    <div style="color:#94a3b8; font-size:0.78rem; margin-top:8px;">数据来源：同花顺财务摘要 | 每次生成实时取最新报告期</div>
  </div>
</div>

<div class="footer">
  生成日期 {date.today().isoformat()} · SPS 形态识别系统 · 仅供研究学习，不构成投资建议
</div>

<script>
  var PLAN = {plot_json};
  var cfg = {{responsive:true, displayModeBar:false, displaylogo:false}};
  Plotly.newPlot('chart', PLAN.data, PLAN.layout, cfg);
</script>
</body>
</html>
"""
    return html


# ---------------------------------------------------------------- 仪表盘


def dashboard_html(candidates: list[dict], symbol_names: dict[str, str]) -> str:
    rows = []
    for i, c in enumerate(candidates, 1):
        score = c.get("score", 0)
        env = c.get("env_fit", "-")
        pf = c.get("pattern", "-")
        sym = c.get("symbol", "")
        sig = c.get("signal_date", "-")
        rps = c.get("rps50")
        entry = c.get("entry", {})
        entry_price = entry.get("price")
        env_cls = "bg-emerald-500" if env == "good" else (
                  "bg-red-500" if env == "bad" else "bg-amber-500")
        env_text = env.capitalize() if env else "未知"
        name = symbol_names.get(sym, "")
        bg = "#1e293b" if i % 2 == 0 else "#172033"
        rows.append(f"""
        <div class="card" data-score="{score}" data-pf="{pf}" data-env="{env}" data-rps="{rps if rps is not None else ''}" style="background:{bg};">
          <div class="inner">
            <div class="top">
              <span class="code"><a href="details/{sym}.html">{sym}</a></span>
              <span class="nm">{name}</span>
              <span class="badge {env_cls}" style="margin-left:auto">{env_text}</span>
            </div>
            <div class="mid">
              <span class="pf">{pf}</span>
              <span class="sig">{sig}</span>
            </div>
            <div class="scorebar">
              <span class="scorenum">{score}</span>
              <div class="track"><div class="fill" style="width:{score*1.2}%"></div></div>
            </div>
            <div class="stat">
              <span><b>RPS</b> {rps if rps is not None else "-"}</span>
              <span><b>信号日</b> {sig}</span>
              <span class="grow"></span>
              <span><b>进场</b> {fmt_price(entry_price)}</span>
            </div>
          </div>
        </div>
        """)

    total = len(candidates)
    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SPS 形态识别系统 · 候选标的仪表盘</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">
<style>
  :root {{
    --bg: #0b1120; --card: #1e293b; --card-alt: #172033; --border: #334155;
    --text: #f1f5f9; --muted: #94a3b8; --accent: #3b82f6; --emerald: #22c55e;
    --red: #ef4444; --amber: #f59e0b;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family:'Inter',system-ui,sans-serif; background:radial-gradient(circle at 20% 10%, #0f172a 0%, var(--bg) 60%); color:var(--text); min-height:100vh; padding:24px 16px 60px; }}
  h1 {{ font-size:1.8rem; font-weight:700; letter-spacing:-0.02em; display:flex; align-items:center; gap:12px; }}
  h1 small {{ font-size:0.9rem; color:var(--muted); font-weight:400; }}
  .container {{ max-width:1200px; margin:0 auto; }}
  .header {{ display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:16px; margin-bottom:18px; }}
  .sub {{ color:var(--muted); font-size:0.92rem; margin-top:4px; }}
  .controls {{ display:flex; flex-wrap:wrap; gap:10px; padding:14px 18px; background:#0f172a; border:1px solid var(--border); border-radius:12px; }}
  .field {{ display:flex; align-items:center; gap:8px; }}
  .field label {{ color:var(--muted); font-size:0.85rem; }}
  .field input, .field select {{ padding:6px 10px; border-radius:8px; background:#1e293bd; color:var(--text); border:1px solid var(--border); font:inherit; }}
  .field input:focus, .field select:focus {{ outline:none; border-color:var(--accent); }}
  .stats {{ display:flex; gap:20px; margin-left:auto; color:var(--muted); font-size:0.85rem; }}
  .stats b {{ color:var(--text); }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:14px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:16px; box-shadow:0 8px 24px rgba(0,0,0,.4); transition:transform .15s ease, box-shadow .15s ease; }}
  .card:hover {{ transform:translateY(-3px); box-shadow:0 16px 32px rgba(0,0,0,.6); }}
  .card .inner {{ pointer-events:none; }}
  .top {{ display:flex; align-items:center; gap:10px; margin-bottom:10px; }}
  .code {{ background:#3b82f6; color:#fff; font-weight:700; padding:5px 10px; border-radius:8px; font-size:1rem; text-decoration:none; }}
  .code:hover {{ background:#2563eb; }}
  .nm {{ font-size:0.9rem; color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:140px; flex:1; }}
  .badge {{ display:inline-block; padding:3px 9px; border-radius:999px; font-size:0.7rem; font-weight:600; color:#fff; text-transform:uppercase; letter-spacing:0.04em; }}
  .bg-emerald {{ background:#22c55e; }} .bg-red {{ background:#ef4444; }} .bg-amber {{ background:#f59e0b; }} .bg-muted {{ background:#475569; color:#fff; }}
  .mid {{ display:flex; justify-content:space-between; font-size:0.82rem; color:var(--muted); margin-bottom:8px; }}
  .mid .pf {{ font-weight:500; color:var(--accent); }}
  .scorebar {{ margin-bottom:10px; }}
  .scorenum {{ font-size:1.5rem; font-weight:700; color:var(--text); }}
  .track {{ height:6px; background:#0f172a; border-radius:3px; margin-top:2px; overflow:hidden; }}
  .fill {{ height:100%; background:linear-gradient(90deg,#3b82f6,#22c55e); border-radius:3px; }}
  .stat {{ display:flex; flex-wrap:wrap; gap:10px 16px; font-size:0.82rem; color:var(--muted); border-top:1px solid rgba(148,163,184,0.12); padding-top:8px; margin-top:4px; }}
  .stat b {{ color:var(--text); }}
  .empty {{ color:var(--muted); text-align:center; padding:40px; font-size:0.95rem; }}
  .empty b {{ color:#f1f5f9; display:block; margin-bottom:6px; }}
  .footer {{ color:var(--muted); font-size:0.8rem; text-align:center; margin-top:30px; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <h1>
        SPS 形态识别系统
        <small>牛股形态 · 环境适配型候选清单</small>
      </h1>
      <div class="sub">每个候选 = 结构形态信号 × 牛熊环境 × 相对强度(RPS)。点击代码进入单票详情页（K线+均线+信号+基本面）。</div>
    </div>
    <div class="stats">
      <span>总共 <b>{total}</b> 个候选</span>
    </div>
  </div>

  <div class="controls">
    <div class="field">
      <label>最低评分</label>
      <input id="scoreMin" type="number" min="0" max="100" value="0">
    </div>
    <div class="field">
      <label>最高评分</label>
      <input id="scoreMax" type="number" min="0" max="100" value="100">
    </div>
    <div class="field">
      <label>形态</label>
      <select id="pfFilter">
        <option value="">全部</option>{"".join(f"<option value='{pf}'>{pf}</option>" for pf in sorted(set(c.get("pattern","") for c in candidates)))}
      </select>
    </div>
    <div class="field">
      <label>环境适配</label>
      <select id="envFilter">
        <option value="">全部</option>
        <option value="good">good</option>
        <option value="neutral">neutral</option>
        <option value="bad">bad</option>
      </select>
    </div>
    <div class="field">
      <label>RPS最小值</label>
      <input id="rpsMin" type="number" min="0" max="1" step="0.1" value="0">
    </div>
  </div>
  </div>

  <div id="grid" class="grid">
    {''.join(rows)}
  </div>
  <div id="empty" class="empty" style="display:none">
    <b>没有符合条件的候选</b>
    尝试放宽评分或环境适配条件。
  </div>
</div>

<script>
  function filterCards() {{
    const sm = parseFloat(document.getElementById('scoreMin').value) || 0;
    const sx = parseFloat(document.getElementById('scoreMax').value) || 100;
    const pf = document.getElementById('pfFilter').value;
    const env = document.getElementById('envFilter').value;
    const rps = parseFloat(document.getElementById('rpsMin').value) || 0;
    const cards = document.querySelectorAll('.card');
    let shown = 0;
    cards.forEach(c => {{
      const sc = parseInt(c.dataset.score,10);
      const p = c.dataset.pf;
      const e = c.dataset.env;
      const r = parseFloat(c.dataset.rps) || 0;
      const ok = (sc >= sm && sc <= sx && (!pf || p===pf) && (!env || e===env) && r >= rps);
      c.style.display = ok ? '' : 'none';
      if (ok) shown++;
    }});
    document.getElementById('empty').style.display = shown === 0 ? '' : 'none';
  }}
  ['scoreMin','scoreMax','pfFilter','envFilter','rpsMin'].forEach(id =>
    document.getElementById(id).addEventListener('input', filterCards));
  filterCards();
</script>
<div class="footer">
  生成日期 {date.today().isoformat()} · SPS 形态识别系统 (规则 V1.1) · 仅供研究学习，不构成投资建议
</div>
</body>
</html>
"""
    return html


# ---------------------------------------------------------------- 主入口


def main() -> None:
    ap = argparse.ArgumentParser(description="生成炫酷形态仪表盘 + 详情页")
    ap.add_argument("--min-score", type=int, default=0)
    ap.add_argument("--max-score", type=int, default=999)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--symbols", type=str, default=None)
    ap.add_argument("--recent", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    DASH_DIR.mkdir(parents=True, exist_ok=True)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out) if args.out else DASH_DIR
    detail_dir = out_dir / "details"
    detail_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates()
    if not candidates:
        print("⚠️  candidates.json 不存在，先运行 run_scan.py 再来。")
        return

    print(f"加载候选 {len(candidates)} 个")

    # 根据条件过滤
    if args.symbols:
        syms = {s.strip() for s in args.symbols.split(",")}
        candidates = [c for c in candidates if c.get("symbol") in syms]
    if args.recent:
        cut = pd.Timestamp(date.today()) - timedelta(days=args.recent)
        candidates = [c for c in candidates
                      if c.get("signal_date") and pd.Timestamp(c["signal_date"]) >= cut]
    # 评分过滤 (不用于 --all / --symbols 时不限)
    candidates = [c for c in candidates
                  if args.min_score <= c.get("score", 0) <= args.max_score]
    if not args.all and not args.symbols and not args.recent and args.limit:
        candidates = candidates[:args.limit]

    # 按评分降序
    candidates.sort(key=lambda c: (c.get("score", 0), c.get("rps50") or 0,
                                   c.get("signal_date") or ""), reverse=True)

    symbol_names = load_symbol_names()
    print(f"已加载 {len(symbol_names)} 个股票名称映射（最新快照）")

    print(f"▶ 生成详情页 共 {len(candidates)} 个")
    generated = 0
    for i, c in enumerate(candidates, 1):
        sym = c["symbol"]
        fu = get_fundamental(sym)
        name = symbol_names.get(sym, "")
        html = detail_html(sym, c, fu, name)
        path = detail_dir / f"{sym}.html"
        path.write_text(html, encoding="utf-8")
        print(f"[{i:>3}/{len(candidates):>3}] {sym} ({name[:20]:20}) score={c.get('score')} → details/{sym}.html")
        generated += 1

    print(f"\n▶ 生成仪表盘 index.html（{len(candidates)} 个候选）")
    dash_html = dashboard_html(candidates, symbol_names)
    (out_dir / "index.html").write_text(dash_html, encoding="utf-8")
    print(f"\n✅ 完成 — 仪表盘: {out_dir / 'index.html'}")
    print(f"   详情页: {detail_dir}/")
    print(f"   打开方式: open_preview(file:/// {out_dir / 'index.html'})")


if __name__ == "__main__":
    main()
