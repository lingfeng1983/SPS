"""SPS 一体化交互界面（Flask 单页应用）。

一个页面集成全部功能：
  - 控制台：跑扫描（全市场/指定股票/最近N天）、生成报告
  - 候选列表：即时筛选（评分/形态/环境/RPS）
  - 详情面板：点候选 → K线图 + 形态信号 + 交易参数 + 基本面，同页切换不跳转

启动：
  .venv/Scripts/python scripts/app.py
  然后浏览器打开 http://127.0.0.1:5000
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if getattr(sys, "frozen", False):   # PyInstaller: 数据/模块解包目录
    sys.path.insert(0, sys._MEIPASS)  # noqa: SLF001

from flask import Flask, jsonify, render_template_string, request

from sps.data import DATA_DIR
from sps.fundamental import get_fundamental

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = DATA_DIR / "runs"
DASH_DIR = RUN_DIR / "dashboard"

app = Flask(__name__)

# ---------------------------------------------------------------- 后台任务管理

_job = {"running": False, "log": [], "done": False, "cmd": None,
        "proc": None, "progress": None}
_screen_task = {"active": False, "done": False, "progress": None, "result": None, "error": None}


def stop_job() -> bool:
    """终止正在运行的扫描进程。"""
    p = _job.get("proc")
    if p is not None and p.poll() is None:
        p.kill()
        _job["log"].append("[已手动停止]")
        return True
    return False


def _run_bg(cmd: list[str]):
    _job["running"] = True
    _job["done"] = False
    _job["log"] = []
    try:
        p = subprocess.Popen(
            cmd, cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        _job["proc"] = p
        for line in p.stdout:
            _job["log"].append(line.rstrip())
            # 保留最近500行
            if len(_job["log"]) > 500:
                _job["log"] = _job["log"][-500:]
        p.wait()
        _job["log"].append(f"[退出码 {p.returncode}]")
    except Exception as e:  # noqa: BLE001
        _job["log"].append(f"[异常] {e}")
    finally:
        _job["running"] = False
        _job["proc"] = None
        _job["done"] = True


def start_job(cmd: list[str]) -> bool:
    if _job["running"]:
        return False
    _job["cmd"] = cmd
    threading.Thread(target=_run_bg, args=(cmd,), daemon=True).start()
    return True


# ---------------------------------------------------------------- 数据读取


def load_candidates() -> list[dict]:
    p = RUN_DIR / "candidates.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def symbol_names() -> dict[str, str]:
    files = sorted((DATA_DIR / "meta").glob("stock_list_*.parquet"),
                   key=lambda p: p.stat().st_mtime)
    if not files:
        return {}
    try:
        df = __import__("pandas").read_parquet(files[-1])
        return dict(zip(df["symbol"].astype(str), df["name"].astype(str)))
    except Exception:
        return {}


def candle_raw(symbol: str) -> dict | None:
    """读K线缓存，返回 {dates,O,H,L,C,V} 或 None。"""
    import pandas as pd
    for f in sorted(DATA_DIR.glob(f"daily/{symbol}_*.parquet"),
                    key=lambda p: p.stat().st_mtime):
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if df.empty:
            continue
        if isinstance(df.index, pd.DatetimeIndex):
            dates = pd.to_datetime(df.index).values
            base = df
        else:
            if "date" not in df.columns:
                continue
            dates = pd.to_datetime(df["date"]).values
            base = df.drop(columns=[c for c in df.columns if c.lower() == "date"])
        need = ["O", "H", "L", "C", "V"]
        if not set(need).issubset(base.columns):
            continue
        n = min(len(dates), *(len(base[c]) for c in need))
        return {"dates": [str(x)[:10] for x in dates[:n]],
                **{c: base[c].tolist()[:n] for c in need}}
    return None


def build_plot(symbol: str, ev: dict) -> dict | None:
    """构造 Plotly K线 JSON（含均线+信号/进场/止损标记）。"""
    raw = candle_raw(symbol)
    if raw is None or len(raw["dates"]) < 5:
        return None
    dates = raw["dates"]
    c = raw["C"]
    n = len(dates)

    def ma(w):
        out, s = [], 0.0
        for i in range(n):
            s += c[i]
            if i >= w:
                s -= c[i - w]
            out.append(round(s / w, 3) if i >= w - 1 else None)
        return out

    shapes = []
    annotations = []
    sig = ev.get("signal_date")
    if sig and sig in dates:
        shapes.append({"type": "line", "x0": sig, "x1": sig,
                       "y0": 0, "y1": 1, "yref": "paper",
                       "line": {"color": "#22c55e", "width": 2, "dash": "dashdot"}})
        # 买入提示标注：信号日 K 线下方 ▲买入
        annotations.append({
            "x": sig, "y": float(raw["L"][dates.index(sig)]) * 0.985,
            "xref": "x", "yref": "y", "showarrow": True,
            "arrowhead": 2, "arrowcolor": "#22c55e", "arrowsize": 1.2,
            "text": "<b>▲触发信号(买)</b>", "font": {"color": "#22c55e", "size": 11},
            "ax": 0, "ay": 26})
    ent = (ev.get("entry") or {}).get("date")
    if ent and ent in dates:
        shapes.append({"type": "line", "x0": ent, "x1": ent,
                       "y0": 0, "y1": 1, "yref": "paper",
                       "line": {"color": "#f59e0b", "width": 1.4, "dash": "dot"}})
    stop7 = ((ev.get("entry") or {}).get("stops") or {}).get("-0.07")
    if stop7:
        shapes.append({"type": "line", "x0": dates[0], "x1": dates[-1],
                       "y0": stop7, "y1": stop7,
                       "line": {"color": "#ef4444", "width": 1.6, "dash": "dash"}})
        # 卖出(失效)提示标注：止损价左端 ✖失效离场
        annotations.append({
            "x": dates[0], "y": stop7,
            "xref": "x", "yref": "y", "showarrow": False,
            "text": "<b>✖失效参考(卖)</b>", "font": {"color": "#ef4444", "size": 11},
            "xanchor": "left", "yanchor": "bottom"})
    # 形态出现区间：紫色矩形框出 pattern_start ~ pattern_end
    ps, pe = ev.get("pattern_start"), ev.get("pattern_end")
    if ps and pe and ps in dates and pe in dates:
        seg_hi = max(raw["H"][dates.index(ps):dates.index(pe) + 1])
        seg_lo = min(raw["L"][dates.index(ps):dates.index(pe) + 1])
        shapes.append({"type": "rect", "x0": ps, "x1": pe,
                       "y0": seg_lo * 0.98, "y1": seg_hi * 1.02,
                       "line": {"color": "#8b5cf6", "width": 2},
                       "fillcolor": "rgba(139,92,246,0.06)"})
    kl = ev.get("key_levels") or {}
    for k, v in kl.items():
        if isinstance(v, (int, float)):
            shapes.append({"type": "line", "x0": dates[0], "x1": dates[-1],
                           "y0": v, "y1": v,
                           "line": {"color": "#8b5cf6", "width": 1.2, "dash": "dot"}})

    data = [
        {"type": "candlestick", "x": dates, "open": raw["O"], "high": raw["H"],
         "low": raw["L"], "close": raw["C"],
         "increasing": {"line": {"color": "#ef4444"}, "fillcolor": "#7f1d1d"},
         "decreasing": {"line": {"color": "#22c55e"}, "fillcolor": "#166534"},
         "name": "K线"},
        {"type": "scatter", "x": dates, "y": ma(5), "mode": "lines",
         "name": "MA5", "line": {"color": "#f59e0b", "width": 1.1}},
        {"type": "scatter", "x": dates, "y": ma(10), "mode": "lines",
         "name": "MA10", "line": {"color": "#3b82f6", "width": 1.3}},
        {"type": "scatter", "x": dates, "y": ma(20), "mode": "lines",
         "name": "MA20", "line": {"color": "#ec4899", "width": 1.3}},
    ]
    layout = {
        "shapes": shapes,
        "annotations": annotations,
        "legend": {"orientation": "h", "y": 1.02, "x": 0},
        "xaxis": {"rangeslider": {"visible": False}, "gridcolor": "#1e293b"},
        "yaxis": {"gridcolor": "#1e293b", "title": "价格"},
        "paper_bgcolor": "#0b1120", "plot_bgcolor": "#0b1120",
        "font": {"color": "#94a3b8"},
        "height": 460, "margin": {"l": 45, "r": 15, "t": 30, "b": 35},
    }
    return {"data": data, "layout": layout}


# ---------------------------------------------------------------- 路由


# ---------------------------------------------------------------- AI 解读（用户自带 API Key）

@app.route("/api/ai/config", methods=["GET"])
def api_ai_config_get():
    from sps.ai_interpret import masked_config
    return jsonify(masked_config())


@app.route("/api/ai/config", methods=["POST"])
def api_ai_config_set():
    from sps.ai_interpret import save_config, masked_config
    body = request.get_json(force=True) or {}
    save_config(body)
    return jsonify({"ok": True, **masked_config()})


@app.route("/api/ai/test", methods=["POST"])
def api_ai_test():
    from sps.ai_interpret import test_connection
    body = request.get_json(force=True) or {}
    cfg = {k: v for k, v in body.items() if k in ("base_url", "api_key", "model")} or None
    return jsonify(test_connection(cfg))


@app.route("/api/ai/interpret", methods=["POST"])
def api_ai_interpret():
    """对最近一次筛选结果做 AI 解读。"""
    from sps.ai_interpret import screen_interpretation
    if not SCREEN_LAST.get("result"):
        return jsonify({"ok": False, "error": "请先执行一次筛选"}), 400
    return jsonify(screen_interpretation(SCREEN_LAST["result"]))


@app.route("/api/data_status")
def api_data_status():
    """数据新鲜度：最新缓存K线日期与滞后天数。"""
    import datetime as _dt
    daily_dir = DATA_DIR / "daily"
    last = None
    if daily_dir.exists():
        files = sorted(daily_dir.glob("*.parquet"), key=lambda f: f.stat().st_mtime, reverse=True)
        for f in files[:3]:  # 取最近更新的几只看最后一根K线日期
            try:
                import pandas as pd
                df = pd.read_parquet(f)
                if len(df):
                    last = max(last, df.index[-1]) if last is not None else df.index[-1]
            except Exception:
                continue
    if last is None:
        return jsonify({"ready": False})
    last_date = pd.Timestamp(last).strftime("%Y-%m-%d")
    age_days = (pd.Timestamp.today().normalize() - pd.Timestamp(last).normalize()).days
    return jsonify({"ready": True, "last_date": last_date, "age_days": int(age_days)})


@app.route("/api/candidates")
def api_candidates():
    names = symbol_names()
    cands = load_candidates()
    for c in cands:
        c["_name"] = names.get(c.get("symbol", ""), "")
    return jsonify({"candidates": cands, "names_count": len(names)})


# ---------------------------------------------------------------- 参数化筛选 API

@app.route("/api/indicators")
def api_indicators():
    from sps.screener import available_indicators
    return jsonify({"indicators": available_indicators()})


# ---------------- 策略保存 / 参数回测成绩 ----------------

@app.route("/api/strategies", methods=["GET"])
def api_strategies():
    from sps.strategies import list_strategies
    return jsonify({"strategies": list_strategies()})


@app.route("/api/strategies", methods=["POST"])
def api_save_strategy():
    from sps.strategies import save_strategy
    body = request.get_json(force=True) or {}
    name = (body.get("name") or "").strip()
    conds = body.get("conditions") or {}
    if not name:
        return jsonify({"error": "策略名不能为空"}), 400
    if not conds:
        return jsonify({"error": "没有可保存的条件"}), 400
    rec = save_strategy(name, conds, float(body.get("stop_pct", 7.0)))
    return jsonify(rec)


@app.route("/api/strategies/<name>", methods=["DELETE"])
def api_del_strategy(name: str):
    from sps.strategies import delete_strategy
    return jsonify({"deleted": delete_strategy(name)})


@app.route("/api/param_stats")
def api_param_stats():
    """各指标参数网格的历史回测成绩（backtest_params.py 产出）。"""
    f = RUN_DIR / "param_stats.json"
    if not f.exists():
        return jsonify({"ready": False})
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
        return jsonify({"ready": True, "generated": d.get("generated"),
                        "stocks": d.get("stocks"), "stats": d.get("stats")})
    except Exception:
        return jsonify({"ready": False})


SCREEN_LAST = {"result": None}   # 最近一次筛选结果（AI 解读用）

@app.route("/api/screen", methods=["POST"])
def api_screen():
    """启动筛选任务（后台运行，通过 /api/screen_status 查询进度）。"""
    if _screen_task["active"]:
        return jsonify({"error": "已有筛选任务在运行中"}), 409
    body = request.get_json(force=True) or {}
    raw_conds = {k: v for k, v in (body.get("conditions") or {}).items()
                 if k in INDICATORS}
    conds, errors = validate_conditions(raw_conds)
    if errors:
        return jsonify({"error": "；".join(errors)}), 400
    if not conds:
        return jsonify({"error": "未选择任何条件"}), 400
    stop_pct = float(body.get("stop_pct", 7.0))
    _screen_task["active"] = True
    _screen_task["done"] = False
    _screen_task["progress"] = {"phase": "准备中", "pct": 0}
    _screen_task["result"] = None
    _screen_task["error"] = None
    threading.Thread(target=_do_screen_worker, args=(conds, stop_pct), daemon=True).start()
    return jsonify({"ok": True})


def _do_screen_worker(conds, stop_pct):
    try:
        import pandas as pd
        from sps.screener import screen, entry_and_stop, INDICATORS
        _screen_task["progress"] = {"phase": "加载日线文件", "pct": 0}
        daily = {}
        files = list(DATA_DIR.glob("daily/*.parquet"))
        total_files = len(files)
        for i, f in enumerate(files):
            sym = f.stem.split("_")[0]
            try:
                df = pd.read_parquet(f)
                if isinstance(df.index, pd.DatetimeIndex) and {"O","H","L","C","V"} <= set(df.columns):
                    daily[sym] = df
                elif "date" in df.columns:
                    df = df.set_index(pd.to_datetime(df["date"]))
                    if {"O","H","L","C","V"} <= set(df.columns):
                        daily[sym] = df[["O","H","L","C","V"]]
            except Exception:
                pass
            # 节流更新进度：每 50 个文件或最后一个文件时更新一次，避免 %100 稀疏导致进度条卡 0%
            if i % 50 == 0 or i == total_files - 1:
                pct = round((i + 1) / total_files * 50) if total_files else 50
                _screen_task["progress"] = {"phase": "加载日线文件", "pct": pct}
        if not daily:
            _screen_task["error"] = "无日线缓存，请先运行扫描"
            return
        _screen_task["progress"] = {"phase": "计算指标", "pct": 50}
        wide = pd.DataFrame({s: d["C"] for s, d in daily.items()})
        _screen_task["progress"] = {"phase": "计算指标", "pct": 75}
        out = screen(daily, wide, conds)
        _screen_task["progress"] = {"phase": "整理结果", "pct": 90}
        names = symbol_names()
        for rec in out["triggered"] + out["near"]:
            rec["_name"] = names.get(rec["symbol"], "")
        for rec in out["triggered"][:80]:
            df = daily.get(rec["symbol"])
            if df is None:
                continue
            pos = len(df) - 1
            rec.update(entry_and_stop(df, pos, stop_pct=stop_pct) or {})
            rec["has_kline"] = True
        near_keep = []
        for rec in out["near"]:
            rec["_name"] = names.get(rec["symbol"], "")
            if rec["symbol"] in daily:
                rec["has_kline"] = True
                near_keep.append(rec)
        out["near"] = near_keep
        out["scanned"] = len(daily)
        try:
            from sps.industry import get_industry_map
            ind_map = get_industry_map()
            if ind_map:
                from collections import Counter
                for grp in ("triggered", "near"):
                    for r in out[grp]:
                        sym = r["symbol"]
                        if sym in ind_map:
                            r["industry"] = ind_map[sym]
                        else:
                            r["industry"] = "沪主板(未分类)" if sym.startswith("6") else \
                                            "创业板(未分类)" if sym.startswith("3") else \
                                            "科创板(未分类)" if sym.startswith("68") else \
                                            "深主板(未分类)"
                ind_cnt = Counter(r["industry"] for r in out["triggered"])
                out["industry_summary"] = [
                    {"industry": k, "count": v} for k, v in ind_cnt.most_common(30)]
        except Exception as e:
            print(f"[warn] industry summary failed: {e}")
        # 观察池升级提醒是次要副产品：写失败/并发写不应影响筛选主结果，独立 try 隔离
        out["upgraded"] = []
        out["upgraded_from"] = None
        try:
            watch_file = RUN_DIR / "watchpool.json"
            prev = {}
            if watch_file.exists():
                try:
                    prev = json.loads(watch_file.read_text(encoding="utf-8"))
                except Exception:
                    prev = {}
            trig_syms = {r["symbol"] for r in out["triggered"]}
            upgraded = []
            for sym in trig_syms:
                if sym in prev.get("near", {}) and sym not in prev.get("triggered", set()):
                    upgraded.append({"symbol": sym, "name": names.get(sym, ""),
                                     "was_missing": prev["near"][sym]})
            watch_file.write_text(json.dumps({
                "updated": time.strftime("%Y-%m-%d %H:%M"),
                "triggered": {r["symbol"]: r.get("met", []) for r in out["triggered"]},
                "near": {r["symbol"]: [names.get(n, n) for n in r.get("missing", [])]
                         for r in out["near"]}}, ensure_ascii=False), encoding="utf-8")
            out["upgraded"] = upgraded
            out["upgraded_from"] = prev.get("updated")
        except Exception as e:
            print(f"[warn] watchpool 写入失败(不影响筛选结果): {e}")
        SCREEN_LAST["result"] = out
        _screen_task["result"] = out
        _screen_task["progress"] = {"phase": "完成", "pct": 100}
    except Exception as e:
        _screen_task["error"] = str(e)
    finally:
        _screen_task["active"] = False
        _screen_task["done"] = True


@app.route("/api/screen_status")
def api_screen_status():
    return jsonify({
        "active": _screen_task["active"],
        "done": _screen_task["done"],
        "progress": _screen_task["progress"],
        "error": _screen_task["error"]})

@app.route("/api/screen_result")
def api_screen_result():
    if _screen_task["result"] is None:
        return jsonify({"error": "无结果"}), 404
    return jsonify(_screen_task["result"])


# ---------------------------------------------------------------- 持仓管理（卖出侧）

def _load_daily_min():
    """筛选/诊断共用的日线缓存载入（精简版）。"""
    import pandas as pd
    daily = {}
    for f in DATA_DIR.glob("daily/*.parquet"):
        sym = f.stem.split("_")[0]
        try:
            df = pd.read_parquet(f)
            if isinstance(df.index, pd.DatetimeIndex) and {"O","H","L","C","V"} <= set(df.columns):
                daily[sym] = df
        except Exception:
            continue
    return daily


@app.route("/api/positions_replay", methods=["POST"])
def api_positions_replay():
    """持仓自检回放：对每只 open 持仓，检查其买入日是否满足当前筛选条件。"""
    import pandas as pd
    from sps.positions import load_positions
    from sps.screener import INDICATORS, validate_conditions
    body = request.get_json(force=True) or {}
    raw = {k: v for k, v in (body.get("conditions") or {}).items() if k in INDICATORS}
    conds, errors = validate_conditions(raw)
    if errors:
        return jsonify({"error": "；".join(errors)}), 400
    if not conds:
        return jsonify({"error": "未选择任何条件"}), 400
    daily = _load_daily_min()
    names = symbol_names()
    opens = [r for r in load_positions() if r["status"] == "open"]
    results = []
    for rec in opens:
        sym = rec["symbol"]
        df = daily.get(sym)
        entry = {"symbol": sym, "name": names.get(sym, rec.get("name", "")),
                 "entry_date": rec.get("entry_date", ""),
                 "pnl_pct": None, "would_trigger": False,
                 "met_count": 0, "total": len(conds), "missing": []}
        if df is None or df.empty:
            entry["note"] = "no_kline"
            results.append(entry)
            continue
        day_ts = pd.Timestamp(rec["entry_date"])
        sub = df.loc[:day_ts]
        if sub.empty or len(sub) < 60:
            entry["note"] = "entry_date_no_data"
            results.append(entry)
            continue
        pos = len(sub) - 1
        # 买入当日是否满足条件（用买入日当天收盘数据）
        met, missing = [], []
        for name, param in conds.items():
            meta = INDICATORS.get(name)
            if meta is None:
                continue
            try:
                series = meta["fn"](df, param)
                ok = bool(series.iloc[pos]) if series is not None and len(series) > pos else False
            except Exception:
                ok = False
            (met if ok else missing).append(name)
        entry["met_count"] = len(met)
        entry["missing"] = missing
        entry["would_trigger"] = not missing
        # 持有至今收益
        price = float(df["C"].iloc[-1])
        entry_price = float(rec.get("entry_price") or 0)
        if entry_price > 0:
            entry["pnl_pct"] = round((price / entry_price - 1) * 100, 2)
        results.append(entry)
    return jsonify({"results": results})


@app.route("/api/positions", methods=["GET"])
def api_positions():
    from sps.positions import load_positions, run_diagnosis, EXIT_RULES
    daily = _load_daily_min()
    diag = run_diagnosis(daily)
    names = symbol_names()
    for r in diag["results"]:
        r.setdefault("name", names.get(r["symbol"], ""))
    hist = [r for r in load_positions() if r["status"] == "closed"]
    return jsonify({"diagnosis": diag, "history": hist[-50:],
                    "exit_rules": {k: {kk: vv for kk, vv in v.items()}
                                   for k, v in EXIT_RULES.items()}})


@app.route("/api/positions", methods=["POST"])
def api_add_position():
    from sps.positions import add_position, validate_rules
    body = request.get_json(force=True) or {}
    rules, errors = validate_rules(body.get("rules") or {})
    if errors:
        return jsonify({"error": "；".join(errors)}), 400
    if not rules:
        return jsonify({"error": "请至少选择一条卖出规则"}), 400
    try:
        rec = add_position(
            symbol=str(body["symbol"]).strip(),
            name=str(body.get("name") or "").strip(),
            entry_price=float(body["entry_price"]),
            entry_date=str(body.get("entry_date") or date.today()),
            stop_pct=float(body.get("stop_pct") or 7.0),
            rules=rules, qty=body.get("qty"))
        return jsonify(rec)
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/positions/<symbol>/close", methods=["POST"])
def api_close_position(symbol: str):
    from sps.positions import close_position
    body = request.get_json(force=True) or {}
    rec = close_position(symbol, float(body["exit_price"]),
                         str(body.get("exit_date") or date.today()),
                         reason=str(body.get("reason") or "manual"))
    if rec is None:
        return jsonify({"error": "未找到该持仓"}), 404
    return jsonify(rec)


@app.route("/api/positions/<symbol>", methods=["DELETE"])
def api_del_position(symbol: str):
    from sps.positions import delete_position
    return jsonify({"deleted": delete_position(symbol)})


@app.route("/api/backtest_exit", methods=["POST"])
def api_backtest_exit():
    """卖出规则组合的历史回测（随机入场口径）。"""
    from sps.positions import backtest_exit_rules, validate_rules
    body = request.get_json(force=True) or {}
    rules, errors = validate_rules(body.get("rules") or {})
    if errors:
        return jsonify({"error": "；".join(errors)}), 400
    if not rules:
        return jsonify({"error": "请至少选择一条卖出规则"}), 400
    daily = _load_daily_min()
    return jsonify(backtest_exit_rules(daily, rules,
                                       stop_pct=float(body.get("stop_pct") or 7.0)))


@app.route("/api/backtest_combo", methods=["POST"])
def api_backtest_combo():
    """组合回测：对当前勾选的完整条件组合跑历史信号统计。

    与单指标徽章互补——回答"这套组合整体胜率如何"。
    """
    import numpy as np
    import pandas as pd
    from sps.screener import screen, validate_conditions
    from sps.positions import COST_PER_TRADE
    body = request.get_json(force=True) or {}
    raw = {k: v for k, v in (body.get("conditions") or {}).items()
           if k in __import__("sps.screener", fromlist=["INDICATORS"]).INDICATORS}
    conds, errors = validate_conditions(raw)
    if errors:
        return jsonify({"error": "；".join(errors)}), 400
    if not conds:
        return jsonify({"error": "未选择任何条件"}), 400
    daily = _load_daily_min()
    if not daily:
        return jsonify({"error": "无日线缓存"}), 400
    wide = pd.DataFrame({s: d["C"] for s, d in daily.items()})
    rps = __import__("sps.screener", fromlist=["_rps_series"])._rps_series(wide, 50)
    latest = max(df.index[-1] for df in daily.values())
    cut = latest - pd.Timedelta(days=180)   # 与参数回测同口径的样本外切分

    fwd20_is, fwd20_oos = [], []
    n_sig = 0
    for sym, df in daily.items():
        if len(df) < 120:
            continue
        df.attrs["symbol"] = sym
        all_true = None
        try:
            for name, param in conds.items():
                meta = __import__("sps.screener", fromlist=["INDICATORS"]).INDICATORS[name]
                s = meta["fn"](df, param, rps=rps)
                if s is None or s.empty:
                    s = pd.Series(False, index=df.index)
                all_true = s if all_true is None else (all_true & s)
        except Exception:
            continue
        if all_true is None:
            continue
        sig = all_true & ~all_true.shift(1).fillna(False).astype(bool)
        for pos in np.where(sig.values)[0][-60:]:
            epos = int(pos) + 1
            if epos + 19 >= len(df):
                continue
            entry = float(df["O"].iloc[epos])
            if entry <= 0:
                continue
            pnl = (float(df["C"].iloc[epos + 19]) / entry - 1 - COST_PER_TRADE) * 100
            n_sig += 1
            (fwd20_oos if df.index[int(pos)] >= cut else fwd20_is).append(pnl)
    if not n_sig:
        return jsonify({"n_signals": 0})
    a_all = np.array(fwd20_is + fwd20_oos)
    out = {"n_signals": n_sig,
           "win20": round(float((a_all > 0).mean()) * 100, 1),
           "avg20": round(float(a_all.mean()), 2)}
    if len(fwd20_is) >= 30 and len(fwd20_oos) >= 15:
        a_is, a_oos = np.array(fwd20_is), np.array(fwd20_oos)
        out["is_win20"] = round(float((a_is > 0).mean()) * 100, 1)
        out["oos_win20"] = round(float((a_oos > 0).mean()) * 100, 1)
    return jsonify(out)


@app.route("/api/replay", methods=["POST"])
def api_replay():
    """历史回放：把全部日线截断到指定日期，重跑筛选，并统计其后20日真实收益。"""
    import pandas as pd
    from sps.screener import screen, entry_and_stop, INDICATORS, validate_conditions
    body = request.get_json(force=True) or {}
    raw_conds = {k: v for k, v in (body.get("conditions") or {}).items()
                 if k in INDICATORS}
    conds, errors = validate_conditions(raw_conds)
    if errors:
        return jsonify({"error": "；".join(errors)}), 400
    if not conds:
        return jsonify({"error": "未选择任何条件"}), 400
    day = str(body.get("day") or "")
    day_ts = pd.Timestamp(day)
    daily = _load_daily_min()
    if not daily:
        return jsonify({"error": "无日线缓存"}), 400
    names = symbol_names()
    # 截断到回放日
    truncated = {}
    for sym, df in daily.items():
        sub = df.loc[:day_ts]
        if len(sub) >= 60:
            truncated[sym] = sub
    if not truncated:
        return jsonify({"error": f"{day} 之前没有足够K线数据"}), 400
    wide = pd.DataFrame({s: d["C"] for s, d in truncated.items()})
    out = screen(truncated, wide, conds)
    stop_pct = float(body.get("stop_pct") or 7.0)
    fwd20s = []
    for rec in out["triggered"][:100]:
        df = truncated.get(rec["symbol"])
        if df is None:
            continue
        rec["_name"] = names.get(rec["symbol"], "")
        pos = len(df) - 1
        rec.update(entry_and_stop(df, pos, stop_pct=stop_pct) or {})
        # 其后20日真实收益（含成本，买入口径 t+1 开盘）
        full = daily[rec["symbol"]]
        epos_f = len(full.loc[:day_ts])     # 全量数据中的位置
        if epos_f + 19 < len(full):
            entry = float(full["O"].iloc[epos_f])
            if entry > 0:
                from sps.positions import COST_PER_TRADE
                fwd = (float(full["C"].iloc[epos_f + 19]) / entry - 1) * 100
                fwd -= COST_PER_TRADE * 100
                rec["fwd20"] = round(fwd, 2)
                fwd20s.append(fwd)
    summary = {}
    if fwd20s:
        import numpy as np
        a = np.array(fwd20s)
        summary = {"win20": round(float((a > 0).mean()) * 100, 1),
                   "avg20": round(float(a.mean()), 2), "n": len(a)}
    out["day"] = str(max(df.index[-1] for df in truncated.values()).date())
    out["asof"] = str(max(df.index[-1] for df in daily.values()).date())
    out["summary"] = summary
    return jsonify(out)


# ---------------------------------------------------------------- 详情页

DETAIL_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>SPS 标的详情</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--bg:#0b1120;--panel:#111a2e;--card:#16213a;--border:#253352;
      --text:#e2e8f0;--muted:#8ea0bd;--accent:#3b82f6;--green:#22c55e;
      --red:#ef4444;--amber:#f59e0b}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:"Segoe UI",system-ui,sans-serif;
     padding:22px 26px;max-width:1280px;margin:0 auto}
a.back{color:var(--accent);text-decoration:none;font-size:13.5px}
h1{font-size:21px;margin:12px 0 4px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.sub{color:var(--muted);font-size:13px;margin-bottom:18px}
.badge{padding:3px 10px;border-radius:99px;font-size:11px;font-weight:700;color:#fff}
.b-good{background:var(--green);color:#052e16}.b-bad{background:var(--red)}
.b-neutral{background:var(--amber);color:#3b2708}
.grid{display:grid;grid-template-columns:1fr 330px;gap:16px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--border);border-radius:13px;padding:15px;margin-bottom:14px}
.card h4{font-size:12px;color:var(--muted);letter-spacing:.05em;text-transform:uppercase;margin-bottom:10px}
table{width:100%;border-collapse:collapse;font-size:13px}
td{padding:6px 4px;border-bottom:1px solid #ffffff0a}
td.lbl{color:var(--muted)}td.val{text-align:right;font-weight:600}
.evtag{display:inline-block;padding:3px 10px;border-radius:8px;background:#0b1120;border:1px solid var(--border);
       font-size:12px;margin:0 6px 6px 0;cursor:pointer}
.evtag.on{border-color:var(--accent);color:var(--accent)}
.health{font-size:14px;font-weight:700}
.ok{color:var(--green)}.no{color:var(--red)}.na{color:var(--muted)}
.legend{font-size:11.5px;color:var(--muted);margin-top:8px;line-height:1.7}
.buybox{display:grid;grid-template-columns:1fr 1fr 1fr;gap:9px;margin-top:4px}
.bk{background:#0b1120;border:1px solid var(--border);border-radius:10px;padding:10px;text-align:center}
.bk .t{font-size:11px;color:var(--muted)}.bk .v{font-size:17px;font-weight:800;margin-top:3px}
.v-buy{color:var(--accent)}.v-stop{color:var(--red)}
</style>
</head>
<body>
<a class="back" href="/">← 返回候选列表</a>
<h1 id="ttl">加载中…</h1>
<div class="sub" id="sub"></div>

<div class="grid">
  <div class="card">
    <h4>日K线 — 触发信号与失效参考</h4>
    <div id="kchart"></div>
    <div class="legend">
      <span style="color:#22c55e">▲触发信号(买)</span> 信号日收盘确认，次日开盘进场 &nbsp;
      <span style="color:#ef4444">✖失效参考(卖)</span> 收盘跌破此红横线=买点逻辑失效，无条件离场 &nbsp;
      <span style="color:#8b5cf6">┄ 紫虚线</span> MA20趋势生命线
    </div>
  </div>
  <div>
    <div class="card">
      <h4>🩺 基本面体检 — 决定是否买入</h4>
      <div id="health" class="health"></div>
      <table id="fmtab"></table>
    </div>
    <div class="card">
      <h4>💰 买卖点与止损</h4>
      <div class="buybox">
        <div class="bk"><div class="t">买点(次日开盘)</div><div class="v v-buy" id="bp">-</div></div>
        <div class="bk"><div class="t">止损 -7%</div><div class="v v-stop" id="sp7">-</div></div>
        <div class="bk"><div class="t">关键位(重点监控)</div><div class="v" id="kl">-</div></div>
      </div>
      <table style="margin-top:9px" id="tradetab"></table>
    </div>
    <div class="card">
      <h4>✅ 当前筛选条件满足情况</h4>
      <table id="sigtab"></table>
    </div>
    <div class="card" id="histCard" style="display:none">
      <h4>📈 这只票的历史信号回放（当前条件 · 最多近8次）</h4>
      <table id="histtab"></table>
    </div>
  </div>
</div>

<script>
const SYM = "{{ symbol }}";
const pct = v => v==null?'-':(v>0?'+':'')+v.toFixed(2)+'%';

// 从 URL 取筛选条件（列表跳转时带上）
const QS = new URLSearchParams(location.search);
const CONDS = JSON.parse(QS.get("conds")||"{}");
const STOP_PCT = QS.get("stop_pct")||7;

async function load(){
  const q = "?conds="+encodeURIComponent(JSON.stringify(CONDS))+"&stop_pct="+STOP_PCT;
  const r = await fetch('/api/screen_detail/'+SYM+q);
  if(!r.ok){
    document.getElementById('ttl').textContent = SYM+' 暂无K线数据';
    document.getElementById('sub').textContent = '请先在主界面点「更新全市场数据缓存」';
    return;
  }
  const j = await r.json();
  const fu = j.fundamental||{};
  renderHealth(fu);
  renderTrade(j.trade||{}, j);
  renderConds(j.cond_status||[]);
  renderHist(j.history||[]);
  if(j.plot) Plotly.newPlot('kchart', j.plot.data, j.plot.layout,
                            {responsive:true, displayModeBar:false});
}

function renderHealth(fu){
  const checks=[
    {k:'净利润同比',v:fu.profit_yoy,ok:v=>v>0,need:'>0'},
    {k:'营收同比',v:fu.revenue_yoy,ok:v=>v>0,need:'>0'},
    {k:'ROE',v:fu.roe,ok:v=>v>=10,need:'≥10%'},
    {k:'毛利率',v:fu.gross_margin,ok:v=>v>=20,need:'≥20%'},
    {k:'资产负债率',v:fu.debt_ratio,ok:v=>v<=70,need:'≤70%'},
  ];
  const known=checks.filter(c=>c.v!=null);
  const pass=known.filter(c=>c.ok(c.v)).length;
  const grade=known.length===0?'未知':(pass>=4?'优秀':pass>=3?'良好':'偏弱');
  const gc=known.length===0?'na':(pass>=4?'ok':pass>=3?'health':'no');
  document.getElementById('ttl').innerHTML =
    `${SYM} ${fu._name||j_name||''} <span class="health ${gc}" style="margin-left:10px">体检：${grade} (${pass}/${known.length||5})</span>`;
  document.getElementById('fmtab').innerHTML=checks.map(c=>{
    const p=c.v!=null&&c.ok(c.v);
    const cls=c.v==null?'na':(p?'ok':'no');
    return `<tr><td class="lbl"><span class="${cls}">●</span> ${c.k}</td>
      <td class="val ${cls}">${pct(c.v)}</td><td class="val na" style="font-weight:400;font-size:11px">${c.need}</td></tr>`;
  }).join('')+`<tr><td class="lbl na">最近报告期</td><td class="val na" colspan="2"
     style="font-weight:400;font-size:11px">${fu.report_date||'-'}</td></tr>`;
}

let j_name='';
function renderTrade(t, j){
  document.getElementById('bp').textContent=t.entry_price??'-';
  document.getElementById('bp').title='参考'+(t.pending?'今日收盘':'次日开盘');
  document.getElementById('sp7').textContent=t.stop_price??'-';
  document.getElementById('sp7').title='-'+(t.stop_pct??7)+'%止损';
  // 关键位：用 MA20 / 近20日低点 / 前高 作为重点关注价格，而非"明日开盘"废话
  const kl = j.key_levels||{};
  const items = [
    kl.ma20!=null && ['MA20均线', kl.ma20, '跌破则趋势转弱'],
    kl.stop_price!=null && ['止损位', kl.stop_price, '收盘跌破无条件卖出'],
    kl.entry_price!=null && ['买点(参考)', kl.entry_price, t.pending?'次日开盘附近挂单':'已触发，可进场'],
  ].filter(Boolean);
  document.getElementById('kl').innerHTML = items.length
    ? items.map(([n,v,tip])=>`<div style="font-size:11px;color:var(--muted);font-weight:400" title="${tip}">${n}</div><div style="font-size:14px">${v}</div>`).join('')
    : '-';
  document.getElementById('tradetab').innerHTML=`
    <tr><td class="lbl">进场口径</td><td class="val">${t.pending?'次日开盘附近(参考今日收盘 '+t.entry_price+')':'信号次日开盘价'}</td></tr>
    <tr><td class="lbl">止损比例</td><td class="val">${t.stop_pct??7}%</td></tr>` +
    (kl.stop_price!=null?`<tr><td class="lbl">止损价</td><td class="val" style="color:var(--red)">${kl.stop_price}</td></tr>`:'');
}

function renderConds(list){
  document.getElementById('sigtab').innerHTML=list.map(c=>{
    const cls=c.ok?'ok':'no';
    const risks=(c.risks||[]).map(r=>`<li>${r}</li>`).join('');
    const watch=(c.watch_points||[]).map(w=>`<li>${w}</li>`).join('');
    const invalid=(c.invalidation||[]).map(i=>`<li>${i}</li>`).join('');
    return `<tr><td class="lbl"><span class="${cls}">${c.ok?'✓':'✗'}</span> ${c.label}</td>
      <td class="val ${cls}">${c.ok?'满足':'未满足'}</td>
      <td style="padding:0 8px;vertical-align:top;max-width:320px">
        ${risks?`<div style="font-size:10.5px;color:#fca5a5">⚠ 风险</div><ul style="margin:1px 0 0 14px;font-size:10.5px;color:var(--muted)">${risks}</ul>`:''}
        ${watch?`<div style="font-size:10.5px;color:#fbbf24">👁 观察</div><ul style="margin:1px 0 0 14px;font-size:10.5px;color:var(--muted)">${watch}</ul>`:''}
        ${invalid?`<div style="font-size:10.5px;color:#22c55e">✖ 失效</div><ul style="margin:1px 0 0 14px;font-size:10.5px;color:var(--muted)">${invalid}</ul>`:''}
      </td></tr>`;
  }).join('')||'<tr><td class="lbl na">无条件信息</td></tr>';
}

function renderHist(hist){
  if(!hist||!hist.length) return;
  document.getElementById('histCard').style.display='';
  const wins=hist.filter(h=>h.fwd20>0).length;
  document.getElementById('histtab').innerHTML=
    `<tr><td class="lbl na" colspan="3" style="font-size:11.5px">历史 ${hist.length} 次触发 ·
      20日胜率 <b style="color:${wins/hist.length>=0.5?'var(--green)':'var(--red)'}">${Math.round(wins/hist.length*100)}%</b>
      （含成本，买点=信号次日开盘）</td></tr>` +
    hist.slice().reverse().map(h=>`
      <tr><td class="lbl">${h.date}</td>
        <td class="val">买点 ${h.entry}</td>
        <td class="val" style="color:${h.fwd20>=0?'var(--green)':'var(--red)'}">${h.fwd20>0?'+':''}${h.fwd20}%</td></tr>`).join('');
}

load();
</script>
</body>
</html>
"""


@app.route("/api/screen_detail/<symbol>")
def api_screen_detail(symbol: str):
    """参数化筛选标的的详情：K线+基本面（不依赖形态事件）。

    query: conds=JSON字符串(条件)，stop_pct=止损比例
    返回K线(标注今日信号)+基本面+买点止损。
    """
    import pandas as pd
    from sps.screener import INDICATORS, entry_and_stop
    from flask import request as _req
    df = None
    for f in DATA_DIR.glob(f"daily/{symbol}_*.parquet"):
        try:
            raw = pd.read_parquet(f)
        except Exception:
            continue
        if isinstance(raw.index, pd.DatetimeIndex):
            df = raw
        elif "date" in raw.columns:
            df = raw.set_index(pd.to_datetime(raw["date"]))
        if df is not None and {"O", "H", "L", "C", "V"} <= set(df.columns):
            break
        df = None
    if df is None or not {"O", "H", "L", "C", "V"} <= set(df.columns):
        return jsonify({"error": "no kline"}), 404

    fu = get_fundamental(symbol)
    names = symbol_names()
    # 逐条评估当前勾选的条件 → 每条的满足情况
    try:
        conds = json.loads(_req.args.get("conds") or "{}")
    except Exception:
        conds = {}
    stop_pct = float(_req.args.get("stop_pct") or 7)
    cond_status = []
    for name, param in conds.items():
        meta = INDICATORS.get(name)
        if meta is None:
            continue
        try:
            series = meta["fn"](df, param)
            ok = bool(series.iloc[-1]) if series is not None and len(series) else False
        except Exception:
            ok = False
        cond_status.append({"name": name, "label": meta["label"], "ok": ok,
                            "risks": meta.get("risks", []),
                            "watch_points": meta.get("watch_points", []),
                            "invalidation": meta.get("invalidation", [])})

    pos = len(df) - 1
    trade = entry_and_stop(df, pos, stop_pct=stop_pct) or {}
    # 单只标的历史信号回放：该股过去触发过几次当前条件、随后20日各赚多少
    hist = []
    try:
        import numpy as np
        from sps.positions import COST_PER_TRADE
        all_true = None
        for name, param in conds.items():
            meta = INDICATORS.get(name)
            if meta is None:
                continue
            s = meta["fn"](df, param)
            if s is None or s.empty:
                s = pd.Series(False, index=df.index)
            all_true = s if all_true is None else (all_true & s)
        if all_true is not None:
            sig = all_true & ~all_true.shift(1).fillna(False).astype(bool)
            sig_pos = [int(i) for i in np.where(sig.values)[0] if i + 20 < len(df)][-8:]
            for p in sig_pos:
                entry = float(df["O"].iloc[p + 1])
                if entry <= 0:
                    continue
                pnl = round((float(df["C"].iloc[p + 20]) / entry - 1 - COST_PER_TRADE) * 100, 2)
                hist.append({"date": str(df.index[p].date()), "entry": round(entry, 2), "fwd20": pnl})
    except Exception:
        hist = []
    # 关键位：MA20（趋势生命线）、止损价、买点
    ma20 = round(float(df["C"].rolling(20).mean().iloc[-1]), 2) if len(df) >= 20 else None
    key_levels = {"ma20": ma20,
                  "stop_price": trade.get("stop_price"),
                  "entry_price": trade.get("entry_price")}

    # K线图：标注最新信号 + 全部历史信号的买卖点
    plot = build_plot(symbol, {"signal_date": str(df.index[-1].date()),
                               "entry": {"price": trade.get("entry_price"),
                                         "date": str(df.index[-1].date()),
                                         "stops": {"-0.07": trade.get("stop_price")}},
                               "key_levels": {"MA20": ma20} if ma20 else {}})
    if plot:
        lay = plot["layout"]
        # 最新信号：买点参考橙竖线
        if trade.get("entry_price"):
            lay["shapes"].append({
                "type": "line", "x0": str(df.index[-1].date()), "x1": str(df.index[-1].date()),
                "y0": 0, "y1": 1, "yref": "paper",
                "line": {"color": "#f59e0b", "width": 1.4, "dash": "dot"}})
        # 每一次历史信号：绿色小▲标在信号日K线下方（含盈亏文字）
        for h in hist:
            d = h["date"]
            if d in [str(x)[:10] for x in df.index]:
                row = df.loc[:d]
                lo = float(row["L"].iloc[-1])
                color = "#22c55e" if h["fwd20"] >= 0 else "#ef4444"
                lay["annotations"].append({
                    "x": d, "y": lo * 0.97,
                    "xref": "x", "yref": "y", "showarrow": True,
                    "arrowhead": 2, "arrowcolor": color, "arrowsize": 1.0,
                    "text": f"▲{h['fwd20']:+}%" if h["fwd20"] >= 0 else f"▲{h['fwd20']}%",
                    "font": {"color": color, "size": 10}, "ax": 0, "ay": 18})
    return jsonify({"symbol": symbol, "_name": names.get(symbol, ""),
                    "fundamental": fu, "plot": plot,
                    "trade": trade, "cond_status": cond_status,
                    "key_levels": key_levels,
                    "history": hist})


@app.route("/api/plot/<symbol>")
def api_plot(symbol: str):
    """按指定信号日画图（详情页切换形态事件时用）。"""
    from flask import request as _req
    sig = _req.args.get("sig")
    evs = [e for e in load_candidates() if e.get("symbol") == symbol]
    if not evs:
        return jsonify({"error": "no event"}), 404
    ev = next((e for e in evs if e.get("signal_date") == sig), None) \
        or sorted(evs, key=lambda e: e.get("score", 0), reverse=True)[0]
    return jsonify({"plot": build_plot(symbol, ev)})


def _is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def _run_scan_inthread(n: int):
    """打包版：进程内跑扫描（stdout 重定向到任务日志）。"""
    import io
    from contextlib import redirect_stdout

    class _Log(io.TextIOBase):
        def write(self, s):
            for ln in s.rstrip("\n").split("\n"):
                if ln:
                    _job["log"].append(ln)
            return len(s)

    def worker():
        _job["running"] = True
        _job["done"] = False
        _job["log"] = []
        try:
            with redirect_stdout(_Log()):
                if getattr(sys, "frozen", False):
                    from run_scan import run          # PyInstaller 已打包
                    from sps.data import get_all_symbols
                    uni = get_all_symbols(include_etf=False, exclude_st_bj=True)
                    syms = uni["symbol"].tolist()
                    if n > 0:
                        syms = syms[:n]
                    km = dict(zip(uni["symbol"], uni.get("kind", "stock")))
                else:
                    sys.path.insert(0, str(ROOT / "scripts"))
                    from run_scan import run
                    from sps.data import get_all_symbols
                    uni = get_all_symbols(include_etf=False, exclude_st_bj=True)
                    syms = uni["symbol"].tolist()
                    if n > 0:
                        syms = syms[:n]
                    km = dict(zip(uni["symbol"], uni.get("kind", "stock")))
                run(syms, kind_map=km)
            _job["log"].append("[完成]")
        except Exception as e:  # noqa: BLE001
            _job["log"].append(f"[错误] {e}")
        finally:
            _job["running"] = False
            _job["done"] = True

    threading.Thread(target=worker, daemon=True).start()


@app.route("/api/run", methods=["POST"])
def api_run():
    body = request.get_json(force=True) or {}
    action = body.get("action", "scan")
    n = int(body.get("max_stocks") or 0)
    if action == "scan" and _is_frozen():
        if _job["running"]:
            return jsonify({"ok": False})
        _run_scan_inthread(n)
        return jsonify({"ok": True})
    py = str(ROOT / ".venv" / "Scripts" / "python.exe")
    if action == "scan":
        cmd = [py, str(ROOT / "scripts" / "run_scan.py")]
        if n > 0:
            cmd += ["--max-stocks", str(n)]
    elif action == "dashboard":
        ms = int(body.get("min_score", 100))
        cmd = [py, str(ROOT / "scripts" / "gen_dashboard.py"), "--min-score", str(ms)]
    elif action == "report":
        cmd = [py, str(ROOT / "scripts" / "build_report.py")]
    else:
        return jsonify({"ok": False, "error": "unknown action"}), 400
    ok = start_job(cmd)
    return jsonify({"ok": ok})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    return jsonify({"stopped": stop_job()})



@app.route("/api/job")
def api_job():
    progress = None
    if _job["log"]:
        for line in reversed(_job["log"][-20:]):
            if "[progress]" in line:
                try:
                    parts = line.split("[progress]")[-1].strip().split("/")
                    cur, tot = int(parts[0]), int(parts[1])
                    progress = {"current": cur, "total": tot, "pct": round(cur/tot*100) if tot else 0}
                except (ValueError, IndexError):
                    pass
                break
    return jsonify({"running": _job["running"], "done": _job["done"],
                    "log": _job["log"][-40:], "total_lines": len(_job["log"]),
                    "progress": progress})


@app.route("/detail/<symbol>")
def detail_page(symbol: str):
    return render_template_string(DETAIL_HTML, symbol=symbol)


@app.route("/")
def index():
    return render_template_string(PAGE_HTML)


# ---------------------------------------------------------------- 页面

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>SPS 牛股形态识别系统</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--bg:#0b1120;--panel:#111a2e;--card:#16213a;--border:#253352;
      --text:#e2e8f0;--muted:#8ea0bd;--accent:#3b82f6;--green:#22c55e;
      --red:#ef4444;--amber:#f59e0b}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);
     font-family:"Segoe UI",system-ui,sans-serif;height:100vh;display:flex;overflow:hidden}
/* 左侧控制台 */
#side{width:300px;min-width:220px;max-width:60vw;background:var(--panel);border-right:1px solid var(--border);
      padding:18px;display:flex;flex-direction:column;gap:14px;overflow-y:auto}
#splitter{width:5px;cursor:col-resize;background:transparent;flex-shrink:0;position:relative;z-index:5}
#splitter:hover,#splitter.on{background:var(--accent)}
h1{font-size:17px;display:flex;align-items:center;gap:8px}
h1 .logo{background:linear-gradient(135deg,#3b82f6,#22c55e);-webkit-background-clip:text;
         background-clip:text;color:transparent;font-weight:800;font-size:20px}
.sec{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:13px}
.sec h3{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:9px}
button{width:100%;padding:9px;border:none;border-radius:9px;background:var(--accent);color:#fff;
       font-weight:600;cursor:pointer;font-size:13px;margin-bottom:7px;transition:.15s}
button:hover{filter:brightness(1.15)}
button:disabled{opacity:.45;cursor:not-allowed}
button.green{background:var(--green);color:#052e16}
button.gray{background:#33415580}
.row{display:flex;gap:8px;align-items:center;margin-bottom:8px;font-size:12.5px;color:var(--muted)}
.row input,.row select{flex:1;padding:6px 8px;border-radius:7px;background:#0b1120;
       border:1px solid var(--border);color:var(--text);font:inherit;width:70px}
/* 日志 */
#log{background:#05080f;border:1px solid var(--border);border-radius:9px;padding:9px;
     font-family:Consolas,monospace;font-size:11px;line-height:1.55;color:#7dd3fc;
     height:180px;overflow-y:auto;white-space:pre-wrap}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;background:var(--green)}
.dot.busy{background:var(--amber);animation:pulse 1s infinite}
@keyframes pulse{50%{opacity:.35}}
/* 主区 */
#main{flex:1;overflow-y:auto;padding:14px 18px}
.cand{background:var(--card);border:1px solid var(--border);border-radius:11px;padding:11px 13px;
      margin-bottom:9px;cursor:pointer;transition:.12s}
.cand:hover{transform:translateX(3px);border-color:var(--accent)}
.cand.sel{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
.cand .r1{display:flex;align-items:center;gap:8px;margin-bottom:5px}
.cand .code{font-weight:700;color:var(--accent);font-size:14px}
.cand .nm{color:var(--text);font-size:13px;flex:1;overflow:hidden;text-overflow:ellipsis;
          white-space:normal;word-break:break-all;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;line-height:1.35}
.badge{padding:2px 8px;border-radius:99px;font-size:10.5px;font-weight:700;color:#fff}
.b-good{background:var(--green);color:#052e16}.b-bad{background:var(--red)}
.b-neutral{background:var(--amber);color:#3b2708}
.cand .r2{display:flex;gap:12px;font-size:11.5px;color:var(--muted)}
.cand .r2 b{color:var(--text)}
.scorebig{font-size:19px;font-weight:800;color:var(--accent)}
/* 详情 */
#detail{flex:1;overflow-y:auto;padding:16px;display:none}
#detail h2{font-size:16px;margin-bottom:4px}
#detail .sub{color:var(--muted);font-size:12px;margin-bottom:14px}
.dgrid{display:grid;grid-template-columns:2fr 1fr;gap:14px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:13px}
.card h4{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:9px}
table{width:100%;border-collapse:collapse;font-size:12.5px}
td{padding:5px 4px;border-bottom:1px solid #ffffff0a}
td.lbl{color:var(--muted)}
td.val{text-align:right;font-weight:600}
.evtag{display:inline-block;padding:2px 9px;border-radius:7px;background:#0b1120;
       border:1px solid var(--border);font-size:11.5px;margin:0 5px 5px 0;cursor:pointer}
.evtag:hover{border-color:var(--accent)}
#empty-detail{color:var(--muted);text-align:center;margin:auto;font-size:13.5px}
</style>
</head>
<body>

<div id="side">
  <h1><span class="logo">SPS</span> 牛股形态识别系统</h1>
  <div class="sec" id="secTpl">
    <h3 style="cursor:pointer;user-select:none;display:flex;align-items:center;justify-content:space-between" onclick="toggleTpl()">
      <span>① 选个风格，一键开筛<span id="tplCount" style="color:var(--accent);font-weight:700;margin-left:6px"></span></span>
      <span id="tplArrow" style="font-size:10px">▾ 展开</span></h3>
    <div id="tplBody" style="display:none">
    <div id="tplBox" style="display:flex;flex-direction:column;gap:7px"></div>
    <div class="row" style="margin-top:8px">🛡 止损%
      <input id="stopPctTpl" type="number" value="7" step="0.5" min="1" max="20" aria-label="止损百分比" style="width:60px">
      <span style="font-size:10.5px;color:var(--muted)">风控参数，不参与选股</span></div>
    <button class="green" style="margin-top:4px" onclick="doScreen()">🎯 开始筛选</button>
    </div>
  </div>

  <div class="sec" id="secScreen">
    <h3 style="cursor:pointer;user-select:none;display:flex;align-items:center;justify-content:space-between" onclick="toggleAdv()">
      <span>② 自定义指标（进阶）</span><span id="advArrow" style="font-size:10px">▾ 展开</span></h3>
    <div id="advBody" style="display:none">
    <div id="indForm" style="font-size:12px;max-height:340px;overflow-y:auto"></div>
    <div class="row">🛡 止损% <input id="stopPct" type="number" value="7" step="0.5" min="1" max="20" aria-label="止损百分比" style="width:60px">
      <span style="font-size:10.5px;color:var(--muted)">风控参数，不参与选股</span></div>
    </div>
    <button style="background:#33415580;margin-top:6px" onclick="doBacktestCombo()">🧪 ① 组合回测（先验证胜率）</button>
    <div id="comboResult" style="font-size:11.5px;color:var(--muted);margin-top:2px" role="status"></div>
    <div class="row" style="margin-top:8px;flex-wrap:wrap">
      <button class="green" onclick="doScreen()">🎯 ② 开始筛选</button>
      <button style="background:#33415580" onclick="saveStrategy()">💾 保存为策略</button>
      <button style="background:#7f1d1d" onclick="clearAllConds()" title="取消全部已选风格与勾选指标，恢复默认参数并清空结果列表">🧹 清空重筛</button>
    </div>
    <div style="margin-top:10px">
      <h3>📁 我的策略</h3>
      <div id="strategyList"></div>
    </div>
    <div class="row" style="font-size:11px;color:var(--muted)">
      ↯徽章 = 该参数的历史20日胜率(样本数)，绿≥55% 黄≥45% 红&lt;45%</div>
  </div>

  <div class="sec" style="font-size:11.5px;color:var(--muted);line-height:1.6">
    参数化选股：选风格 → 全市场筛选 → 已触发=买点(次日开盘)，止损=买点×(1-止损%)。<br>
    💡 <b style="color:var(--text)">组合回测用法</b>：先选好条件（点风格卡或自选），点「组合回测」，系统用这组条件模拟历史每天选股，给出胜率/平均收益（已扣手续费）。胜率&lt;45%的建议换条件。
  </div>
</div>
<div id="splitter" title="拖动调整左右宽度"></div>

<div id="main">
  <div id="freshBar" style="display:flex;gap:10px;align-items:center;font-size:11.5px;color:var(--muted);margin-bottom:10px;padding:7px 12px;background:var(--card);border:1px solid var(--border);border-radius:8px">
    <span class="dot" id="freshDot"></span><span id="freshText">检查数据新鲜度…</span>
    <div id="freshProgress" style="display:none;flex:1;margin-left:8px;max-width:200px">
      <div style="height:6px;background:#1e293b;border-radius:3px;overflow:hidden">
        <div id="freshProgressBar" style="height:100%;width:0%;background:var(--accent);border-radius:3px;transition:width 0.3s"></div>
      </div>
      <div id="freshProgressText" style="font-size:10.5px;margin-top:2px;color:var(--muted)"></div>
    </div>
    <button onclick="runScanData()" id="freshBtn" style="width:auto;padding:3px 10px;margin:0;font-size:11px;background:#33415580">⬇ 更新数据</button>
    <button onclick="openAISettings()" style="width:auto;padding:3px 10px;margin:0;margin-left:auto;font-size:11px;background:#33415580">⚙ AI 设置</button>
  </div>
  <div id="tabbar" style="display:flex;gap:8px;margin-bottom:12px">
    <button onclick="showTab('filter')" id="tab-filter" class="tabbtn on" style="width:auto;padding:7px 18px;margin:0;border-radius:8px;background:var(--accent)">🔍 筛选</button>
    <button onclick="showTab('pos')" id="tab-pos" class="tabbtn" style="width:auto;padding:7px 18px;margin:0;border-radius:8px;background:#33415580">💼 持仓管理</button>
    <button onclick="showTab('replay')" id="tab-replay" class="tabbtn" style="width:auto;padding:7px 18px;margin:0;border-radius:8px;background:#33415580">⏪ 历史回放</button>
  </div>
  <div id="view-filter">
    <div id="tplDetailBox" style="display:none"></div>
    <div id="list"></div>
    </div>
  <div id="view-pos" style="display:none"></div>
  <div id="view-replay" style="display:none"></div>
</div>

<script>
const $ = id => document.getElementById(id);

// 左右分栏拖拽调宽（记忆上次宽度）
(function(){
  const sp=document.getElementById('splitter'), side=document.getElementById('side');
  if(!sp) return;
  const saved=localStorage.getItem('sps_side_w');
  if(saved) side.style.width=saved+'px';
  let dragging=false;
  sp.addEventListener('mousedown',e=>{dragging=true; sp.classList.add('on'); document.body.style.cursor='col-resize'; e.preventDefault();});
  window.addEventListener('mousemove',e=>{
    if(!dragging) return;
    const w=Math.min(Math.max(e.clientX, 220), window.innerWidth*0.6);
    side.style.width=w+'px';
  });
  window.addEventListener('mouseup',()=>{
    if(!dragging) return;
    dragging=false; sp.classList.remove('on'); document.body.style.cursor='';
    localStorage.setItem('sps_side_w', parseInt(side.style.width));
  });
})();
const PAT_CN={FLAT_BREAKOUT:'平台突破',W_BOTTOM:'W底',CUP_HANDLE:'杯柄',
              POCKET_PIVOT:'口袋支点'};

async function runScanData(){
  // 更新数据缓存：拉取全市场日线+财务（供筛选与回测使用）
  const body={action:'scan', max_stocks:0};
  const r=await fetch('/api/run',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const j=await r.json();
  if(!j.ok) alert('已有任务在运行中');
  else { setFresh('running'); pollJobFresh(); }
}

// ================= 数据新鲜度 =================
function setFresh(state, text){
  const dot=$('freshDot'), t=$('freshText');
  if(!dot) return;
  dot.className='dot'+(state==='busy'?' busy':'');
  dot.style.background = state==='stale' ? 'var(--amber)' :
                         state==='running' ? 'var(--amber)' : 'var(--green)';
  if(text) t.textContent=text;
}
async function checkFresh(){
  try{
    const j=await (await fetch('/api/data_status')).json();
    if(!j.ready){ setFresh('stale','数据未就绪'); return; }
    const days=j.age_days;
    if(days<=1) setFresh('ok',`数据截至 ${j.last_date} ✅ 最新`);
    else if(days<=4) setFresh('ok',`数据截至 ${j.last_date}（${days}天前，可能为节假日）`);
    else setFresh('stale',`⚠️ 数据截至 ${j.last_date}，已落后 ${days} 天，建议更新`);
  }catch(e){ setFresh('ok','数据状态未知'); }
}
let _freshPoll=null;
function pollJobFresh(){
  if(_freshPoll) return;
  _freshPoll=setInterval(async()=>{
    try{
      const j=await (await fetch('/api/job')).json();
      if(!j.running){ clearInterval(_freshPoll); _freshPoll=null; checkFresh(); return; }
      const prog=j.progress;
      if(prog && prog.total>10){
        document.getElementById('freshProgress').style.display='';
        document.getElementById('freshProgressBar').style.width=prog.pct+'%';
        document.getElementById('freshProgressText').textContent=`${prog.current}/${prog.total} (${prog.pct}%)`;
        setFresh('running',`更新数据中…`);
      } else {
        document.getElementById('freshProgress').style.display='none';
        setFresh('running','更新数据中…');
      }
    }catch(e){}
  },2000);
}

// ================= 策略模板卡片（P0） =================
const TEMPLATES=[
 {key:'trend', icon:'🐢', name:'稳健趋势', tag:'趋势+量', desc:'站上20日线 + 放量确认',
  conds:{above_ma:20, vol_ratio:1.5}, stop_pct:7,
  plain:'一句话：只买"趋势走好 + 有真金白银进场"的票。20日线站上去=最近一个月买的人平均赚钱、抛压小；成交量放大=有资金确认，不是虚涨。两者同时满足才出手，胜率比单看一个更高。',
  detail:{
   title:'稳健趋势 — 为什么这样选？',
   logic:`<b>核心思想：</b>只买"趋势已经走好、且有真金白银确认"的票。<br><br>
    <b>条件1 · 站上20日线：</b>20日均线约等于一个月的交易成本线，收盘站上去代表过去一个月买入的人平均是赚钱的，抛压小、趋势向上。<br><br>
    <b>条件2 · 量比≥1.5：</b>当日成交量是20日均量的1.5倍以上。价格涨了但没量可能是假突破；放量上涨说明有资金真金白银进场。<br><br>
    <b>历史依据：</b>全市场回测（5500+只，含手续费），站上20日线20日胜率约46.6%，样本外不衰减。<br><br>
    <b>适合：</b>震荡向上的市场。 <b>不适合：</b>单边下跌市。`,
   exit:'卖出参考：跌破20日线或触发止损价（买点×(1-止损%)）任一即离场。'}},
 {key:'breakout', icon:'🚀', name:'突破追涨', tag:'强势+量', desc:'距52周新高≤10% + 放量确认',
  conds:{above_ma:60, near_high:10, vol_ratio:2}, stop_pct:7,
  plain:'一句话：买"快要创一年新高"的最强票，赌它继续涨。离新高只差10%=上方没有套牢盘、涨过去毫无阻力；要求成交量放大到平时2倍=确认是真突破不是骗炮。⚠️追高天生胜率低，所以三个条件缺一不可，止损必须严格执行。',
  detail:{
   title:'突破追涨 — 为什么这样选？',
   logic:`<b>核心思想：</b>买"即将创一年新高"的强势票，赌动量延续。<br><br>
    <b>条件1 · 距52周新高≤10%：</b>上方套牢盘已消化，突破后无解套抛压（欧奈尔CANSLIM核心买点）。<br><br>
    <b>条件2 · 站上60日线：</b>确认是上升趋势中的接近新高，而非下跌反抽。<br><br>
    <b>条件3 · 量比≥2：</b>突破必须放巨量，缩量新高大概率假突破。<br><br>
    <b>⚠️ 诚实提示：</b>A股直接追高胜率仅38-41%（均值回归剧烈），务必三条件同时满足+严格止损。<br><br>
    <b>适合：</b>市场强势、主线明确时。 <b>不适合：</b>弱势市。`,
   exit:'卖出参考：跌回新高下方3%或触发止损价即离场，不恋战。'}},
 {key:'pullback', icon:'🎯', name:'缩量回踩', tag:'趋势+位置', desc:'上升趋势回调企稳，低吸',
  conds:{above_ma:60, pullback_stable:5}, stop_pct:7,
  plain:'一句话：在上涨趋势里等回调结束再上车，买在别人恐慌时。先确认大趋势向上（60日线上方），然后等它回调超过5%后连续5天站稳——回调结束的信号。买点比别人低，止损空间更大。回测里表现最好的位置类因子。',
  detail:{
   title:'缩量回踩 — 为什么这样选？',
   logic:`<b>核心思想：</b>上升趋势中等回调结束再上车，买在别人恐慌时。<br><br>
    <b>条件1 · 站上60日线：</b>确认大趋势向上，不接下跌的飞刀。<br><br>
    <b>条件2 · 回撤企稳：</b>从高点回撤≥5%（真回调过）且连续5天收稳于5日线上（企稳了）。<br><br>
    <b>历史依据：</b>全场回测最好的位置类因子——20日胜率49.1%。买点成本低，止损空间大。<br><br>
    <b>适合：</b>有主线的牛市/结构市，龙头股回调低吸。 <b>不适合：</b>下降趋势（回踩会变下跌中继）。`,
   exit:'卖出参考：跌破企稳平台低点或止损价即离场。'}},
 {key:'vol_boom', icon:'💥', name:'放量启动', tag:'量+动量', desc:'连续放量上涨 ≥2天 + 当日涨≥3%',
  conds:{up_days:2, gain_today:3}, stop_pct:7,
  plain:'一句话：抓"资金连续进场 + 今天突然发力"的启动点。连续2天上涨且放量=有资金在偷偷吸筹；今天再涨3%以上=吸筹结束开始拉升。买在启动初期，相当于跟随主力进场。',
  detail:{
   title:'放量启动 — 为什么这样选？',
   logic:`<b>核心思想：</b>捕捉主力吸筹完成、开始拉升的启动点。<br><br>
    <b>条件1 · 连续2天放量上涨：</b>资金持续流入的最稳健信号（回测20日均收益+0.8~1%）。<br><br>
    <b>条件2 · 当日涨幅≥3%：</b>吸筹结束、开始发力的确认。调低会抓到更多假启动。<br><br>
    <b>适合：</b>题材轮动期的早期跟随。 <b>不适合：</b>尾盘追（买点已抬高，止损空间小）。`,
   exit:'卖出参考：跌破5日线或止损价离场；启动失败（次日跌回启动价）立即走。'}},
 {key:'ma_bull', icon:'🐂', name:'均线多头', tag:'趋势', desc:'多头排列 ≥3天',
  conds:{ma_align:3}, stop_pct:7,
  plain:'一句话：只买"短期线在中期线上方、中期线在长期线上方"的标准上涨结构票。5日、10日、20日线从上到下排好队=短中长线持有者全部赚钱、没人急着卖，是最教科书式的健康上涨形态。',
  detail:{
   title:'均线多头 — 为什么这样选？',
   logic:`<b>核心思想：</b>买教科书式的健康上涨结构。<br><br>
    <b>条件 · 多头排列≥3天：</b>5日>10日>20日线且持续3天以上=短中长线买入者全部获利、无人急卖，趋势最扎实。单条件胜率46-47%中游，适合做组合基底。<br><br>
    <b>适合：</b>趋势市做底仓条件，常与其他条件组合。 <b>不适合：</b>单独使用（信号太泛）。`,
   exit:'卖出参考：5日线下穿10日线（死叉）或触发止损价即离场。'}},
 {key:'box_squeeze', icon:'📦', name:'横盘蓄势', tag:'位置+波动', desc:'20日振幅≤10% + 波动收敛≤3%',
  conds:{box_amp:10, vol_narrow:3}, stop_pct:7,
  plain:'一句话：找"憋了很久、马上要选方向"的票。股价一个月内波动不超过10%、最近波动率还收敛到3%以内=多空暂时平衡、筹码高度稳定。这种"弹簧压到最紧"的状态一旦配合放量突破，爆发力最强。⚠️本条件只选"蓄势"，不含方向——建议搭配趋势条件使用。',
  detail:{
   title:'横盘蓄势 — 为什么这样选？',
   logic:`<b>核心思想：</b>买"弹簧压到最紧"的蓄势状态，等方向选择。<br><br>
    <b>条件1 · 20日振幅≤10%：</b>横盘蓄势，振幅越小筹码越稳，突破爆发力越强（Darvas箱体）。<br><br>
    <b>条件2 · ATR波动≤3%：</b>波动压缩（Bollinger Squeeze前置形态），方向爆发的前奏。<br><br>
    <b>⚠️ 提示：</b>蓄势≠必涨，可能向下突破。建议与趋势条件组合使用。<br><br>
    <b>适合：</b>与趋势/量价条件组合。 <b>不适合：</b>单独使用。`,
   exit:'卖出参考：若按突破进场，跌回箱体上沿即离场。'}},
 {key:'strong_rps', icon:'👑', name:'强势龙头', tag:'趋势强度', desc:'RPS≥0.90 + 多头排列',
  conds:{rps50:0.90, ma_align:3}, stop_pct:7,
  plain:'一句话：只买全市场最强的那批票。RPS≥0.9=涨幅跑赢90%的股票（欧奈尔"强者恒强"），再叠加均线多头=强且结构健康。强势股回调有人接、上涨有惯性，是趋势交易者的核心标的池。',
  detail:{
   title:'强势龙头 — 为什么这样选？',
   logic:`<b>核心思想：</b>只买全市场最强的前10%，强者恒强（欧奈尔体系）。<br><br>
    <b>条件1 · RPS≥0.90：</b>50日涨幅排名前10%的股票。回调有人接、上涨有惯性。<br><br>
    <b>条件2 · 均线多头≥3天：</b>强势且结构健康，排除"靠一波脉冲上榜"的票。<br><br>
    <b>适合：</b>趋势市主战场。 <b>不适合：</b>熊市（强势股补跌最狠）。`,
   exit:'卖出参考：RPS跌出前20%或触发止损价即离场。'}},
 {key:'hyper_trend', icon:'🚄', name:'趋势+强势+量确认', tag:'组合', desc:'多头排列 + RPS≥0.85 + 量比≥1.3',
  conds:{ma_align:3, rps50:0.85, vol_ratio:1.3}, stop_pct:7,
  plain:'一句话：三重保险买法——结构健康（多头排列）+ 够强（跑赢85%的票）+ 资金确认（放量）。三个角度互相验证，信号少但质量高，适合"不想天天盯盘、宁可少做几笔"的用户。',
  detail:{
   title:'趋势+强势+量确认 — 为什么这样选？',
   logic:`<b>核心思想：</b>三重保险——结构、强度、资金互相验证。<br><br>
    <b>多头排列：</b>结构健康，短中长线都没套牢盘。<br>
    <b>RPS≥0.85：</b>只做市场前15%强的票。<br>
    <b>量比≥1.3：</b>温和放量确认，排除无人问津的阴涨。<br><br>
    信号少但质量高，适合低频波段。<br><br>
    <b>适合：</b>不想盯盘、少而精的用户。 <b>不适合：</b>想天天有票可买的活跃用户。`,
   exit:'卖出参考：跌破20日线或触发止损价即离场。'}},
];
let activeTpls=[];   // 多选：已选中的风格 key 列表，默认空（启动时无任何选项）
let TPL_DETAIL_OPEN=null;
function renderTemplates(){
  $('tplBox').innerHTML = TEMPLATES.map(t=>{
    const on = activeTpls.includes(t.key);
    return `<button class="tplCard ${on?'tplOn':''}" onclick="useTemplate('${t.key}')"
      style="text-align:left;padding:9px 12px;border-radius:9px;border:1px solid ${on?'var(--accent)':'var(--border)'};
             background:${on?'#13203a':'#0b1120'};color:var(--text);cursor:pointer;font-size:12.5px;margin:0;width:100%">
      <div style="font-weight:700;font-size:13px">${t.icon} ${t.name}</div>
      <div style="color:var(--muted);font-size:11.5px;margin-top:2px">${t.desc}</div>
    </button>`;
  }).join('') +
  `<div style="font-size:10.5px;color:var(--muted);margin-top:4px">💡 可点击多张卡<b>组合</b>使用；再点一次已选中的卡即取消</div>`;
  updateTplCount();
}
// 指标互斥组：同一组内的条件同时勾选时相互矛盾
const IND_CONFLICTS = [
  {group:'位置类互斥', items:['near_high','pullback_stable','box_amp'],
   why:'接近新高(高位)、回调企稳(刚跌过)、横盘(没涨没跌)三个位置不可能同时成立'},
  {group:'波动类互斥', items:['vol_narrow','gain_today'],
   why:'波动收敛(平静)与当日大涨(剧烈)矛盾；大涨日ATR必然抬高'},
];
function detectConflicts(selKeys){
  // selKeys: 当前所有勾选的指标名集合
  const msgs=[];
  for(const c of IND_CONFLICTS){
    const hit = c.items.filter(k=>selKeys.has(k));
    if(hit.length>=2) msgs.push(`⚠️ ${c.group}：${hit.join(' + ')} 同时选了 —— ${c.why}`);
  }
  return msgs;
}
function useTemplate(key){
  const t=TEMPLATES.find(x=>x.key===key); if(!t) return;
  // 多选：已选中则取消
  if(activeTpls.includes(key)){
    activeTpls = activeTpls.filter(k=>k!==key);
  } else {
    activeTpls.push(key);
    // 首次选中时自动展开风格区
    const b=$('tplBody');
    if(b && b.style.display==='none'){ toggleTpl(); }
  }
  renderTemplates();
  rebuildCondsFromTpls();
}
// 按已选模板集合重建勾选条件（同名指标参数取第一个模板的值）
function rebuildCondsFromTpls(){
  const merged = {};
  for(const key of activeTpls){
    const t = TEMPLATES.find(x=>x.key===key); if(!t) continue;
    Object.assign(merged, t.conds);
  }
  // 清空再填
  document.querySelectorAll('.indChk').forEach(c=>{c.checked=false; c.dispatchEvent(new Event('change',{bubbles:true}));});
  for(const [k,v] of Object.entries(merged)){
    const chk=document.querySelector(`.indChk[data-ind="${k}"]`);
    if(!chk) continue;
    chk.checked=true; chk.dispatchEvent(new Event('change',{bubbles:true}));
    const inputs=[...document.querySelectorAll(`.indP[data-ind="${k}"]`)];
    if(inputs.length>1 && Array.isArray(v)) inputs.forEach((x,i)=>x.value=v[i]);
    else if(inputs.length) inputs[0].value=v;
  }
  const sel = new Set(Object.keys(merged));
  const conflicts = detectConflicts(sel);
  const allTpls = activeTpls.map(k=>TEMPLATES.find(x=>x.key===k));
  const stop = allTpls.length ? allTpls[allTpls.length-1].stop_pct : 7;
  $('stopPct').value=stop; $('stopPctTpl').value=stop;
  renderTplDetail(allTpls, merged, conflicts);
}
// 右侧主区显示该风格的详细解读（选中即出，点筛选才执行）
function renderTplDetail(tpls, mergedConds, conflicts){
  TPL_DETAIL_OPEN = tpls;
  const box = document.getElementById('tplDetailBox');
  if(!box) return;
  if(!tpls || !tpls.length){ closeTplDetail(); return; }
  const title = tpls.length===1
    ? `${tpls[0].icon} ${tpls[0].detail.title}`
    : `🧩 组合：${tpls.map(t=>t.name).join(' + ')}`;
  // 大白话：单选=该模板的 plain；组合=各模板一句话拼接 + 组合逻辑说明
  const plain = tpls.length===1
    ? `<div style="font-size:13.5px;line-height:1.8;color:var(--text);margin-bottom:10px">💬 ${tpls[0].plain}</div>`
    : `<div style="font-size:13.5px;line-height:1.8;color:var(--text);margin-bottom:10px">💬 ${tpls.map(t=>t.plain).join('<br><br>')}</div>
       <div style="font-size:12.5px;line-height:1.8;color:var(--muted);margin-bottom:10px;padding:10px 14px;background:#0b1120;border-left:3px solid var(--accent);border-radius:6px">
       🧩 <b style="color:var(--text)">组合逻辑</b>：这 ${tpls.length} 个风格的条件会<b>同时要求全部满足</b>（AND 关系）——票必须符合每一条才入选。条件越多→选出的票越少但质量越高；条件之间是互补关系（比如"横盘蓄势+趋势向上"=等蓄势票向上突破的那一刻）。</div>`;
  const condsDesc = Object.entries(mergedConds).map(([k,v])=>{
    const ind = INDS.find(x=>x.name===k);
    return `<span style="display:inline-block;background:#0b1120;border:1px solid var(--border);border-radius:7px;padding:3px 10px;margin:0 6px 6px 0;font-size:12px">${ind?ind.label:k} = ${v}</span>`;
  }).join('');
  const conflictHtml = conflicts && conflicts.length
    ? `<div style="margin:10px 0;padding:10px 14px;background:#2a1215;border-left:3px solid var(--red);border-radius:6px;font-size:12.5px;color:#fca5a5">
         ${conflicts.map(c=>c.replace(/\n/g,'<br>')).join('<br>')}<br>
         <span style="color:var(--muted)">这样的组合筛选结果会恒为空（或失真），请取消其中一组。</span></div>`
    : '';
  const exits = tpls.map(t=>t.detail.exit).join('；');
  const stop = tpls[tpls.length-1].stop_pct;
  box.innerHTML = `<div style="animation:none">
    <div style="background:linear-gradient(135deg,#13203a,#0f1a30);border:1px solid var(--accent);border-radius:12px;padding:16px 20px;margin-bottom:14px;position:relative">
      <span onclick="closeTplDetail()" title="关闭解读"
        style="position:absolute;top:10px;right:14px;cursor:pointer;color:var(--muted);font-size:16px">✕</span>
      <div style="font-size:16px;font-weight:800;margin-bottom:8px">${title}
        <span style="font-size:11.5px;color:var(--muted);font-weight:400;margin-left:10px">风格可选可不选，也可直接用下方自定义指标</span></div>
      ${conflictHtml}
      <div style="font-size:12.5px;color:var(--muted);margin-bottom:8px">已为你填好以下条件（尚未筛选）：</div>
      <div style="margin-bottom:10px">${condsDesc}</div>
      <button class="green" style="width:auto;padding:7px 22px;font-size:13px" onclick="doScreen()" ${conflicts.length?'disabled title="先解决冲突条件"':''}>🎯 看明白了，开始筛选</button>
      <span style="font-size:11.5px;color:var(--muted);margin-left:10px">组合回测在左侧「② 自定义指标」展开区内，用于验证胜率</span>
      <details style="margin-top:10px">
        <summary style="cursor:pointer;font-size:12.5px;color:var(--accent);user-select:none">📖 展开/收起 详细解读</summary>
        <div style="padding:12px 4px 0;line-height:1.9;font-size:13px;color:var(--text)">
          ${tpls.map(t=>t.detail.logic).join('<hr style="border:none;border-top:1px solid var(--border);margin:12px 0">')}
          <div style="margin-top:12px;padding:10px 14px;background:#0b1120;border-left:3px solid var(--red);border-radius:6px;font-size:12.5px;color:var(--muted)">
            🛡 <b style="color:var(--text)">风控</b> · ${exits}（当前止损 ${stop}%，可在左侧调整）</div>
        </div>
      </details>
    </div>
  </div>`;
  box.style.display='';
}
function closeTplDetail(){
  TPL_DETAIL_OPEN = null;
  const box = document.getElementById('tplDetailBox');
  if(box){ box.style.display='none'; box.innerHTML=''; }
}
function toggleTpl(){
  const b=$('tplBody'), a=$('tplArrow');
  const open=b.style.display==='none';
  b.style.display=open?'':'none';
  a.textContent=open?'▴ 收起':'▾ 展开';
}
function updateTplCount(){
  const el=$('tplCount'); if(!el) return;
  el.textContent = activeTpls.length ? `（已选${activeTpls.length}个）` : '';
}
function toggleAdv(){
  const b=$('advBody'), a=$('advArrow');
  const open=b.style.display==='none';
  b.style.display=open?'':'none';
  a.textContent=open?'▴ 收起':'▾ 展开';
}

// ================= 行业分布条形图（可视化，可点击筛选） =================
let indFilter = null;   // 当前点击选中的行业
function industryBars(summary){
  const items=(summary||[]).slice(0,30);
  if(!items.length) return '';
  const max=Math.max(...items.map(x=>x.count));
  const row=x=>`
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:3px;cursor:pointer" onclick="filterByIndustry('${x.industry}')" title="点击只看${x.industry}">
      <span style="width:86px;text-align:right;${indFilter===x.industry?'color:var(--accent);font-weight:700':'color:var(--muted)'};overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${x.industry}</span>
      <div style="flex:1;height:13px;background:#0b1120;border-radius:4px;overflow:hidden">
        <div style="height:100%;width:${Math.round(x.count/max*100)}%;background:${indFilter===x.industry?'var(--green)':'linear-gradient(90deg,var(--accent),#60a5fa)'};border-radius:4px"></div>
      </div>
      <b style="width:28px;color:var(--text)">${x.count}</b>
    </div>`;
  const half=Math.ceil(items.length/2);
  const col=a=>a.map(row).join('');
  return `<div style="font-size:11.5px;margin:0 4px 10px">
    <div style="color:var(--muted);margin-bottom:5px">🏭 行业分布（共${items.length}类 · <span style="color:var(--accent)">点击行业过滤列表</span>${indFilter?` · 当前：<b style="color:var(--accent)">${indFilter}</b> <a href="javascript:void(0)" onclick="indFilter=null;renderScreenResult()" style="color:var(--red)">✕ 取消</a>`:''}）</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0 18px">
      <div>${col(items.slice(0,half))}</div><div>${col(items.slice(half))}</div>
    </div>
  </div>`;
}
function filterByIndustry(ind){
  indFilter = (indFilter===ind) ? null : ind;
  renderScreenResult();
}

// ================= 自定义筛选（勾选式表单） =================
let INDS = [], SCREEN_RESULT = null, PARAM_STATS = null;
const IND_LABEL = {};

// 勾选联动：选中才允许改该指标参数，展开回测明细 + 右侧显示指标解读
function bindIndRowEvents(){
  const form = document.getElementById('indForm');
  form.addEventListener('change', e => {
    if(!e.target.classList.contains('indChk')) return;
    const name = e.target.dataset.ind;
    const row = form.querySelector(`.indRow[data-name="${name}"]`);
    if(!row) return;
    row.querySelectorAll('.indP').forEach(x => x.disabled = !e.target.checked);
    row.style.background = e.target.checked ? '#13203a' : '';
    updateBtDetail(row, name, e.target.checked);
    const tip = row.querySelector('.rngTip');
    if(tip) tip.style.display = e.target.checked ? '' : 'none';
    renderIndPreview();
  });
  // 参数越界实时标红
  form.addEventListener('input', e => {
    if(!e.target.classList.contains('indP')) return;
    const name = e.target.dataset.ind;
    const ind = INDS.find(x=>x.name===name);
    if(!ind || !ind.valid_range) return;
    const v = parseFloat(e.target.value);
    const [lo,hi] = ind.valid_range;
    const bad = isNaN(v) || v < lo || v > hi;
    e.target.style.borderColor = bad ? 'var(--red)' : '';
    e.target.title = bad ? `超出有效区间 [${lo}~${hi}]，该档位指标失灵` : '';
  });
}

function renderIndPreview(){
  // 右侧顶部预览：当前勾选指标的说明（含调参影响）
  const boxId = 'indPreview';
  let box = document.getElementById(boxId);
  const list = document.getElementById('list');
  const checked = [...document.querySelectorAll('.indChk:checked')].map(c=>c.dataset.ind);
  if(!checked.length){
    box && box.remove();
    return;
  }
  const html = `<div id="${boxId}" style="margin-bottom:12px">` +
    checked.map(n => {
      const ind = INDS.find(x=>x.name===n); if(!ind) return '';
      const risks = (ind.risks||[]).map(r=>`<li>${r}</li>`).join('');
      const watch = (ind.watch_points||[]).map(w=>`<li>${w}</li>`).join('');
      const invalid = (ind.invalidation||[]).map(i=>`<li>${i}</li>`).join('');
      return `<div class="card" style="border-left:3px solid var(--accent)">
        <div style="font-weight:700;font-size:13px;margin-bottom:4px">${ind.label}
          <span style="color:var(--muted);font-weight:400;font-size:11.5px">(默认 ${Array.isArray(ind.default)?ind.default.join('~'):ind.default})</span></div>
        <div style="font-size:12.5px;line-height:1.65;color:var(--muted)">${ind.desc||''}</div>
        ${risks?`<div style="margin-top:6px"><div style="font-size:11.5px;font-weight:600;color:#fca5a5">⚠ 风险</div><ul style="margin:2px 0 0 16px;font-size:11.5px;color:var(--muted)">${risks}</ul></div>`:''}
        ${watch?`<div style="margin-top:4px"><div style="font-size:11.5px;font-weight:600;color:#fbbf24">👁 观察</div><ul style="margin:2px 0 0 16px;font-size:11.5px;color:var(--muted)">${watch}</ul></div>`:''}
        ${invalid?`<div style="margin-top:4px"><div style="font-size:11.5px;font-weight:600;color:#22c55e">✖ 失效</div><ul style="margin:2px 0 0 16px;font-size:11.5px;color:var(--muted)">${invalid}</ul></div>`:''}
      </div>`;
    }).join('') + '</div>';
  if(box){ box.outerHTML = html; }
  else { list.insertAdjacentHTML('afterbegin', html); }
}

function updateBtDetail(row, name, show){
  const el = row.querySelector('.btDetail');
  if(!show || !PARAM_STATS){ el.style.display='none'; return; }
  const g = (PARAM_STATS.stats||{})[name];
  if(!g){ el.style.display='none'; return; }
  const rows = Object.entries(g).map(([p,r])=>
    `<b>${p}</b>: 胜率5/10/20日 ${r.win5??'-'}/${r.win10??'-'}/${r.win20??'-'}% · 20日均收益 ${r.avg20??'-'}% · 样本${r.n_signals}`
  ).join('<br>');
  el.innerHTML = `↯ 参数回测（全市场${PARAM_STATS.stocks}只，截至${PARAM_STATS.generated}）：<br>${rows}`;
  el.style.display = '';
}

async function saveStrategy(){
  const conds = collectConds();
  if (!Object.keys(conds).length){ alert('请先勾选至少一个指标'); return; }
  const name = prompt('给这组策略起个名字：', '我的策略'+new Date().toISOString().slice(5,10));
  if(!name) return;
  const r = await fetch('/api/strategies',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name, conditions:conds,
      stop_pct: parseFloat(document.getElementById('stopPct').value)||7})});
  if(r.ok){ loadStrategies(); } else { const e=await r.json(); alert(e.error||'保存失败'); }
}

async function delStrategy(name){
  if(!confirm('删除策略「'+name+'」？')) return;
  await fetch('/api/strategies/'+encodeURIComponent(name),{method:'DELETE'});
  loadStrategies();
}

async function applyStrategy(name){
  const s = STRATEGIES.find(x=>x.name===name); if(!s) return;
  // 先全不选
  document.querySelectorAll('.indChk').forEach(c=>{c.checked=false; c.dispatchEvent(new Event('change',{bubbles:true}));});
  for(const [k,v] of Object.entries(s.conditions)){
    const chk = document.querySelector(`.indChk[data-ind="${k}"]`);
    if(!chk) continue;
    chk.checked = true; chk.dispatchEvent(new Event('change',{bubbles:true}));
    const inputs=[...document.querySelectorAll(`.indP[data-ind="${k}"]`)];
    if(inputs.length>1 && Array.isArray(v)) inputs.forEach((x,i)=>x.value=v[i]);
    else if(inputs.length) inputs[0].value=v;
  }
  document.getElementById('stopPct').value = s.stop_pct||7;
}

let STRATEGIES=[];
async function loadStrategies(){
  try{
    const j=await (await fetch('/api/strategies')).json();
    STRATEGIES=j.strategies||[];
  }catch(e){ STRATEGIES=[]; }
  const el=document.getElementById('strategyList');
  el.innerHTML = STRATEGIES.map(s=>`
    <div class="row" style="margin-bottom:4px">
      <a href="javascript:void(0)" onclick="applyStrategy('${s.name.replace(/'/g,"\\'")}')" 
         style="color:var(--accent);flex:1;font-size:12px">▸ ${s.name}</a>
      <span onclick="delStrategy('${s.name.replace(/'/g,"\\\'")}')" 
         style="cursor:pointer;color:var(--red);font-size:11px">✕</span>
    </div>`).join('') ||
    '<div style="color:var(--muted);font-size:11.5px">暂无保存的策略</div>';
}

function collectConds(){
  const conds = {};
  document.querySelectorAll('.indChk:checked').forEach(chk => {
    const name = chk.dataset.ind;
    const inputs = [...document.querySelectorAll(`.indP[data-ind="${name}"]`)];
    if (!inputs.length) return;
    conds[name] = inputs.length > 1
      ? inputs.map(x => parseFloat(x.value))
      : parseFloat(inputs[0].value);
  });
  return conds;
}

// 一键清空：取消全部风格卡 + 勾选指标，参数恢复默认，并清空结果列表
function clearAllConds(){
  // 1. 清空已选风格（activeTpls）并重绘模板卡
  activeTpls = [];
  renderTemplates();
  closeTplDetail();
  // 2. 全部指标复选框取消勾选 + 参数恢复默认值
  document.querySelectorAll('.indChk').forEach(c=>{
    c.checked = false;
    c.dispatchEvent(new Event('change',{bubbles:true}));   // 触发禁用参数/清预览联动
  });
  const byName = {};
  INDS.forEach(i=>{ byName[i.name]=i.default; });
  document.querySelectorAll('.indP').forEach(x=>{
    const def = byName[x.dataset.ind];
    if(def===undefined) return;
    x.value = Array.isArray(def) ? def[x.dataset.idx|0] : def;
  });
  // 3. 清空结果列表与预览
  SCREEN_RESULT = null;
  indFilter = null;
  const listEl = document.getElementById('list');
  if(listEl) listEl.innerHTML =
    '<div style="color:var(--muted);text-align:center;padding:40px">已清空。请选择风格或勾选指标后重新筛选。</div>';
}

window.addEventListener('error', e => {
  const el = document.getElementById('indForm');
  if (el && !el.innerHTML) el.innerHTML = `<span style="color:var(--red)">页面脚本错误: ${e.message}</span>`;
});

async function loadIndicators(){
  try{
    const j = await (await fetch('/api/indicators')).json();
    INDS = j.indicators || [];
  }catch(e){ INDS = []; }
  // 参数回测成绩
  try{
    const ps = await (await fetch('/api/param_stats')).json();
    PARAM_STATS = ps.ready ? ps : null;
  }catch(e){ PARAM_STATS = null; }
  const byCat = {};
  INDS.forEach(i => { IND_LABEL[i.name]=i.label;
                     (byCat[i.category] = byCat[i.category] || []).push(i); });
  let html = '';
  for (const [cat, list] of Object.entries(byCat)){
    html += `<div style="color:var(--accent);font-size:11px;margin:8px 0 4px;font-weight:700">${cat}</div>`;
    for (const ind of list){
      html += indRowHtml(ind);
    }
  }
  document.getElementById('indForm').innerHTML =
    html || '<span style="color:var(--muted)">指标加载失败</span>';
}

function btBadge(name, param){
  // 参数历史回测徽章：20日胜率/样本数 + 样本外验证
  if(!PARAM_STATS) return '';
  const g = (PARAM_STATS.stats||{})[name];
  if(!g) return '';
  const keys = Object.keys(g).map(Number)
    .sort((a,b)=>Math.abs(a-param)-Math.abs(b-param));
  const rec = keys.length ? g[String(keys[0])] : null;
  if(!rec || rec.win20==null) return '';
  const col = rec.win20>=55?'var(--green)':rec.win20>=45?'var(--amber)':'var(--red)';
  let oos = '';
  if(rec.oos_win20!=null){
    const oosCol = rec.oos_decay ? 'var(--red)' : 'var(--green)';
    oos = ` <span style="color:${oosCol}" title="样本外(最近${PARAM_STATS.oos_days}交易日,${rec.oos_n}信号)：${rec.oos_decay?'衰减警告！样本内外差>8pp':'样本外验证通过'}">⏳${rec.oos_win20}%</span>`;
  }
  return `<span title="参数${keys[0]}的历史回测(全市场${PARAM_STATS.stocks}只,${PARAM_STATS.generated},含交易成本)" 
    style="color:${col};font-size:10.5px;font-weight:600">↯${rec.win20}%<span style="color:var(--muted)">(${rec.n_signals})</span>${oos}</span>`;
}

async function doBacktestCombo(){
  const conds = collectConds();
  if(!Object.keys(conds).length){ alert('请至少勾选一个指标'); return; }
  const el = document.getElementById('comboResult');
  el.innerHTML = '组合回测中…';
  const r = await fetch('/api/backtest_combo',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({conditions: conds})});
  const j = await r.json();
  if(!r.ok){ el.innerHTML = '<span style="color:var(--red)">'+(j.error||'失败')+'</span>'; return; }
  if(!j.n_signals){ el.innerHTML = '历史信号不足，无法回测'; return; }
  let s = `🧪 <b>${j.n_signals}</b> 个历史信号 · 组合20日胜率 <b style="color:${j.win20>=50?'var(--green)':'var(--red)'}">${j.win20}%</b> · 平均 <b>${j.avg20}%</b> (含成本)`;
  if(j.oos_win20!=null){
    s += ` · 样本内 ${j.is_win20}% → 样本外 <b style="color:${j.oos_win20>=j.is_win20-8?'var(--green)':'var(--red)'}">${j.oos_win20}%</b>`;
  }
  el.innerHTML = s;
}

function indRowHtml(ind){
  const arr = Array.isArray(ind.default);
  const d0 = arr ? ind.default[0] : ind.default;
  const d1 = arr ? ind.default[1] : '';
  const vr = ind.valid_range;           // [lo,hi] 或 null
  const rmin = vr ? vr[0] : null, rmax = vr ? vr[1] : null;
  const rngAttr = (el) => vr ? ` min="${rmin}" max="${rmax}"` : '';
  const inputs = arr
    ? `<input type="number" class="indP" data-ind="${ind.name}" data-idx="0" value="${d0}" style="width:50px" disabled${vr?' min="'+rmin+'" max="'+rmax+'"':''}>
       <span class="indSep" style="color:var(--muted)">-</span>
       <input type="number" class="indP" data-ind="${ind.name}" data-idx="1" value="${d1}" style="width:50px" disabled>`
    : `<input type="number" class="indP" data-ind="${ind.name}" value="${d0}" style="width:62px" step="any" disabled${vr?' min="'+rmin+'" max="'+rmax+'"':''}>`;
  return `<div class="indRow" data-name="${ind.name}"
            style="border-bottom:1px solid #ffffff08;padding:5px 2px;cursor:pointer">
    <div class="row" style="margin-bottom:0">
      <label style="flex:1;display:flex;align-items:center;gap:6px" title="${(ind.desc||'').replace(/"/g,'&quot;')}">
        <input type="checkbox" class="indChk" data-ind="${ind.name}">
        <span>${ind.label}</span>
      </label>
      <span class="btWrap">${btBadge(ind.name, d0)}</span>
      ${inputs}
    </div>
    ${vr?`<div class="rngTip" style="display:none;font-size:10.5px;color:var(--muted);padding:2px 24px">有效区间 [${rmin} ~ ${rmax}]，超出视为指标失灵</div>`:''}
    <div class="btDetail" style="display:none;font-size:10.5px;color:var(--muted);padding:3px 24px"></div>
  </div>`;
}

function getStopPct(){
  const a=$('stopPctTpl'), b=$('stopPct');
  const src = (a && document.getElementById('secTpl').offsetParent) ? a : b; // 可见面板优先
  const v = parseFloat((src&&src.value)) || 7;
  if(a) a.value=v; if(b) b.value=v;   // 双向同步
  return v;
}

async function doScreen(){
  const conds = collectConds();
  if (!Object.keys(conds).length){ alert('请先选一个风格或勾选指标'); return; }
  const listEl = document.getElementById('list');
  const prog = `<div id="screenProg" style="padding:30px;text-align:center">
    <div style="color:var(--muted);margin-bottom:12px" id="screenProgText">筛选中，全市场计算指标…</div>
    <div style="width:240px;height:8px;background:#1e293b;border-radius:4px;margin:0 auto;overflow:hidden">
      <div id="screenProgBar" style="height:100%;width:0%;background:var(--accent);border-radius:4px;transition:width 0.3s"></div>
    </div>
    <div id="screenProgPct" style="font-size:11px;margin-top:6px;color:var(--muted)"></div>
  </div>`;
  listEl.innerHTML = prog;
  try {
    const r = await fetch('/api/screen', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({conditions: conds, stop_pct: getStopPct()})});
    if(r.status === 409){ alert('已有筛选任务在运行中'); return; }
    if(!r.ok){ const e=await r.json().catch(()=>({})); listEl.innerHTML=`<div style="color:var(--red);padding:20px">${e.error||'筛选失败'}</div>`; return; }
  } catch(e){
    listEl.innerHTML=`<div style="color:var(--red);padding:20px">筛选启动失败：${e.message}</div>`;
    return;
  }
  // Poll until done
  await pollScreenProgress();
}
function pollScreenProgress(){
  return new Promise((resolve, reject) => {
    const poll = setInterval(async () => {
      try {
        const s = await (await fetch('/api/screen_status')).json();
        if(s.error){ clearInterval(poll); reject(new Error(s.error)); return; }
        const p = s.progress;
        if(p){
          const bar = document.getElementById('screenProgBar');
          const txt = document.getElementById('screenProgText');
          const pct = document.getElementById('screenProgPct');
          if(bar) bar.style.width = p.pct + '%';
          if(txt) txt.textContent = p.phase;
          if(pct) pct.textContent = p.pct + '%';
        }
        if(s.done){
          clearInterval(poll);
          const r = await fetch('/api/screen_result');
          if(!r.ok){ reject(new Error('获取结果失败')); return; }
          SCREEN_RESULT = await r.json();
          renderScreenResult();
          resolve();
        }
      } catch(e){ clearInterval(poll); reject(e); }
    }, 500);
  });
}

// ================= 内嵌个股详情（不跳新页，返回即回到筛选结果） =================
let DETAIL_STATE = null;   // {symbol, data, range:'60'|'120'|'250'|'all'}
function openDetail(symbol, fromTab){
  const conds = collectConds();
  const stopQ = getStopPct();
  DETAIL_STATE = {symbol, data:null, fromTab: fromTab||null};
  if(fromTab) showTab('filter');          // 从其他页签进来 → 切到筛选页查看板
  const listEl = document.getElementById('list');
  listEl.innerHTML = '<div style="color:var(--muted);padding:30px;text-align:center">加载 '+symbol+' 详情…</div>';
  fetch('/api/screen_detail/'+symbol+'?conds='+encodeURIComponent(JSON.stringify(conds))+'&stop_pct='+stopQ)
    .then(r=>{ if(!r.ok) throw new Error('无K线数据'); return r.json(); })
    .then(j=>{ DETAIL_STATE.data=j; renderDetail(symbol, j); })
    .catch(()=>{ listEl.innerHTML='<div style="color:var(--red);padding:20px">'+symbol+' 无K线数据</div>'; });
}
function closeDetail(){
  const from = DETAIL_STATE && DETAIL_STATE.fromTab;
  if(from){ showTab(from); DETAIL_STATE.fromTab=null; }   // 返回来源页签（如历史回放）
  if(SCREEN_RESULT){ renderScreenResult(); }
  else if(!from){ document.getElementById('list').innerHTML='<div style="color:var(--muted);text-align:center;padding:40px">请先选择风格或条件并开始筛选</div>'; }
}
function renderDetail(sym, j){
  const fu = j.fundamental||{};
  const t = j.trade||{};
  const kl = j.key_levels||{};
  const checks=[
    {k:'净利润同比',v:fu.profit_yoy,ok:v=>v>0,need:'>0'},
    {k:'营收同比',v:fu.revenue_yoy,ok:v=>v>0,need:'>0'},
    {k:'ROE',v:fu.roe,ok:v=>v>=10,need:'≥10%'},
    {k:'毛利率',v:fu.gross_margin,ok:v=>v>=20,need:'≥20%'},
    {k:'资产负债率',v:fu.debt_ratio,ok:v=>v<=70,need:'≤70%'},
  ];
  const known=checks.filter(c=>c.v!=null);
  const pass=known.filter(c=>c.ok(c.v)).length;
  const grade=known.length===0?'未知':(pass>=4?'优秀':pass>=3?'良好':'偏弱');
  const html = `
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;flex-wrap:wrap">
    <button onclick="closeDetail()" style="width:auto;padding:5px 14px;margin:0;background:#33415580">← 返回候选列表</button>
    <span style="font-size:19px;font-weight:800">${sym} ${j._name||''}</span>
    <span class="health ${pass>=4?'ok':pass>=3?'':'no'}" style="font-size:13px">基本面体检：${grade} (${pass}/${known.length||5})</span>
  </div>
  <div id="detailBody">
    <div class="card" style="margin-bottom:12px">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap">
        <h4 style="margin:0">日K线 — 触发信号与失效参考</h4>
        <span style="margin-left:auto;display:flex;gap:6px">
          ${['60日','半年','一年','全部'].map((lb,i)=>`<button onclick="setRange('${sym}','${['60','120','250','all'][i]}')"
            style="width:auto;padding:3px 12px;margin:0;font-size:11.5px;border-radius:6px;background:${['60','120','250','all'][i]==='250'?'var(--accent)':'#33415580'}" class="rngBtn" data-r="${['60','120','250','all'][i]}">${['60日','半年','一年','全部'][i]}</button>`).join('')}
        </span>
      </div>
      <div id="kchartWrap" style="width:100%;overflow:hidden"><div id="kchartInline"></div></div>
      <div class="legend" style="font-size:11.5px;color:var(--muted);margin-top:8px;line-height:1.7">
        <span style="color:#22c55e">▲触发信号(买)</span> 信号日收盘确认，次日开盘进场 &nbsp;
        <span style="color:#ef4444">✖失效参考(卖)</span> 收盘跌破红横线=买点逻辑失效，无条件离场 &nbsp;
        <span style="color:#8b5cf6">┄ 紫虚线</span> MA20生命线 &nbsp;
        <span style="color:var(--muted)">💡 每一次历史信号都标注了▲买卖提示</span>
      </div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 330px;gap:12px">
      <div>
        <div class="card"><h4>📈 这只票的历史信号回放（当前条件 · 最多近8次）</h4><table id="histtab"></table></div>
        <div class="card"><h4>✅ 当前筛选条件满足情况</h4><table id="sigtab2"></table></div>
      </div>
      <div>
        <div class="card">
          <h4>🩺 基本面体检</h4><div class="health" id="health2"></div><table id="fmtab2"></table>
        </div>
        <div class="card">
          <h4>💰 买卖点与止损</h4>
          <div class="buybox">
            <div class="bk"><div class="t">买点(次日开盘)</div><div class="v v-buy" id="bp2">-</div></div>
            <div class="bk"><div class="t">止损位</div><div class="v v-stop" id="sp2">-</div></div>
            <div class="bk"><div class="t">关键位(监控)</div><div class="v" id="kl2">-</div></div>
          </div>
        </div>
      </div>
    </div>
  </div>`;
  document.getElementById('list').innerHTML = html;
  renderDetailBody(j);
}
function setRange(sym, range){
  if(!DETAIL_STATE||!DETAIL_STATE.data) return;
  drawChart(DETAIL_STATE.data, range);
}
// K线渲染：直接按区间渲染（不用plotly内置缩放），买卖标注完整保留
function drawChart(j, range){
  if(!j||!j.plot) return;
  DETAIL_STATE.range=range;
  const plot=j.plot;
  const dates=plot.data[0].x;
  const layout=JSON.parse(JSON.stringify(plot.layout));
  layout.xaxis.rangeselector={visible:false};
  // 缩放/拖动范围约束：x 只能在数据日期内，y 只能在价格范围内 —— 不会"缩没了"
  const x0=dates[0], x1=dates[dates.length-1];
  const lows=plot.data[0].low, highs=plot.data[0].high;
  const yLo=Math.min(...lows)*0.98, yHi=Math.max(...highs)*1.02;
  layout.xaxis.fixedrange=false;
  layout.yaxis.fixedrange=false;
  layout.dragmode='pan';                 // 拖动=平移，比 zoom 框选更直观
  layout.xaxis.rangeslider={visible:false};
  layout.hovermode='x unified';
  // 统一宽度：跟随容器，60日与其他周期行为完全一致
  const holder=document.getElementById('kchartWrap')||document.getElementById('kchartInline');
  layout.width=Math.max(600, (holder?holder.clientWidth:900));
  layout.height=460;
  layout.margin={l:45,r:15,t:10,b:35};
  // 默认定位：最近交易日收尾
  let iFrom=0;
  if(range && range!=='all'){
    const n={'60':60,'120':125,'250':250}[range]||250;
    iFrom=Math.max(0, dates.length-n);
    layout.xaxis.range=[dates[iFrom], x1];
  } else { layout.xaxis.range=[x0, x1]; }
  // y 轴按当前窗口内的实际价格区间自适应（缩放/切换后波动形变可见，不再"平滑"）
  const wLo=Math.min(...lows.slice(iFrom))*0.995, wHi=Math.max(...highs.slice(iFrom))*1.005;
  layout.yaxis.range=[wLo, wHi];
  Plotly.react('kchartInline', plot.data, layout, {responsive:true, displayModeBar:true,
    scrollZoom:true,                       // 滚轮缩放
    modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d','hoverClosestGl2d']});
  document.querySelectorAll('.rngBtn').forEach(b=>{
    b.style.background = b.dataset.r===range ? 'var(--accent)' : '#33415580';});
  // 缩放/平移后把 x 轴范围夹回数据边界，防止拖出画面
  const gd=document.getElementById('kchartInline');
  gd.removeAllListeners('plotly_relayout');
  gd.on('plotly_relayout', function(ev){
    const xr=ev['xaxis.range'];
    if(!xr) return;
    let [a,b]=xr;
    const iA=nearestDateIndex(dates,a), iB=nearestDateIndex(dates,b);
    if(iA<=0 && iB>=dates.length-1) return;   // 已在全景内
    const na=dates[Math.max(0,iA)], nb=dates[Math.min(dates.length-1,iB)];
    if(na!==a || nb!==b){
      Plotly.relayout(gd, {'xaxis.range':[na,nb]});
    }
    // y 轴跟随当前 x 窗口自适应
    const lo=Math.min(...lows.slice(iA, iB+1)), hi=Math.max(...highs.slice(iA, iB+1));
    if(isFinite(lo)&&isFinite(hi)&&hi>lo){
      Plotly.relayout(gd, {'yaxis.range':[lo*0.995, hi*1.005]});
    }
  });
}
function nearestDateIndex(dates, d){
  // 二分找最近的交易日索引
  if(d<=dates[0]) return 0;
  if(d>=dates[dates.length-1]) return dates.length-1;
  let lo=0, hi=dates.length-1;
  while(lo<hi){
    const mid=(lo+hi)>>1;
    if(dates[mid]<d) lo=mid+1; else hi=mid;
  }
  return lo;
}

function renderDetailBody(j){
  const t=j.trade||{}, kl=j.key_levels||{}, fu=j.fundamental||{};
  document.getElementById('bp2').textContent=t.entry_price??'-';
  document.getElementById('sp2').textContent=t.stop_price??'-';
  const klItems=[
    kl.ma20!=null&&['MA20均线',kl.ma20,'跌破则趋势转弱'],
    kl.stop_price!=null&&['止损位',kl.stop_price,'收盘跌破无条件卖出'],
    kl.entry_price!=null&&['买点(参考)',kl.entry_price,'次日开盘附近挂单'],
  ].filter(Boolean);
  document.getElementById('kl2').innerHTML = klItems.length
    ? klItems.map(x=>`<div style="font-size:10.5px;color:var(--muted);font-weight:400" title="${x[2]}">${x[0]}</div><div style="font-size:13.5px">${x[1]}</div>`).join('')
    : '-';
  const checks=[
    {k:'净利润同比',v:fu.profit_yoy,ok:v=>v>0,need:'>0'},
    {k:'营收同比',v:fu.revenue_yoy,ok:v=>v>0,need:'>0'},
    {k:'ROE',v:fu.roe,ok:v=>v>=10,need:'≥10%'},
    {k:'毛利率',v:fu.gross_margin,ok:v=>v>=20,need:'≥20%'},
    {k:'资产负债率',v:fu.debt_ratio,ok:v=>v<=70,need:'≤70%'},
  ];
  const pct=v=>v==null?'-':(v>0?'+':'')+v.toFixed(2)+'%';
  document.getElementById('fmtab2').innerHTML=checks.map(c=>{
    const p=c.v!=null&&c.ok(c.v);
    const cls=c.v==null?'na':(p?'ok':'no');
    return `<tr><td class="lbl"><span class="${cls}">●</span> ${c.k}</td>
      <td class="val ${cls}">${pct(c.v)}</td><td class="val na" style="font-weight:400;font-size:11px">${c.need}</td></tr>`;
  }).join('');
  document.getElementById('sigtab2').innerHTML=(j.cond_status||[]).map(c=>{
    const cls=c.ok?'ok':'no';
    const risks=(c.risks||[]).map(r=>`<li>${r}</li>`).join('');
    const watch=(c.watch_points||[]).map(w=>`<li>${w}</li>`).join('');
    const invalid=(c.invalidation||[]).map(i=>`<li>${i}</li>`).join('');
    return `<tr><td class="lbl"><span class="${cls}">${c.ok?'✓':'✗'}</span> ${c.label}</td>
      <td class="val ${cls}">${c.ok?'满足':'未满足'}</td>
      <td style="padding:0 8px;vertical-align:top;max-width:320px">
        ${risks?`<div style="font-size:10.5px;color:#fca5a5">⚠ 风险</div><ul style="margin:1px 0 0 14px;font-size:10.5px;color:var(--muted)">${risks}</ul>`:''}
        ${watch?`<div style="font-size:10.5px;color:#fbbf24">👁 观察</div><ul style="margin:1px 0 0 14px;font-size:10.5px;color:var(--muted)">${watch}</ul>`:''}
        ${invalid?`<div style="font-size:10.5px;color:#22c55e">✖ 失效</div><ul style="margin:1px 0 0 14px;font-size:10.5px;color:var(--muted)">${invalid}</ul>`:''}
      </td></tr>`;
  }).join('')||'<tr><td class="lbl">无条件信息</td></tr>';
  const hist=j.history||[];
  const hEl=document.getElementById('histtab');
  if(hist.length){
    const wins=hist.filter(h=>h.fwd20>0).length;
    hEl.innerHTML=`<tr><td class="lbl" colspan="3" style="font-size:11.5px;line-height:1.7">历史 ${hist.length} 次触发 ·
      20日持有胜率 <b style="color:${wins/hist.length>=0.5?'var(--green)':'var(--red)'}">${Math.round(wins/hist.length*100)}%</b><br>
      <span style="color:var(--muted)">口径：信号日收盘确认 → <b>次日开盘价买入</b>（"买入价"即次日开盘价，非信号日价格）→ 持有20个交易日后按收盘价卖出，已扣双边手续费约0.7%。固定窗口用于衡量买点质量，不代表实际卖点。</span></td></tr>
    <tr style="color:var(--muted);font-size:11px"><td class="lbl">信号日</td><td class="val">次日开盘买入价</td><td class="val">持有20日收益</td></tr>`+
      hist.slice().reverse().map(h=>`
        <tr><td class="lbl">${h.date}</td><td class="val">${h.entry}</td>
        <td class="val" style="color:${h.fwd20>=0?'var(--green)':'var(--red)'}">${h.fwd20>0?'+':''}${h.fwd20}%</td></tr>`).join('');
  } else {
    hEl.innerHTML='<tr><td class="lbl" style="color:var(--muted)">当前条件下该股历史无信号</td></tr>';
  }
  drawChart(j, '250');
}

function renderScreenResult(){
  const R = SCREEN_RESULT; if(!R) return;
  const label = n => IND_LABEL[n] || PAT_CN[n] || n;
  const stopQ = parseFloat(document.getElementById('stopPct').value)||7;
  const card = r => `
    <div class="cand" onclick="openDetail('${r.symbol}')">
      <div class="r1">
        <span class="code">${r.symbol}</span><span class="nm" title="${r._name||''}">${r._name||''}</span>
        <span class="badge b-${r.status==='triggered'?'good':'neutral'}">${r.status==='triggered'?'已触发':'接近 '+r.score+'%'}</span>
      </div>
      <div class="r2" style="flex-wrap:wrap">
        ${r.entry_price?`<span>买点 <b>${r.entry_price}</b></span>
        <span>止损 <b style="color:var(--red)">${r.stop_price}</b></span>`:''}
        ${r.missing&&r.missing.length?`<span style="color:var(--amber)">缺: ${(r.missing).map(label).join('、')}</span>`:''}
      </div>
    </div>`;
  const nTrig=(R.triggered||[]).length, nNear=(R.near||[]).length;
  let html = `<div style="background:linear-gradient(135deg,#13203a,#0f1a30);border:1px solid var(--border);border-radius:12px;padding:14px 18px;margin-bottom:12px;display:flex;align-items:center;gap:22px;flex-wrap:wrap">
    <div><div style="font-size:11.5px;color:var(--muted)">今日已触发买点</div>
      <div style="font-size:30px;font-weight:800;color:var(--green);line-height:1.15">${nTrig}<span style="font-size:13px;color:var(--muted);font-weight:400"> 只</span></div></div>
    <div><div style="font-size:11.5px;color:var(--muted)">接近触发</div>
      <div style="font-size:30px;font-weight:800;color:var(--amber);line-height:1.15">${nNear}<span style="font-size:13px;color:var(--muted);font-weight:400"> 只</span></div></div>
    <div style="font-size:11.5px;color:var(--muted)">扫描 ${R.scanned} 只 · 买点=次日开盘 · 止损=买点×(1-止损%)只作风控</div>
    ${(R.upgraded&&R.upgraded.length)?`<div style="color:var(--amber);font-weight:700;font-size:12px;flex-basis:100%">🔔 升级提醒：${R.upgraded.map(u=>u.symbol+(u.name?' '+u.name:'')).join('、')} 已从"接近"转为"已触发"！（上次筛选 ${R.upgraded_from||''}）</div>`:''}
  </div>
  <div style="color:var(--muted);font-size:12px;padding:0 4px 10px;display:flex;gap:14px;align-items:center;flex-wrap:wrap">
    <button onclick="exportCsv()" style="width:auto;padding:4px 12px;margin:0;font-size:11.5px;background:#33415580">⬇ 导出CSV</button>
    <button onclick="aiInterpret()" id="aiBtn" style="width:auto;padding:4px 12px;margin:0;font-size:11.5px;background:#33415580">🤖 AI解读</button>
    <span style="margin-left:auto">排序：
      <select id="sortSel" onchange="resort()" aria-label="排序方式" style="padding:3px 7px;border-radius:6px;background:#0b1120;color:var(--text);border:1px solid var(--border);font-size:11.5px">
        <option value="score">接近程度 高→低（仅接近组有效）</option>
        <option value="entry">买点价 从低到高</option>
      </select></span>
    </div>`;
  if(R.industry_summary && R.industry_summary.length){
    html += industryBars(R.industry_summary);
  }
  const trigList = (R.triggered||[]).filter(r=>!indFilter || r.industry===indFilter);
  const nearList = (R.near||[]).filter(r=>!indFilter || r.industry===indFilter);
  html += `<div id="trigBox">${trigList.map(card).join('') ||
    '<div style="color:var(--muted);font-size:12.5px;padding-bottom:8px">该行业无完全触发的标的</div>'}</div>`;
  if(nearList.length){
    html += `<div style="color:var(--amber);font-weight:700;font-size:12.5px;margin:12px 4px 6px">
      ▸ 接近触发 — 差少数条件，可提前关注</div><div id="nearBox">${nearList.map(card).join('')}</div>`;
  }
  document.getElementById('list').innerHTML =
    html || '<div style="color:var(--muted);text-align:center;padding:40px">无匹配标的</div>';
}

function exportCsv(){
  const R = SCREEN_RESULT; if(!R) return;
  const label = n => IND_LABEL[n] || PAT_CN[n] || n;
  const rows = [["代码","名称","状态","满足度%","买点","止损","满足条件","缺失条件"]];
  [...(R.triggered||[]), ...(R.near||[])].forEach(r=>{
    rows.push([r.symbol, r._name||'', r.status==='triggered'?'已触发':'接近',
      r.score, r.entry_price||'', r.stop_price||'',
      (r.met||[]).map(label).join('+'), (r.missing||[]).map(label).join('+')]);
  });
  const csv = '\ufeff' + rows.map(r=>r.map(x=>`"${String(x).replace(/"/g,'""')}"`).join(',')).join('\n');
  const blob = new Blob([csv], {type:'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'SPS筛选_'+new Date().toISOString().slice(0,10)+'.csv';
  a.click();
}

function resort(){
  const R = SCREEN_RESULT; if(!R) return;
  const mode = document.getElementById('sortSel').value;
  const cmp = {
    score: (a,b)=>(b.score||0)-(a.score||0),
    entry: (a,b)=>(a.entry_price??1e9)-(b.entry_price??1e9)
  }[mode];
  R.triggered.sort(cmp);
  R.near.sort(cmp);
  renderScreenResult();
  document.getElementById('sortSel').value = mode;
}

// ================= AI 解读 & 设置（用户自带 API Key，本地存储） =================
async function openAISettings(){
  let cfg={base_url:'https://api.openai.com/v1', model:'', api_key_masked:'', api_key_set:false};
  try{ cfg={...cfg, ...(await (await fetch('/api/ai/config')).json())}; }catch(e){}
  // 复用回放弹层思路：简单模态
  let modal=document.getElementById('aiModal');
  if(!modal){
    modal=document.createElement('div');
    modal.id='aiModal';
    modal.style.cssText='position:fixed;inset:0;background:#000a;display:flex;align-items:center;justify-content:center;z-index:99';
    document.body.appendChild(modal);
  }
  modal.innerHTML=`<div style="background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:22px 26px;width:480px;max-width:92vw">
    <div style="font-size:16px;font-weight:800;margin-bottom:6px">⚙ AI 解读设置</div>
    <div style="font-size:11.5px;color:var(--muted);line-height:1.7;margin-bottom:12px">
      填入你自己的大模型 API Key（OpenAI 兼容接口均可：DeepSeek / 通义 / Kimi / GLM / Ollama 等）。
      Key 只保存在你本机 data/meta/ai_config.json，不会上传到任何服务器，调用费用由你自己的账户承担。</div>
    <div style="display:flex;flex-direction:column;gap:9px;font-size:12.5px">
      <label>接口地址 Base URL
        <input id="aiBase" value="${cfg.base_url||''}" placeholder="https://api.deepseek.com/v1"
          style="width:100%;padding:7px;border-radius:7px;background:#0b1120;color:var(--text);border:1px solid var(--border);margin-top:3px"></label>
      <label>API Key ${cfg.api_key_set?`<span style="color:var(--green)">（已设置 ${cfg.api_key_masked}，留空则不修改）</span>`:'<span style="color:var(--amber)">（必填）</span>'}
        <input id="aiKey" type="password" placeholder="${cfg.api_key_set?'保持不变':'sk-...'}"
          style="width:100%;padding:7px;border-radius:7px;background:#0b1120;color:var(--text);border:1px solid var(--border);margin-top:3px"></label>
      <label>模型名 Model
        <input id="aiModel" value="${cfg.model||''}" placeholder="deepseek-chat / gpt-4o-mini / qwen-plus"
          style="width:100%;padding:7px;border-radius:7px;background:#0b1120;color:var(--text);border:1px solid var(--border);margin-top:3px"></label>
    </div>
    <div id="aiTestResult" style="font-size:12px;margin-top:10px;min-height:16px"></div>
    <div style="display:flex;gap:8px;margin-top:12px;justify-content:flex-end">
      <button onclick="this.closest('#aiModal').remove()" style="width:auto;padding:7px 16px;background:#33415580">取消</button>
      <button onclick="aiTestConn()" style="width:auto;padding:7px 16px;background:#33415580">🔗 测试连接</button>
      <button onclick="aiSaveCfg()" class="green" style="width:auto;padding:7px 20px">💾 保存</button>
    </div>
  </div>`;
}
function _aiModalVals(){
  return {base_url: document.getElementById('aiBase').value.trim(),
          api_key: document.getElementById('aiKey').value.trim(),
          model: document.getElementById('aiModel').value.trim()};
}
async function aiSaveCfg(){
  const v=_aiModalVals();
  if(!v.api_key) delete v.api_key;   // 留空=不修改
  const r=await fetch('/api/ai/config',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(v)});
  const j=await r.json();
  if(j.ok){ document.getElementById('aiModal').remove(); }
  else alert('保存失败');
}
async function aiTestConn(){
  const v=_aiModalVals();
  if(!v.api_key) delete v.api_key;
  const el=document.getElementById('aiTestResult');
  el.innerHTML='<span style="color:var(--muted)">测试中…</span>';
  const r=await fetch('/api/ai/test',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(v)});
  const j=await r.json();
  el.innerHTML = j.ok
    ? `<span style="color:var(--green)">✅ 连接成功（${j.elapsed}s）：${j.reply||'OK'}</span>`
    : `<span style="color:var(--red)">❌ ${j.error||'失败'}</span>`;
}
async function aiInterpret(){
  const btn=document.getElementById('aiBtn');
  btn.disabled=true; btn.textContent='🤖 AI解读中…';
  try{
    const r=await fetch('/api/ai/interpret',{method:'POST'});
    const j=await r.json();
    if(!j.ok){
      if((j.error||'').includes('尚未配置')){ openAISettings(); return; }
      alert(j.error||'解读失败'); return;
    }
    // 展示在结果区顶部
    const list=document.getElementById('list');
    const box=document.createElement('div');
    box.style.cssText='background:linear-gradient(135deg,#13203a,#0f1a30);border:1px solid var(--accent);border-radius:12px;padding:16px 20px;margin-bottom:12px;position:relative';
    box.innerHTML=`<span onclick="this.parentElement.remove()" title="关闭"
        style="position:absolute;top:10px;right:14px;cursor:pointer;color:var(--muted)">✕</span>
      <div style="font-size:14px;font-weight:800;margin-bottom:8px">🤖 AI 解读 <span style="font-size:11px;color:var(--muted);font-weight:400">模型 ${j.model} · ${j.elapsed}s · 由你自己的 API Key 计费</span></div>
      <div style="font-size:13px;line-height:1.9;color:var(--text);white-space:pre-wrap">${j.text}</div>`;
    list.prepend(box);
    window.scrollTo(0,0);
  } finally {
    btn.disabled=false; btn.textContent='🤖 AI解读';
  }
}

loadIndicators().then(()=>{bindIndRowEvents(); renderTemplates();});
loadStrategies();
checkFresh();

// ================= 页签切换 =================
function showTab(t){
  document.querySelectorAll('.tabbtn').forEach(b=>{
    b.style.background = b.id==='tab-'+t ? 'var(--accent)' : '#33415580';
  });
  document.getElementById('view-filter').style.display = t==='filter'?'':'none';
  document.getElementById('view-pos').style.display = t==='pos'?'':'none';
  document.getElementById('view-replay').style.display = t==='replay'?'':'none';
  if(t==='pos') loadPositions();
  if(t==='replay') initReplay();
}

// ================= 持仓管理页 =================
let POS_DATA = null;
const EXIT_RULE_CN = {stop_loss:'跌破固定止损价', break_ma:'收盘跌破MA',
  mom_reverse:'N日动量转负', trail_stop:'浮盈回撤止盈%', time_stop:'持有N日仍亏损退出'};

async function loadPositions(){
  const box = document.getElementById('view-pos');
  box.innerHTML = '<div style="color:var(--muted);padding:30px;text-align:center">载入中…</div>';
  try{
    POS_DATA = await (await fetch('/api/positions')).json();
  }catch(e){ box.innerHTML='<div style="color:var(--red)">加载失败</div>'; return; }
  renderPositions();
}

function posRulesHtml(rec){
  return Object.entries(rec.rules||{}).map(([k,v])=>
    EXIT_RULE_CN[k] + (typeof v==='number'?`(${v})`:'')).join('、');
}

function renderPositions(){
  const D = POS_DATA, box = document.getElementById('view-pos');
  const s = D.diagnosis.summary;
  let html = `
  <div class="card" style="margin-bottom:12px">
    <h4>⚙️ 添加持仓（买入后登记，用于每日卖出诊断）</h4>
    <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;font-size:12.5px">
      <input id="posSym" placeholder="代码 如600519" style="width:110px;padding:6px;border-radius:7px;background:#0b1120;color:var(--text);border:1px solid var(--border)">
      <input id="posPrice" type="number" placeholder="买入价" step="any" style="width:90px;padding:6px;border-radius:7px;background:#0b1120;color:var(--text);border:1px solid var(--border)">
      <input id="posDate" type="date" style="padding:6px;border-radius:7px;background:#0b1120;color:var(--text);border:1px solid var(--border)">
      <span style="color:var(--muted)">止损%</span><input id="posStop" type="number" value="7" style="width:52px;padding:6px;border-radius:7px;background:#0b1120;color:var(--text);border:1px solid var(--border)">
    </div>
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;font-size:12px" id="ruleChks">
      ${Object.entries(D.exit_rules).map(([k,v])=>`
        <label style="display:flex;align-items:center;gap:4px;color:var(--muted)" title="${(v.desc||'').replace(/"/g,'&quot;')}">
          <input type="checkbox" class="ruleChk" data-rule="${k}" ${k==='stop_loss'?'checked':''}> ${v.label}
          ${v.valid_range?`<input type="number" class="ruleP" data-rule="${k}" value="${v.default}" style="width:48px;padding:3px;border-radius:6px;background:#0b1120;color:var(--text);border:1px solid var(--border)" disabled>`:''}
        </label>`).join('')}
    </div>
    <button class="green" style="width:auto;padding:7px 20px;margin-top:9px" onclick="addPos()">＋ 登记持仓</button>
    <span id="posMsg" style="font-size:12px;margin-left:10px"></span>
  </div>
  <div style="color:var(--muted);font-size:12.5px;margin:0 4px 8px">
    持仓中 <b style="color:var(--text)">${s.open}</b> ·
    <b style="color:var(--red)">应卖出 ${s.should_sell}</b> · 继续持有 ${s.hold}
    <span style="margin-left:8px">(诊断日 ${D.diagnosis.day||'-'}，最新K线触发仅预警待确认)</span>
  </div>`;

  const HEALTH = {
    green:  {dot:'🟢', color:'var(--green)',  label:'健康'},
    yellow: {dot:'🟡', color:'var(--amber)',  label:'警惕'},
    red:    {dot:'🔴', color:'var(--red)',    label:'危险'},
  };
  html += (D.diagnosis.results||[]).map(r=>{
    const sell = r.action==='sell'||r.action==='warn_sell';
    const sigHtml = (r.signals||[]).map(x=>
      `<div style="color:var(--red);font-size:11.5px">▸ ${EXIT_RULE_CN[x.rule]||x.rule}: ${x.detail}</div>`).join('');
    const h = HEALTH[r.health] || HEALTH.green;
    // 健康度明细：距止损余量 / MA20状态 / 距峰值回撤
    const healthDetail = [
      r.stop_buffer!=null && `距止损价 <b style="color:${r.stop_buffer<=4?'var(--red)':'var(--text)'}">${r.stop_buffer>0?'+':''}${r.stop_buffer}%</b>`,
      r.ma20!=null && `MA20 <b style="color:${r.above_ma20?'var(--green)':'var(--red)'}">${r.above_ma20?'上方':'下方'}</b>(${r.ma20})`,
      `峰值回撤 <b style="color:${r.dd_from_peak<=-12?'var(--red)':'var(--text)'}">${r.dd_from_peak}%</b>`,
    ].filter(Boolean).join(' · ');
    return `<div class="cand" style="${sell||r.health==='red'?'border-color:var(--red)':r.health==='yellow'?'border-color:var(--amber)':''}">
      <div class="r1">
        <span title="${r.health_tip||''}" style="cursor:help">${h.dot}</span>
        <span class="code">${r.symbol}</span><span class="nm" title="${r.name||''}">${r.name||''}</span>
        <span class="badge ${sell?'b-bad':r.health==='yellow'?'b-neutral':'b-good'}">${r.action==='warn_sell'?'⚠明日卖':sell?'应卖出':h.label}</span>
        <span style="margin-left:auto;font-weight:700;color:${(r.pnl_pct??0)>=0?'var(--red)':'var(--green)'}">${(r.pnl_pct??0)>=0?'+':''}${r.pnl_pct??0}%</span>
              </div>
              <div class="r2" style="flex-wrap:wrap">
                <span>成本 <b>${r.entry_price??'-'}</b></span><span>现价 <b>${r.price??'-'}</b></span>
        <span>持有 <b>${r.held_days??'-'}日</b></span>
        <span style="color:var(--muted)">规则: ${posRulesHtml(r)||'-'}</span>
        <a href="javascript:void(0)" onclick="manualSell('${r.symbol}',${r.price??0})" style="color:var(--amber)">手动平仓</a>
        <a href="javascript:void(0)" onclick="delPos('${r.symbol}')" style="color:var(--red)">删除</a>
      </div>
      <div class="r2" style="flex-wrap:wrap;font-size:11.5px">
        <span title="${r.health_tip||''}">${h.dot} ${healthDetail}</span>
      </div>
      ${sigHtml}
    </div>`;
  }).join('') || '<div style="color:var(--muted);padding:20px;text-align:center">暂无持仓。筛选出已触发标的买入后，在此登记即可每日自动诊断卖出。</div>';

  if((D.history||[]).length){
    html += `<div style="color:var(--amber);font-weight:700;font-size:12.5px;margin:14px 4px 6px">▸ 平仓历史</div>`;
    html += D.history.slice().reverse().map(r=>`
      <div class="cand" style="cursor:default">
        <div class="r1"><span class="code">${r.symbol}</span><span class="nm" title="${r.name||''}">${r.name||''}</span>
            <span style="margin-left:auto;font-weight:700;color:${(r.pnl_pct??0)>=0?'var(--red)':'var(--green)'}">${(r.pnl_pct??0)>=0?'+':''}${r.pnl_pct??0}%</span></div>
                    <div class="r2"><span>${r.entry_date} → ${r.exit_date}</span>
          <span>退出原因: ${EXIT_RULE_CN[r.exit_reason]||r.exit_reason}</span></div>
      </div>`).join('');
  }
  // 持仓自检回放入口
  html += `<div style="color:var(--amber);font-weight:700;font-size:12.5px;margin:14px 4px 6px">▸ 持仓自检回放</div>
  <div class="card" style="margin-bottom:12px">
    <div style="font-size:12px;color:var(--muted);margin-bottom:8px">用你当前勾选的筛选条件，回放每只持仓的买入日：看看"如果那天用系统筛，这只票会不会被选中、其后走势如何"。用于自检买入逻辑是否站得住。</div>
    <button class="green" style="width:auto;padding:7px 20px;margin:0" onclick="runPosReplay()">🔍 回放我的持仓</button>
    <span id="posReplayMsg" style="font-size:12px;color:var(--muted);margin-left:10px"></span>
  </div>
  <div id="posReplayResult"></div>`;
  box.innerHTML = html;
}

async function runPosReplay(){
  const conds = collectConds();
  if(!Object.keys(conds).length){
    document.getElementById('posReplayMsg').textContent='⚠ 请先在左侧勾选筛选条件（或选一个风格）';
    showTab('filter'); return;
  }
  const msg = document.getElementById('posReplayMsg');
  const res = document.getElementById('posReplayResult');
  msg.textContent='回放中…';
  res.innerHTML='';
  const r = await fetch('/api/positions_replay',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({conditions: conds,
      stop_pct: getStopPct()})});
  const j = await r.json();
  if(!r.ok){ msg.textContent='❌ '+(j.error||'失败'); return; }
  msg.textContent = `完成：${j.results.length} 只持仓`;
  if(!j.results.length){ res.innerHTML='<div style="color:var(--muted);padding:14px">暂无持仓</div>'; return; }
  res.innerHTML = j.results.map(p=>{
    const ok = p.would_trigger;
    const color = ok?'var(--green)':'var(--red)';
    return `<div class="cand" onclick="openDetail('${p.symbol}','pos')" title="点击查看K线与基本面">
      <div class="r1">
        <span class="code">${p.symbol}</span><span class="nm" title="${p.name||''}">${p.name||''}</span>
        <span class="badge ${ok?'b-good':'b-bad'}">${ok?'✓ 当日会被选中':'✗ 当日不会选中'}</span>
        <span style="margin-left:auto;font-weight:700;color:${(p.pnl_pct??0)>=0?'var(--red)':'var(--green)'}">${(p.pnl_pct??0)>=0?'+':''}${p.pnl_pct??0}%</span>
      </div>
      <div class="r2" style="flex-wrap:wrap">
        <span>买入日 <b>${p.entry_date}</b></span>
        <span>当日满足 <b style="color:${color}">${p.met_count}/${p.total}</b> 条件</span>
        ${p.missing&&p.missing.length?`<span style="color:var(--amber)">缺: ${p.missing.map(n=>IND_LABEL[n]||n).join('、')}</span>`:''}
        ${p.fwd20!=null?`<span>买入后至今 <b style="color:${p.pnl_pct>=0?'var(--red)':'var(--green)'}">${(p.pnl_pct>0?'+':'')}${p.pnl_pct}%</b></span>`:''}
      </div>
    </div>`;
  }).join('');
}

document.addEventListener('change', e=>{
  if(e.target.classList.contains('ruleChk')){
    const row = e.target.closest('label');
    const p = row && row.querySelector('.ruleP');
    if(p) p.disabled = !e.target.checked;
  }
});

function collectRules(){
  const rules = {};
  document.querySelectorAll('.ruleChk:checked').forEach(c=>{
    const k = c.dataset.rule;
    const p = document.querySelector(`.ruleP[data-rule="${k}"]`);
    rules[k] = p ? parseFloat(p.value) : true;
  });
  return rules;
}

async function addPos(){
  const msg = document.getElementById('posMsg');
  const body = {symbol: document.getElementById('posSym').value.trim(),
    name: '', entry_price: parseFloat(document.getElementById('posPrice').value),
    entry_date: document.getElementById('posDate').value,
    stop_pct: parseFloat(document.getElementById('posStop').value)||7,
    rules: collectRules()};
  if(!body.symbol || !body.entry_price){ msg.textContent='❌ 请填代码和买入价'; msg.style.color='var(--red)'; return; }
  const r = await fetch('/api/positions',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const j = await r.json();
  if(!r.ok){ msg.textContent='❌ '+(j.error||'失败'); msg.style.color='var(--red)'; return; }
  msg.textContent='✅ 已登记'; msg.style.color='var(--green)';
  loadPositions();
}

async function delPos(sym){
  if(!confirm('删除 '+sym+' 的持仓记录？')) return;
  await fetch('/api/positions/'+sym,{method:'DELETE'});
  loadPositions();
}

async function manualSell(sym, price){
  const v = prompt('平仓 '+sym+'：卖出价格', price||'');
  if(v===null) return;
  await fetch('/api/positions/'+sym+'/close',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({exit_price:parseFloat(v), reason:'manual'})});
  loadPositions();
}

async function backtestExit(){
  const rules = collectRules();
  if(!Object.keys(rules).length){ alert('请先勾选卖出规则'); return; }
  const el = document.getElementById('exitBtResult');
  el.innerHTML = '<span style="color:var(--muted)">回测中（全市场随机入场口径）…</span>';
  const r = await fetch('/api/backtest_exit',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({rules, stop_pct: parseFloat(document.getElementById('posStop')?.value)||7})});
  const j = await r.json();
  if(!r.ok){ el.innerHTML='<span style="color:var(--red)">'+(j.error||'失败')+'</span>'; return; }
  el.innerHTML = `<span style="color:var(--text)">样本 <b>${j.n_trades}</b> 笔 ·
    20日口径胜率 <b style="color:${j.win20>=50?'var(--green)':'var(--red)'}">${j.win20}%</b> ·
    平均收益 <b>${j.avg_pnl}%</b> · 平均持有 <b>${j.avg_hold_days}日</b>
    <span style="color:var(--muted)">(${j.cost_model})</span></span>`;
}

// ================= 历史回放页 =================
function initReplay(){
  const box = document.getElementById('view-replay');
  if(box.dataset.init) return;
  box.dataset.init = '1';
  box.innerHTML = `
  <div class="card" style="margin-bottom:12px">
    <h4>⏪ 历史回放 — 用指定日期的数据重现"当天系统会选出什么"</h4>
    <div style="display:flex;gap:8px;align-items:center;font-size:12.5px;flex-wrap:wrap">
      <span style="color:var(--muted)">回放日期</span>
      <input id="replayDate" type="date" style="padding:6px;border-radius:7px;background:#0b1120;color:var(--text);border:1px solid var(--border)">
      <button class="green" style="width:auto;padding:7px 18px;margin:0" onclick="doReplay()">▶ 回放</button>
      <span id="replayMsg" style="font-size:12px;color:var(--muted)">回放 = 用你当前勾选的条件，在全市场所有股票上模拟"那天收盘会选出哪些票"，并统计其后20日真实收益。需先勾选筛选条件。</span>
    </div>
    <div style="display:flex;gap:6px;margin-top:8px;flex-wrap:wrap;align-items:center;font-size:11.5px">
      <span style="color:var(--muted)">快捷：</span>
      <button onclick="replayQuick(30)" style="width:auto;padding:3px 10px;margin:0;font-size:11px;background:#33415580">30天前</button>
      <button onclick="replayQuick(90)" style="width:auto;padding:3px 10px;margin:0;font-size:11px;background:#33415580">90天前</button>
      <button onclick="replayQuick(180)" style="width:auto;padding:3px 10px;margin:0;font-size:11px;background:#33415580">半年前</button>
      <button onclick="replayQuick(365)" style="width:auto;padding:3px 10px;margin:0;font-size:11px;background:#33415580">一年前</button>
      <span id="replayRange" style="color:var(--muted);margin-left:8px"></span>
    </div>
  </div>
  <div id="replayResult"></div>`;
  // 默认日期：90天前；并显示可回放范围（数据最早/最新日）
  const d = new Date(Date.now() - 1000*60*60*24*90);
  const el=document.getElementById('replayDate');
  el.value = d.toISOString().slice(0,10);
  fetch('/api/data_status').then(r=>r.json()).then(j=>{
    if(j.ready){
      const last = new Date(j.last_date);
      el.max = j.last_date;
      const min = new Date(last.getTime() - 1000*60*60*24*365*3); // 数据最早约3年前
      el.min = min.toISOString().slice(0,10);
      document.getElementById('replayRange').textContent =
        `可回放范围：约 ${min.toISOString().slice(0,10)} ~ ${j.last_date}（数据最新日）`;
    }
  }).catch(()=>{});
}
function replayQuick(days){
  const d = new Date(Date.now() - 1000*60*60*24*days);
  document.getElementById('replayDate').value = d.toISOString().slice(0,10);
}

async function doReplay(){
  const conds = collectConds();
  if(!Object.keys(conds).length){ alert('请先在左侧勾选指标'); showTab('filter'); return; }
  const day = document.getElementById('replayDate').value;
  if(!day){ alert('请选择回放日期'); return; }
  const res = document.getElementById('replayResult');
  const msg = document.getElementById('replayMsg');
  res.innerHTML = '<div style="color:var(--muted);padding:20px;text-align:center">回放计算中（全市场截断到指定日期）…</div>';
  const r = await fetch('/api/replay',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({day, conditions: conds,
      stop_pct: parseFloat(document.getElementById('stopPct').value)||7})});
  if(!r.ok){
    const e = await r.json().catch(()=>({}));
    res.innerHTML = `<div style="color:var(--red);padding:16px">${e.error||'回放失败'}</div>`; return;}
  const j = await r.json();
  msg.textContent = `回放日 ${j.day} · 其后走势统计截至 ${j.asof}`;
  const card = t => `
      <div class="cand" onclick="openDetail('${t.symbol}','replay')">
      <div class="r1"><span class="code">${t.symbol}</span><span class="nm" title="${t._name||''}">${t._name||''}</span>
        <span class="badge b-good">已触发</span>
        <span style="margin-left:auto;font-weight:700;color:${(t.fwd20??0)>=0?'var(--red)':'var(--green)'}">${t.fwd20==null?'-':(t.fwd20>0?'+':'')+t.fwd20+'%'}</span>
      </div>
      <div class="r2"><span>信号日 <b>${t.signal_date}</b></span>
        <span>买点 <b>${t.entry_price??'-'}</b></span>
        <span>其后20日收益（含成本）<b>${t.fwd20==null?'不足20日':(t.fwd20>0?'+':'')+t.fwd20+'%'}</b></span></div>
    </div>`;
  const stats = j.summary;
  let html = `<div style="color:var(--muted);font-size:12.5px;margin:0 4px 10px">
    当日已触发 <b style="color:var(--green)">${(j.triggered||[]).length}</b> 只 ·
    其后20日：胜率 <b style="color:${stats.win20>=50?'var(--green)':'var(--red)'}">${stats.win20??'-'}%</b>
    平均收益 <b>${stats.avg20??'-'}%</b> · 这就是"如果那天用了系统"的真实成绩单</div>`;
  const sorted = (j.triggered||[]).slice().sort((a,b)=>{
    const va = a.fwd20==null ? -1e9 : a.fwd20;
    const vb = b.fwd20==null ? -1e9 : b.fwd20;
    return vb - va;    // 收益率高→低，无数据的垫底
  });
  html += sorted.map(card).join('') || '<div style="color:var(--muted);padding:20px">该日无触发标的</div>';
  res.innerHTML = html;
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("SPS 交互界面: http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
