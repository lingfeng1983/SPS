"""持仓管理与卖出规则引擎。

设计：
- 持仓记录存 data/positions.json：[{symbol, name, entry_date, entry_price,
  stop_price, stop_pct, rules, qty?, status, exit_date, exit_price, exit_reason}]
- 状态: open / closed
- 每日诊断 run_diagnosis(daily, wide, day=None)：
  对每个 open 持仓逐条评估卖出规则，返回应卖出/预警列表
- 卖出规则（可多选，任一触发即卖出，reason 记录首触发的规则）：
    stop_loss   : 跌破止损价（entry×(1-stop%)，随持仓固定）
    break_ma    : 收盘跌破 N 日均线（默认10）
    mom_reverse : N日动量转负（默认20日）
    trail_stop  : 浮盈回撤 X% 止盈（从持有期最高收盘回撤）
    time_stop   : 持有超过 N 日仍亏损（默认20日）
- 规则参数均有 valid_range，超出在后端校验拒绝。

回测口径：入场 t 收盘确认 → t+1 开盘进场；退出以确认日收盘价近似（保守）。
"""
from __future__ import annotations

import json
import time

import pandas as pd

from sps.paths import DATA_DIR

POS_FILE = DATA_DIR / "positions.json"

# 卖出规则定义：label / 默认参数 / 有效区间 / 说明
EXIT_RULES = {
    "stop_loss": {
        "label": "跌破固定止损价", "has_param": False,
        "desc": "收盘价跌破买点×(1-止损%)。与买入侧止损价联动，最基础的保命规则。",
    },
    "break_ma": {
        "label": "收盘跌破 MA", "default": 10, "valid_range": [5, 60],
        "desc": "收盘跌破N日均线=趋势破坏。N小→灵敏早退但易被洗；N大→容忍回调但回吐多。",
    },
    "mom_reverse": {
        "label": "N日动量转负", "default": 20, "valid_range": [10, 60],
        "desc": "近N日累计涨幅<0=动能消失。避免在深度回调后才离场，动量体系的自然退出。",
    },
    "trail_stop": {
        "label": "浮盈回撤止盈 %", "default": 10, "valid_range": [3, 30],
        "desc": "从持有期最高收盘回撤X%即锁定利润。让利润奔跑、回撤到口子就走。",
    },
    "time_stop": {
        "label": "持有N日仍亏损则退出", "default": 20, "valid_range": [5, 60],
        "desc": "持有N日仍不赚=当初判断错误，认错离场把资金换成新机会。欧奈尔风格的时间止损。",
    },
}

COMMISSION_RATE = 0.00025   # 佣金 万2.5（双边）
STAMP_TAX = 0.0005          # 印花税 卖出 千0.5
SLIPPAGE = 0.003            # 滑点 0.3%（按保守估计）
COST_PER_TRADE = COMMISSION_RATE * 2 + STAMP_TAX + SLIPPAGE * 2   # 往返合计 ~0.68%

# ---------------------------------------------------------------- 存储

def load_positions() -> list[dict]:
    if POS_FILE.exists():
        try:
            return json.loads(POS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_positions(recs: list[dict]) -> None:
    POS_FILE.parent.mkdir(parents=True, exist_ok=True)
    POS_FILE.write_text(json.dumps(recs, ensure_ascii=False, indent=1),
                        encoding="utf-8")


def validate_rules(rules: dict) -> tuple[dict, list[str]]:
    """校验卖出规则参数。rules: {rule_name: param}，无参数规则值为 true。"""
    errors, ok = [], {}
    for name, param in (rules or {}).items():
        meta = EXIT_RULES.get(name)
        if meta is None:
            errors.append(f"{name}: 未知卖出规则")
            continue
        vr = meta.get("valid_range")
        if vr is None:
            ok[name] = True
            continue
        try:
            v = float(param)
        except (TypeError, ValueError):
            errors.append(f"{meta['label']}: 参数缺失")
            continue
        if not (vr[0] <= v <= vr[1]):
            errors.append(f"{meta['label']}: 参数 {v} 超出有效区间 [{vr[0]},{vr[1]}]")
            continue
        ok[name] = v
    return ok, errors


# ---------------------------------------------------------------- 持仓 CRUD

def add_position(symbol: str, name: str, entry_price: float, entry_date: str,
                 stop_pct: float, rules: dict, qty: float | None = None,
                 mode: str = "paper") -> dict:
    if mode not in ("paper", "live"):
        raise ValueError(f"未知持仓模式: {mode}")
    recs = load_positions()
    # 同一票若已有 open 持仓则拒绝（简化：不摊平）
    for r in recs:
        if r["symbol"] == symbol and r["status"] == "open":
            raise ValueError(f"{symbol} 已有持仓中记录，请先卖出再新增")
    rec = {
        "symbol": symbol, "name": name,
        "entry_price": round(float(entry_price), 3),
        "entry_date": entry_date,
        "stop_pct": float(stop_pct),
        "stop_price": round(float(entry_price) * (1 - float(stop_pct) / 100), 3),
        "rules": rules, "qty": qty,
        "mode": mode,   # paper=模拟仓 / live=实盘仓
        "status": "open",
        "added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    recs.append(rec)
    save_positions(recs)
    return rec


def close_position(symbol: str, exit_price: float, exit_date: str,
                   reason: str = "manual") -> dict | None:
    recs = load_positions()
    for r in reversed(recs):
        if r["symbol"] == symbol and r["status"] == "open":
            r.update({
                "status": "closed",
                "exit_price": round(float(exit_price), 3),
                "exit_date": exit_date,
                "exit_reason": reason,
                "pnl_pct": round((float(exit_price) / float(r["entry_price"]) - 1) * 100, 2),
            })
            save_positions(recs)
            return r
    return None


def delete_position(symbol: str) -> bool:
    """从记录中彻底删除（含历史）。用于误录入。"""
    recs = load_positions()
    n0 = len(recs)
    recs = [r for r in recs if r["symbol"] != symbol]
    save_positions(recs)
    return len(recs) < n0


# ---------------------------------------------------------------- 诊断引擎

def _first_true_pos(series: pd.Series, start: int) -> int | None:
    """从 start 位置起第一个 True 的位置。"""
    if series is None:
        return None
    sub = series.iloc[start:]
    idx = sub.idxmax() if sub.any() else None
    if idx is None:
        return None
    return sub.index.get_loc(idx) + start


def diagnose_position(rec: dict, df: pd.DataFrame, day: str | None = None,
                      is_last_day: bool = False) -> dict:
    """对单个持仓做诊断。day=None 用最新K线。

    返回 {symbol, price, pnl_pct, signals: [{rule, detail}], action: sell|warn|hold}
    is_last_day=True 表示 day 是最新K线（实时场景，卖出一律 warn 供人工确认）。
    """
    n = len(df)
    if day is not None:
        day_ts = pd.Timestamp(day)
        if day_ts not in df.index:
            # 停牌等：用该日之前的最后K线
            sub = df.loc[:day_ts]
            if sub.empty:
                return {"symbol": rec["symbol"], "action": "hold", "signals": [],
                        "note": "no_data"}
            pos = len(sub) - 1
        else:
            pos = df.index.get_loc(day_ts)
    else:
        pos = n - 1

    c = df["C"].iloc[:pos + 1]
    price = float(c.iloc[-1])
    entry = float(rec["entry_price"])
    pnl = (price / entry - 1) * 100
    held_days = int(pos - df.index.get_loc(pd.Timestamp(rec["entry_date"]))
                    ) if pd.Timestamp(rec["entry_date"]) in df.index else 0

    signals = []
    rules = rec.get("rules") or {}

    if "stop_loss" in rules:
        if price <= float(rec["stop_price"]):
            signals.append({"rule": "stop_loss",
                            "detail": f"收盘 {price:.2f} ≤ 止损价 {rec['stop_price']:.2f}"})

    if "break_ma" in rules:
        p = int(rules["break_ma"])
        ma = df["C"].rolling(p, min_periods=p).mean()
        if pos >= p and price < float(ma.iloc[pos]):
            signals.append({"rule": "break_ma",
                            "detail": f"收盘 {price:.2f} 跌破 MA{p} {float(ma.iloc[pos]):.2f}"})

    if "mom_reverse" in rules:
        p = int(rules["mom_reverse"])
        if pos >= p:
            mom = price / float(c.iloc[-1 - p]) - 1
            if mom < 0:
                signals.append({"rule": "mom_reverse",
                                "detail": f"{p}日动量 {mom*100:.1f}% < 0"})

    if "trail_stop" in rules:
        p = float(rules["trail_stop"])
        peak = float(c.max())
        dd = (price / peak - 1) * 100
        if peak > entry and dd <= -p:
            signals.append({"rule": "trail_stop",
                            "detail": f"距持有期最高 {peak:.2f} 回撤 {dd:.1f}% ≥ {p}%"})

    if "time_stop" in rules:
        p = int(rules["time_stop"])
        if held_days >= p and pnl <= 0:
            signals.append({"rule": "time_stop",
                            "detail": f"持有 {held_days} 日收益 {pnl:.1f}% ≤ 0"})

    action = "sell" if signals else "hold"
    if is_last_day and signals:
        action = "warn_sell"  # 实时：提示用户明日开盘执行
    # ---- 健康度红绿灯（方案B）：距各卖出线的余量，提前预警 ----
    ma20 = float(df["C"].rolling(20, min_periods=20).mean().iloc[pos]) if pos >= 19 else None
    above_ma20 = (price > ma20) if ma20 else None
    peak = float(c.max())
    dd_from_peak = round((price / peak - 1) * 100, 1)          # 距持有期最高回撤%
    stop_price = float(rec.get("stop_price") or 0)
    stop_buffer = round((price / stop_price - 1) * 100, 1) if stop_price > 0 else None  # 距止损还剩%
    # 绿=价格在MA20上且回撤<8%且距止损>4%；黄=任一接近；红=触发卖出信号
    if signals:
        health, health_tip = "red", "已触发卖出信号"
    elif above_ma20 is False or (stop_buffer is not None and stop_buffer <= 4) or dd_from_peak <= -12:
        health, health_tip = "yellow", "接近危险线：趋势转弱或回撤较深，提高警惕"
    else:
        health, health_tip = "green", "趋势健康：在MA20上方且回撤可控"
    return {"symbol": rec["symbol"], "name": rec.get("name", ""),
            "price": round(price, 3), "pnl_pct": round(pnl, 2),
            "held_days": held_days, "signals": signals, "action": action,
            "health": health, "health_tip": health_tip,
            "above_ma20": above_ma20, "ma20": round(ma20, 2) if ma20 else None,
            "dd_from_peak": dd_from_peak,
            "stop_price": stop_price or None, "stop_buffer": stop_buffer}


def run_diagnosis(daily: dict[str, pd.DataFrame], day: str | None = None) -> dict:
    """对全部 open 持仓跑诊断。返回 {results, summary}。"""
    recs = [r for r in load_positions() if r["status"] == "open"]
    results = []
    latest = max((df.index[-1] for df in daily.values()), default=None)
    latest_str = str(latest.date()) if latest is not None else None
    for rec in recs:
        df = daily.get(rec["symbol"])
        if df is None:
            results.append({"symbol": rec["symbol"], "name": rec.get("name", ""),
                            "action": "hold", "signals": [], "note": "no_kline"})
            continue
        results.append(diagnose_position(rec, df, day=day,
                                         is_last_day=(day in (None, latest_str))))
    sells = [r for r in results if r["action"] in ("sell", "warn_sell")]
    return {"day": day or latest_str,
            "results": results,
            "summary": {"open": len(recs), "should_sell": len(sells),
                        "hold": len(recs) - len(sells)}}


# ---------------------------------------------------------------- 卖出规则回测

def backtest_exit_rules(daily: dict[str, pd.DataFrame],
                        rules: dict, stop_pct: float = 7.0,
                        cost: bool = True) -> dict:
    """对给定卖出规则组合做历史回测。

    入场：模拟买入侧——用每只股票过去所有"满足 demo 入场条件"的日子不现实，
    这里采用均匀采样口径：对每只股票每 20 个交易日取一个入场点
    （t 收盘 → t+1 开盘进场），统计每笔交易按规则退出后的收益分布。
    这回答的问题是"这些卖出规则在随机入场下表现如何"，与参数徽章的
    条件信号口径互补。

    返回 {n_trades, win20, avg_pnl, avg_hold_days, per_rule_counts}
    """
    per_rule = {r: 0 for r in rules}
    pnls, holds = [], []
    for sym, df in daily.items():
        if len(df) < 120:
            continue
        o = df["O"].to_numpy()
        c = df["C"].to_numpy()
        idx = df.index
        for start in range(60, len(df) - 3, 20):   # 每20日采样一个入场
            entry_pos = None
            for k in range(1, 4):
                if start + k < len(df) and o[start + k] > 0:
                    entry_pos = start + k
                    break
            if entry_pos is None:
                continue
            entry = float(o[entry_pos])
            stop = entry * (1 - stop_pct / 100)
            exit_pos, reason = None, None
            held_peak = entry
            for i in range(entry_pos, len(df)):
                price = float(c[i])
                held_peak = max(held_peak, price)
                # 各规则并列检查（任一触发即退出）；同日多规则触发时
                # 按保命优先级取一：止损 > 破线 > 动量 > 止盈 > 时间
                hits = []
                if "stop_loss" in rules and price <= stop:
                    hits.append(("stop_loss", price))
                if "break_ma" in rules:
                    p = int(rules["break_ma"])
                    if i >= entry_pos + p:
                        ma = float(df["C"].iloc[i - p + 1:i + 1].mean())
                        if price < ma:
                            hits.append(("break_ma", price))
                if "mom_reverse" in rules:
                    p = int(rules["mom_reverse"])
                    j = i - p
                    if j >= entry_pos and j < len(c):
                        if price / float(c[j]) - 1 < 0:
                            hits.append(("mom_reverse", price))
                if "trail_stop" in rules:
                    p = float(rules["trail_stop"])
                    if held_peak > entry and price / held_peak - 1 <= -p / 100:
                        hits.append(("trail_stop", price))
                if "time_stop" in rules:
                    p = int(rules["time_stop"])
                    if i - entry_pos >= p and price / entry - 1 <= 0:
                        hits.append(("time_stop", price))
                # 无论是否触发：最后一日强制平仓（open trade 计浮动）
                if hits:
                    exit_pos, reason = i, hits[0][0]
                    break
                if i == len(df) - 1:
                    exit_pos, reason = i, "eod"
            if exit_pos is None or exit_pos <= entry_pos:
                continue
            exit_price = float(c[exit_pos])
            # 次日开盘退出近似 → 直接用确认日收盘，保守
            pnl = exit_price / entry - 1
            if cost:
                pnl -= (COMMISSION_RATE * 2 + STAMP_TAX + SLIPPAGE * 2)
            pnls.append(pnl * 100)
            holds.append(exit_pos - entry_pos)
            if reason in per_rule:
                per_rule[reason] += 1
    if not pnls:
        return {"n_trades": 0}
    import numpy as np
    a = np.array(pnls)
    return {"n_trades": len(a),
            "win20": round(float((a > 0).mean()) * 100, 1),
            "avg_pnl": round(float(a.mean()), 2),
            "avg_hold_days": int(np.mean(holds)),
            "per_rule_counts": per_rule,
            "cost_model": f"佣金{COMMISSION_RATE*1e4:.0f}万双边+印花税{STAMP_TAX*1e3:.1f}千+滑点{SLIPPAGE*1e3:.0f}千" if cost else "无"}


# ---------------------------------------------------------------- 模拟盘复盘

def paper_record_count() -> int:
    """模拟仓记录总数（开仓+已平仓），用于新手模式门槛。

    mode 字段引入前的存量记录一律视为模拟——与升级前的实际使用方式一致，
    避免老用户的模拟跟踪史被误标成实盘、并被新手门槛卡住。
    """
    return sum(1 for r in load_positions() if r.get("mode", "paper") == "paper")


def review_report(days: int = 30) -> dict:
    """模拟盘复盘：统计近 N 天平仓的模拟仓成绩与在持清单。

    这是"教练模式"的核心输出——用真实跟踪记录回答
    "跟着系统模拟操作一段时间后，成绩到底如何"。
    """
    recs = [r for r in load_positions() if r.get("mode", "paper") == "paper"]
    now = time.time()

    def _within(d: str) -> bool:
        try:
            t = time.mktime(time.strptime((d or "")[:10], "%Y-%m-%d"))
        except (ValueError, TypeError):
            return False
        return now - t <= days * 86400

    closed = [r for r in recs if r["status"] == "closed" and _within(r.get("exit_date", ""))]
    open_recs = [r for r in recs if r["status"] == "open"]
    out = {"days": days, "n_closed": len(closed), "n_open": len(open_recs),
           "n_total": len(recs)}
    if closed:
        pnls = [float(r.get("pnl_pct") or 0.0) for r in closed]
        reasons: dict[str, int] = {}
        holds = []
        for r in closed:
            reasons[r.get("exit_reason", "manual")] = \
                reasons.get(r.get("exit_reason", "manual"), 0) + 1
            try:
                d1 = time.strptime(r["entry_date"][:10], "%Y-%m-%d")
                d2 = time.strptime(r["exit_date"][:10], "%Y-%m-%d")
                holds.append(max(0.0, (time.mktime(d2) - time.mktime(d1)) / 86400))
            except (ValueError, TypeError):
                pass
        out.update({
            "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1),
            "avg_pnl": round(sum(pnls) / len(pnls), 2),
            "best": round(max(pnls), 2),
            "worst": round(min(pnls), 2),
            "avg_hold_days": round(sum(holds) / len(holds)) if holds else None,
            "exit_reasons": reasons,
        })
    out["open_positions"] = [
        {"symbol": r["symbol"], "name": r.get("name", ""),
         "entry_date": r["entry_date"], "entry_price": r["entry_price"]}
        for r in open_recs]
    return out
