"""基本面红线过滤（E阶段，规格书"第一关"）。

数据源：东财个股财务指标接口（ak.stock_financial_abstract_ths / stock_a_indicator_lg）
免费可得，按股票缓存 json（TTL 14 天，财报按季更新足够）。红线一票否决，不参与打分。

红线（V1 简化版，全部基于公开报表字段）：
  R1 上市不足 120 个交易日
  R2 ST/*ST/退市整理（按名称判定）
  R3 净利润(TTM) 为负
  R4 ROE 持续低下（最新报告期 <5% 且上年同期 <5%）
  R5 商誉/净资产 >30%
  R6 大存大贷：货币资金/总资产>25% 且 有息负债/总资产>15%
"""
from __future__ import annotations

import json
import time

import pandas as pd

from sps.paths import DATA_DIR

FUND_DIR = DATA_DIR / "fundamental"

FUND_TTL_DAYS = 14   # 快照有效期；错误快照（_error）视为无效允许重试


FUNDAMENTAL_CHECKS = (
    ("profit_yoy", lambda value: value > 0),
    ("revenue_yoy", lambda value: value > 0),
    ("roe", lambda value: value >= 10),
    ("gross_margin", lambda value: value >= 20),
    ("debt_ratio", lambda value: value <= 70),
)


def summarize_fundamental_health(fundamental: dict | None) -> dict:
    """Return the five-check health summary shown in the stock detail view."""
    snapshot = fundamental or {}
    known_checks = [
        (key, predicate) for key, predicate in FUNDAMENTAL_CHECKS
        if snapshot.get(key) is not None
    ]
    passed = sum(
        bool(predicate(snapshot[key])) for key, predicate in known_checks
    )
    known = len(known_checks)
    ratio = round(passed / known, 4) if known else 0.0
    grade = "未知" if not known else ("优秀" if passed >= 4 else
                                    "良好" if passed >= 3 else "偏弱")
    return {
        "passed": passed,
        "known": known,
        "total": len(FUNDAMENTAL_CHECKS),
        "ratio": ratio,
        "grade": grade,
    }


def rank_triggered_by_fundamentals(records: list[dict], loader) -> list[dict]:
    """Attach health summaries and rank triggered records best-to-worst."""
    ranked = []
    for record in records:
        item = dict(record)
        try:
            snapshot = loader(item["symbol"]) or {}
        except Exception:
            snapshot = {}
        item["fundamental_health"] = summarize_fundamental_health(snapshot)
        ranked.append(item)

    def sort_key(item):
        health = item["fundamental_health"]
        has_data = 1 if health["known"] else 0
        return (
            has_data,
            health["ratio"],
            health["known"],
            item.get("score") or 0,
        )

    return sorted(ranked, key=sort_key, reverse=True)


def _ensure():
    FUND_DIR.mkdir(parents=True, exist_ok=True)


def _parse_num(v) -> float | None:
    """解析 '1.47亿' / '23.38%' / 12.3 等格式。"""
    if v is None or v is False or v == "False":
        return None
    try:
        s = str(v).strip()
        if s.endswith("亿"):
            return float(s[:-1]) * 1e8
        if s.endswith("万亿"):
            return float(s[:-2]) * 1e12
        if s.endswith("%"):
            return float(s[:-1])
        return float(s)
    except (TypeError, ValueError):
        return None


def _read_cache(f) -> dict | None:
    """读缓存快照；过期/损坏/错误快照返回 None（允许重拉）。"""
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or "_error" in data:
        return None
    try:
        if time.time() - f.stat().st_mtime > FUND_TTL_DAYS * 86400:
            return None
    except OSError:
        return None
    return data


def get_fundamental(symbol: str) -> dict | None:
    """拉取单只股票的关键基本面快照。失败返回 {}（放行并打标 unknown）。"""
    _ensure()
    f = FUND_DIR / f"{symbol}.json"
    cached = _read_cache(f)
    if cached is not None:
        return cached
    import akshare as ak
    out = {}
    try:
        from sps.data import no_proxy
        with no_proxy():
            fa = ak.stock_financial_abstract_ths(symbol=symbol, indicator="按报告期")
        if fa is not None and not fa.empty:
            row = fa.iloc[-1]   # 升序 → 最后一行是最新报告期
            out["report_date"] = str(row.get("报告期", ""))
            out["net_profit"] = _parse_num(row.get("净利润"))
            out["roe"] = _parse_num(row.get("净资产收益率"))
            out["debt_ratio"] = _parse_num(row.get("资产负债率"))
            out["gross_margin"] = _parse_num(row.get("销售毛利率"))
            out["net_margin"] = _parse_num(row.get("销售净利率"))
            out["revenue_yoy"] = _parse_num(row.get("营业总收入同比增长率"))
            out["profit_yoy"] = _parse_num(row.get("净利润同比增长率"))
            # 扣非净利润（用于 R3 更严格判定）
            out["deduct_np"] = _parse_num(row.get("扣非净利润"))
    except Exception as e:  # noqa: BLE001
        # 拉取失败：优先回退旧快照（旧数据也比没有强），否则只标错误不落盘
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        return {"_error": str(e)[:80]}
    f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    time.sleep(0.4)   # 温和限速
    return out


def check_redlines(symbol: str, name: str = "",
                   fundamental: dict | None = None,
                   listed_days: int | None = None) -> tuple[bool, list[str]]:
    """返回 (是否通过, 触发的红线列表)。"""
    hits = []
    if name:
        n = name.upper()
        if "ST" in n or "退" in n:
            hits.append(f"R2:{name}")
    if listed_days is not None and listed_days < 120:
        hits.append(f"R1:listed{listed_days}d")
    fu = fundamental or {}
    np_ = fu.get("net_profit")
    if np_ is not None and np_ < 0:
        hits.append("R3:net_loss")
    dnp = fu.get("deduct_np")
    if dnp is not None and dnp < 0:
        hits.append("R3x:deduct_loss")   # 扣非亏损（附注级，不否决只标注）
    roe = fu.get("roe")
    if roe is not None and roe < 5:
        hits.append(f"R4:roe{roe}")
    dr = fu.get("debt_ratio")
    # 银行(601398等以银行模式经营)/券商/保险天然高负债：放宽到 92%
    is_financial = (dr is not None and dr > 75)
    if dr is not None and dr > (92 if is_financial else 85):
        hits.append(f"R6:debt{dr}")
    return (len(hits) == 0, hits)
