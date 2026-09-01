"""基本面红线过滤（E阶段，规格书"第一关"）。

数据源：东财个股财务指标接口（ak.stock_financial_abstract_ths / stock_a_indicator_lg）
免费可得，按股票缓存 parquet。红线一票否决，不参与打分。

红线（V1 简化版，全部基于公开报表字段）：
  R1 上市不足 120 个交易日
  R2 ST/*ST/退市整理（按名称判定）
  R3 净利润(TTM) 为负
  R4 ROE 持续低下（最新报告期 <5% 且上年同期 <5%）
  R5 商誉/净资产 >30%
  R6 大存大贷：货币资金/总资产>25% 且 有息负债/总资产>15%
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FUND_DIR = DATA_DIR / "fundamental"


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


def get_fundamental(symbol: str) -> dict | None:
    """拉取单只股票的关键基本面快照。失败返回 {}（放行并打标 unknown）。"""
    _ensure()
    f = FUND_DIR / f"{symbol}.json"
    if f.exists():
        import json
        return json.loads(f.read_text(encoding="utf-8"))
    import akshare as ak
    out = {}
    try:
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
        out.setdefault("_error", str(e)[:80])
    import json
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
