"""行业归属映射（新浪 stock_classify_sina，带本地缓存，含代码段回退）。

该接口一次调用返回全市场股票及其行业分类（class 列），约6分钟拉完。
缓存7天；失败时回退旧缓存。供筛选结果的行业聚合与行业RPS使用。

代码段回退规则（针对不在 map 中的标的，按代码前缀归类到板块）：
    600xxx, 601xxx, 603xxx, 605xxx → 沪主板
    000xxx, 001xxx, 002xxx           → 深主板
    300xxx, 301xxx                    → 创业板
    688xxx                            → 科创板
    83xxxx, 87xxxx, 43xxxx           → 北交所

用法：
    from sps.industry import get_industry_map, get_industry, coverage_stats
    ind_map = get_industry_map()          # {symbol: 行业名}
    industry = get_industry("600519")     # 先查 map，再回退代码段
    stats = coverage_stats()              # 覆盖统计
"""
from __future__ import annotations

import json
import time
from typing import Literal

from sps.paths import DATA_DIR

IND_FILE = DATA_DIR / "meta" / "industry_map.json"

# 代码段 → 板块回退规则（按前缀长度从长到短匹配）
# 注意：这是板块(board)而非行业(industry)，用于无法从 map 获取行业时的保底分类
CODE_RANGES: list[tuple[str, str]] = [
    # 科创板
    ("688", "科创板"),
    # 创业板
    ("300", "创业板"),
    ("301", "创业板"),
    # 沪主板
    ("600", "沪主板"),
    ("601", "沪主板"),
    ("603", "沪主板"),
    ("605", "沪主板"),
    # 深主板（含中小板 002xxx 已并入主板）
    ("000", "深主板"),
    ("001", "深主板"),
    ("002", "深主板"),
    # 北交所
    ("83", "北交所"),
    ("87", "北交所"),
    ("43", "北交所"),
]


def get_industry_map(refresh: bool = False) -> dict[str, str]:
    """返回 {symbol: 行业名}。"""
    if not refresh and IND_FILE.exists():
        try:
            d = json.loads(IND_FILE.read_text(encoding="utf-8"))
            if d.get("map") and time.time() - d.get("_ts", 0) < 7 * 86400:
                return d["map"]
        except Exception:
            pass
    import warnings
    warnings.filterwarnings("ignore")
    try:
        import akshare as ak
        from sps.data import no_proxy
        with no_proxy():
            df = ak.stock_classify_sina()
        mapping = {str(r["code"]).zfill(6): str(r["class"])
                   for _, r in df.iterrows()
                   if r.get("class") and str(r["class"]) != "nan"}
    except Exception as e:  # noqa: BLE001
        print(f"[warn] industry fetch failed: {e}")
        mapping = {}
    if mapping:
        IND_FILE.parent.mkdir(parents=True, exist_ok=True)
        IND_FILE.write_text(json.dumps(
            {"_ts": time.time(), "source": "sina", "map": mapping},
            ensure_ascii=False), encoding="utf-8")
        return mapping
    # 失败回退旧缓存（即使过期也比空好）
    if IND_FILE.exists():
        try:
            return json.loads(IND_FILE.read_text(encoding="utf-8")).get("map", {})
        except Exception:
            pass
    return {}


def _code_range_fallback(symbol: str) -> str | None:
    """根据股票代码前缀返回板块分类（保底规则）。"""
    sym = symbol.strip().zfill(6)
    if len(sym) != 6 or not sym.isdigit():
        return None
    for prefix, board in CODE_RANGES:
        if sym.startswith(prefix):
            return board
    return None


def get_industry(symbol: str) -> str | None:
    """返回股票的行业/板块分类。

    优先从 industry_map 中获取真实行业；若不在 map 中，则按代码段回退到板块。
    两者都失败返回 None。
    """
    ind_map = get_industry_map()
    if symbol in ind_map:
        return ind_map[symbol]
    # 回退：代码段规则
    return _code_range_fallback(symbol)


def coverage_stats(symbols: list[str] | None = None) -> dict[str, int | float]:
    """统计行业覆盖情况。

    Args:
        symbols: 待统计的股票列表。为 None 时，使用 industry_map 的 keys 作为全集。

    Returns:
        {
            "total": 总股票数,
            "real_industry": 有真实行业映射的数量,
            "code_fallback": 仅能回退到板块的数量,
            "unclassified": 完全无法分类的数量,
            "coverage": 总覆盖率（含回退）,
            "real_coverage": 仅真实行业的覆盖率,
        }
    """
    ind_map = get_industry_map()

    if symbols is None:
        # 如果没有传入 symbols，用 map keys + 常见代码段模拟一个全集
        symbols = list(ind_map.keys())

    total = len(symbols)
    real = 0
    fallback = 0
    unclassified = 0

    for sym in symbols:
        if sym in ind_map:
            real += 1
        elif _code_range_fallback(sym) is not None:
            fallback += 1
        else:
            unclassified += 1

    return {
        "total": total,
        "real_industry": real,
        "code_fallback": fallback,
        "unclassified": unclassified,
        "coverage": round((real + fallback) / total * 100, 2) if total else 0.0,
        "real_coverage": round(real / total * 100, 2) if total else 0.0,
    }


if __name__ == "__main__":
    # 快速自检
    import sys
    syms = sys.argv[1:] or ["600519", "000001", "300750", "688981", "835368", "999999"]
    print("=== get_industry 测试 ===")
    for s in syms:
        print(f"  {s}: {get_industry(s)}")
    print("\n=== coverage_stats ===")
    stats = coverage_stats(syms)
    for k, v in stats.items():
        print(f"  {k}: {v}")
