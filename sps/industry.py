"""行业归属映射（新浪 stock_classify_sina，带本地缓存）。

该接口一次调用返回全市场股票及其行业分类（class 列），约6分钟拉完。
缓存7天；失败时回退旧缓存。供筛选结果的行业聚合与行业RPS使用。

用法：
    from sps.industry import get_industry_map
    ind_map = get_industry_map()   # {symbol: 行业名}
"""
from __future__ import annotations

import json
import time

from sps.paths import DATA_DIR

IND_FILE = DATA_DIR / "meta" / "industry_map.json"


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
