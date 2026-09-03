"""数据层：HiThink（主）+ akshare（备）混合数据源。

设计：
- HiThink Finance-API：同花顺官方，稳定、支持批量，需 API Key
- akshare：免费备用源，HiThink 失败时自动回退
- 本地 Parquet 缓存避免重复拉取
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

def _resolve_data_dir() -> Path:
    """数据目录：打包版(exe同目录/data)优先，开发模式用项目根/data。"""
    if getattr(sys, "frozen", False):          # PyInstaller 打包环境
        return Path(sys.executable).resolve().parent / "data"
    return Path(__file__).resolve().parent.parent / "data"


DATA_DIR = _resolve_data_dir()
DAILY_DIR = DATA_DIR / "daily"
META_DIR = DATA_DIR / "meta"


def _ensure_dirs():
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================ HiThink 检测

def _hithink_available() -> bool:
    """HiThink CLI 是否可用（已安装且已配置 API Key）"""
    if shutil.which("hithink-finance") is None:
        return False
    try:
        r = subprocess.run(
            ["hithink-finance", "auth", "status", "--format", "json"],
            capture_output=True, text=True, timeout=10
        )
        if r.returncode != 0:
            return False
        data = json.loads(r.stdout)
        return data.get("ok", False) and data.get("data", {}).get("configured", False)
    except Exception:
        return False


HIT_HINK_AVAILABLE = _hithink_available()


# ================================================================ HiThink 数据获取

def _symbol_to_thscode(symbol: str) -> str:
    """600519 → 600519.SH, 000001 → 000001.SZ"""
    if symbol.endswith(('.SH', '.SZ', '.BJ')):
        return symbol
    if symbol.startswith(('6', '9', '5')):
        return f"{symbol}.SH"
    return f"{symbol}.SZ"


def get_daily_hithink(symbol: str, start: str | None = None,
                      end: str | None = None) -> pd.DataFrame | None:
    """从 HiThink 获取单只股票日线。失败返回 None。"""
    thscode = _symbol_to_thscode(symbol)
    cmd = ["hithink-finance", "market", "history", "--thscode", thscode, "--format", "json"]
    if start:
        cmd += ["--start-ms", str(int(pd.Timestamp(start).timestamp() * 1000))]
    if end:
        cmd += ["--end-ms", str(int(pd.Timestamp(end).timestamp() * 1000))]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return None
        # 解析 NDJSON（首行可能是 type:message，后面才是 JSON）
        for line in r.stdout.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('{"type":"message"'):
                continue
            data = json.loads(line)
            if not data.get("ok"):
                continue
            items = data.get("data", {}).get("items", [])
            if not items:
                continue
            df = pd.DataFrame(items)
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.rename(columns={
                "open": "O", "high": "H", "low": "L", "close": "C",
                "volume": "V", "amount": "amount"
            })
            for c in ("O", "H", "L", "C", "V"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df[["date", "O", "H", "L", "C", "V"]].set_index("date").sort_index()
    except Exception:
        pass
    return None


def get_batch_hithink(symbols: list[str], start: str | None = None,
                      end: str | None = None, max_workers: int = 5,
                      on_progress=None) -> dict[str, pd.DataFrame]:
    """并发批量从 HiThink 下载日线。返回 {symbol: df}。"""
    results = {}

    def _fetch_one(sym):
        df = get_daily_hithink(sym, start, end)
        return sym, df

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, sym): sym for sym in symbols}
        done_count = 0
        total = len(symbols)
        for future in as_completed(futures):
            done_count += 1
            try:
                sym, df = future.result(timeout=60)
                if df is not None and len(df) > 0:
                    results[sym] = df
                if on_progress and done_count % 50 == 0:
                    on_progress(done_count, total)
            except Exception:
                pass
    return results


# ================================================================ akshare 备用

def get_daily_akshare(symbol: str, start: str = "20150101",
                      end: str | None = None, kind: str = "stock") -> pd.DataFrame:
    """akshare 备用数据获取（带 5 次退避重试）"""
    import akshare as ak

    _ensure_dirs()
    tag = f"{start}_{end or 'latest'}"
    f = DAILY_DIR / f"{symbol.replace('.', '_')}_{tag}_akshare.parquet"
    if f.exists():
        return pd.read_parquet(f)

    end_s = end or dt.date.today().strftime("%Y%m%d")
    frames = []
    last_err = None

    fetch_ak = (lambda: ak.fund_etf_hist_em(symbol=symbol, period="daily",
                 start_date=start, end_date=end_s, adjust="qfq"))
    if kind != "etf":
        fetch_ak = (lambda: ak.stock_zh_a_hist(symbol=symbol, period="daily",
                     start_date=start, end_date=end_s, adjust="qfq"))

    for attempt in range(5):
        try:
            raw = fetch_ak()
            break
        except Exception as e:
            last_err = e
            time.sleep(3 + attempt * 4)
    else:
        raise RuntimeError(f"{symbol} akshare failed: {last_err}")

    if raw is None or raw.empty:
        raise RuntimeError(f"no data for {symbol}")

    g = raw.rename(columns={
        "日期": "date", "开盘": "O", "收盘": "C", "最高": "H",
        "最低": "L", "成交量": "V", "成交额": "amount"
    })
    g["date"] = pd.to_datetime(g["date"])
    out = g.set_index("date").sort_index()
    out[["O", "H", "L", "C", "V"]].to_parquet(f)
    return out


# ================================================================ 统一入口

def get_daily(symbol: str, start: str = "20150101",
              end: str | None = None, kind: str = "stock") -> pd.DataFrame:
    """统一入口：HiThink 优先 → akshare 备用"""
    _ensure_dirs()

    # 1. 检查缓存（已有数据直接返回）
    cache_tag = f"{start}_{end or 'latest'}"
    cache_file = DAILY_DIR / f"{symbol.replace('.', '_')}_{cache_tag}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    akshare_cache = DAILY_DIR / f"{symbol.replace('.', '_')}_{cache_tag}_akshare.parquet"
    if akshare_cache.exists():
        return pd.read_parquet(akshare_cache)

    # 2. HiThink（如果可用）
    if HIT_HINK_AVAILABLE:
        df = get_daily_hithink(symbol, start, end)
        if df is not None and len(df) > 0:
            df[["O", "H", "L", "C", "V"]].to_parquet(cache_file)
            return df

    # 3. akshare 备用
    df = get_daily_akshare(symbol, start, end, kind=kind)
    df[["O", "H", "L", "C", "V"]].to_parquet(akshare_cache)
    return df


def batch_get_daily(symbols: list[str], start: str = "20150101",
                    end: str | None = None, max_workers: int = 5,
                    on_progress=None) -> dict[str, pd.DataFrame]:
    """批量获取日线。HiThink 并发 → 失败回退 akshare。"""
    _ensure_dirs()
    results = {}
    need_fetch = []

    # 1. 检查缓存
    cache_tag = f"{start}_{end or 'latest'}"
    for sym in symbols:
        cache_file = DAILY_DIR / f"{sym.replace('.', '_')}_{cache_tag}.parquet"
        akshare_cache = DAILY_DIR / f"{sym.replace('.', '_')}_{cache_tag}_akshare.parquet"
        if cache_file.exists():
            results[sym] = pd.read_parquet(cache_file)
        elif akshare_cache.exists():
            results[sym] = pd.read_parquet(akshare_cache)
        else:
            need_fetch.append(sym)

    if not need_fetch:
        return results

    # 2. HiThink 批量（如果可用）
    if HIT_HINK_AVAILABLE:
        hithink_results = get_batch_hithink(need_fetch, start, end,
                                            max_workers=max_workers,
                                            on_progress=on_progress)
        results.update(hithink_results)
        # 缓存 HiThink 结果
        cache_tag = f"{start}_{end or 'latest'}"
        for sym, df in hithink_results.items():
            cache_file = DAILY_DIR / f"{sym.replace('.', '_')}_{cache_tag}.parquet"
            df[["O", "H", "L", "C", "V"]].to_parquet(cache_file)
        need_fetch = [s for s in need_fetch if s not in hithink_results]

    # 3. akshare 回退（剩余未成功的）
    if need_fetch:
        def _fetch_ak(sym):
            try:
                return sym, get_daily_akshare(sym, start, end)
            except Exception:
                return sym, None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_ak, sym): sym for sym in need_fetch}
            for future in as_completed(futures):
                try:
                    sym, df = future.result(timeout=120)
                    if df is not None and len(df) > 0:
                        results[sym] = df
                except Exception:
                    pass

    return results


# ================================================================ 其他辅助函数（保留兼容性）

def get_index(symbol: str = "000300", start: str = "20180101",
              end: str | None = None) -> pd.DataFrame:
    """基准指数日线（默认沪深300）。"""
    import akshare as ak
    _ensure_dirs()
    end_s = end or dt.date.today().strftime("%Y%m%d")
    f = DATA_DIR / "meta" / f"index_{symbol}_{start}_{end_s}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    raw = None
    last_err = None
    for attempt in range(5):
        try:
            raw = ak.index_zh_a_hist(symbol=symbol, period="daily",
                                     start_date=start, end_date=end_s)
            break
        except Exception as e:
            last_err = e
            time.sleep(3 + attempt * 4)
    if raw is None or raw.empty:
        raise RuntimeError(f"index {symbol} failed: {last_err}")
    g = raw.rename(columns={"日期": "date", "收盘": "C"})
    g["date"] = pd.to_datetime(g["date"])
    out = g.set_index("date").sort_index()
    out[["C"]].to_parquet(f)
    return out


def get_fundamental(symbol: str) -> dict | None:
    """保留兼容：走 akshare 财务摘要"""
    import akshare as ak
    _ensure_dirs()
    f = META_DIR / "fundamental" / f"{symbol}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    try:
        fa = ak.stock_financial_abstract_ths(symbol=symbol, indicator="按报告期")
        if fa is not None and not fa.empty:
            row = fa.iloc[-1]
            out = {
                "report_date": str(row.get("报告期", "")),
                "net_profit": _parse_num(row.get("净利润")),
                "roe": _parse_num(row.get("净资产收益率")),
                "debt_ratio": _parse_num(row.get("资产负债率")),
                "gross_margin": _parse_num(row.get("销售毛利率")),
                "net_margin": _parse_num(row.get("销售净利率")),
                "revenue_yoy": _parse_num(row.get("营业总收入同比增长率")),
                "profit_yoy": _parse_num(row.get("净利润同比增长率")),
                "deduct_np": _parse_num(row.get("扣非净利润")),
            }
            f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            time.sleep(0.4)
            return out
    except Exception as e:
        return {"_error": str(e)[:80]}
    return {}


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


def check_redlines(symbol: str, name: str = "",
                   fundamental: dict | None = None,
                   listed_days: int | None = None) -> tuple[bool, list[str]]:
    """基本面红线过滤。返回 (是否通过, 触发的红线列表)。"""
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
        hits.append("R3x:deduct_loss")
    roe = fu.get("roe")
    if roe is not None and roe < 5:
        hits.append(f"R4:roe{roe}")
    dr = fu.get("debt_ratio")
    is_financial = (dr is not None and dr > 75)
    if not is_financial and dr is not None and dr > 70:
        hits.append(f"R5:debt{dr}")
    return (not hits, hits)


def get_all_symbols(include_etf: bool = False, exclude_st_bj: bool = True) -> pd.DataFrame:
    """获取 A 股股票列表。"""
    import akshare as ak
    _ensure_dirs()
    today = dt.date.today().strftime("%Y%m%d")
    cache = META_DIR / f"stock_list_{today}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    for attempt in range(3):
        try:
            df = ak.stock_info_a_code_name()
            df["symbol"] = df["code"].astype(str).str.zfill(6)
            df["name"] = df["name"].astype(str)
            if exclude_st_bj:
                df = df[~df["name"].str.contains("ST|退|BJ", na=False)]
            df = df[["symbol", "name"]].reset_index(drop=True)
            df["kind"] = "stock"
            df.to_parquet(cache)
            return df
        except Exception as e:
            time.sleep(5 + attempt * 5)
    raise RuntimeError(f"stock list fetch failed: {e}")


def today_str() -> str:
    return dt.date.today().strftime("%Y-%m-%d")


def symbol_names() -> dict[str, str]:
    """返回 {symbol: name} 映射。"""
    cache_files = sorted(META_DIR.glob("stock_list_*.parquet"), reverse=True)
    if not cache_files:
        return {}
    try:
        df = pd.read_parquet(cache_files[0])
        return dict(zip(df["symbol"], df.get("name", "")))
    except Exception:
        return {}
