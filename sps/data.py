"""数据层：AKShare 拉取 + 本地 Parquet 缓存。

设计要点（对应规格书 0.1.1）：
- 前复权 OHLC 用于形态几何；成交量用原始量
- 股票列表按日期快照保存，避免幸存者偏差
- 全部落地为 parquet，重复运行不重拉
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
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
    for d in (DATA_DIR, DAILY_DIR, META_DIR):
        d.mkdir(parents=True, exist_ok=True)


def today_str() -> str:
    return dt.date.today().isoformat()


def get_all_symbols(trade_date: str | None = None,
                    include_etf: bool = True,
                    exclude_st_bj: bool = True) -> pd.DataFrame:
    """股票+ETF 列表快照（含代码/名称），缓存按日。

    exclude_st_bj=True 时排除：ST/*ST/退市整理、北交所、新三板。
    include_etf=True 时附带场内 ETF（fund_etf_spot_em）。
    """
    _ensure_dirs()
    d = trade_date or today_str()
    f = META_DIR / f"stock_list_{d}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    import akshare as ak
    df = None
    last_err = None
    for attempt in range(4):
        try:
            df = ak.stock_info_a_code_name()
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(5 + attempt * 5)
    if df is None:
        # 回退：用最近一次缓存列表
        old = sorted(META_DIR.glob("stock_list_*.parquet"))
        if old:
            print(f"[warn] list fetch failed ({last_err}), using {old[-1].name}")
            return pd.read_parquet(old[-1])
        raise RuntimeError(f"symbol list failed: {last_err}")
    df = df.rename(columns={"code": "symbol", "name": "name"})
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["market"] = df["symbol"].str[:2].map(
        {"60": "SH", "68": "STAR", "00": "SZ", "30": "ChiNext",
         "43": "BJ", "83": "BJ", "87": "BJ", "92": "BJ"}
    ).fillna("OTHER")
    df["kind"] = "stock"
    # ---- 排除 ST / 退市 / 北交所 / 新三板 ----
    if exclude_st_bj:
        name_up = df["name"].str.upper()
        mask_st = name_up.str.contains("ST") | name_up.str.contains("退")
        mask_bj = (df["market"] == "BJ") | (df["market"] == "OTHER")
        df = df[~mask_st & ~mask_bj]
    # ---- ETF 场内基金 ----
    if include_etf:
        etf = None
        for attempt in range(3):
            try:
                etf = ak.fund_etf_spot_em()
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(4 + attempt * 4)
        if etf is not None and not etf.empty:
            cols = {c: c for c in etf.columns}
            code_col = next(c for c in etf.columns if "代码" in c)
            name_col = next(c for c in etf.columns if "名称" in c)
            e2 = pd.DataFrame({
                "symbol": etf[code_col].astype(str).str.zfill(6),
                "name": etf[name_col].astype(str),
            })
            e2["market"] = e2["symbol"].str[:1].map({"5": "SH", "1": "SZ"}).fillna("OTHER")
            e2["kind"] = "etf"
            # 排除货币/债券类可选——默认全保留，用户可自行过滤
            df = pd.concat([df, e2], ignore_index=True)
        else:
            print(f"[warn] ETF list fetch failed ({last_err}), stocks only")
    df.to_parquet(f, index=False)
    return df


def get_daily(symbol: str, start: str = "20150101", end: str | None = None,
              adjust: str = "qfq", kind: str = "stock") -> pd.DataFrame:
    """单只股票/ETF日线。返回列: date,O,H,L,C,V(原始),amount。

    kind='etf' 时走东财基金接口（前复权）。
    """
    import akshare as ak
    _ensure_dirs()
    tag = f"{adjust}_{start}_{end or 'latest'}"
    f = DAILY_DIR / f"{symbol.replace('.', '_')}_{tag}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    end_s = end or today_str().replace("-", "")
    frames = []
    # 东财 stock_zh_a_hist 的"成交量"本身就是原始成交量（股），
    # 前复权只改价格不改量 → 一次调用同时拿到 qfq 价格 + 原始量。
    adj = adjust or ""
    raw = None
    last_err = None
    fetch = (lambda: ak.fund_etf_hist_em(symbol=symbol, period="daily",
             start_date=start, end_date=end_s, adjust=adj))
    if kind != "etf":
        fetch = (lambda: ak.stock_zh_a_hist(symbol=symbol, period="daily",
                 start_date=start, end_date=end_s, adjust=adj))
    for attempt in range(5):  # 连接不稳：5次退避重试
        try:
            raw = fetch()
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(3 + attempt * 4)
    if raw is None or getattr(raw, "empty", True):
        # ---- 备用源：新浪财经（无复权，但量价完整）----
        try:
            raw2 = None
            if kind == "etf":
                sina_sym = ("sh" if symbol.startswith("5") else "sz") + symbol
                raw2 = ak.fund_etf_hist_sina(symbol=sina_sym)
                if raw2 is not None and not raw2.empty and start:
                    raw2["date"] = pd.to_datetime(raw2["date"])
                    raw2 = raw2[raw2["date"] >= pd.to_datetime(start)]
            else:
                prefix = "sh" if symbol.startswith(("6", "9", "5")) else "sz"
                raw2 = ak.stock_zh_a_daily(symbol=prefix + symbol, start_date=start,
                                           end_date=end_s)
            if raw2 is not None and not raw2.empty:
                cols_map = {"open": "O", "close": "C", "high": "H", "low": "L",
                            "volume": "V", "amount": "amount"}
                g = raw2.rename(columns=cols_map)
                g["date"] = pd.to_datetime(g["date"])
                keep = ["date"] + [c for c in ("O","H","L","C","V","amount") if c in g.columns]
                g = g[keep].dropna(subset=["C"])
                out = g.set_index("date").sort_index()
                # 补齐缺列（sina ETF 无 amount 时置0）
                for c in ("O","H","L","C","V"):
                    if c not in out.columns:
                        out[c] = out["C"] if c != "V" else 0
                out = out[["O","H","L","C","V"]] if "amount" not in out.columns else out
                out.to_parquet(f)
                return out
        except Exception as e:  # noqa: BLE001
            last_err = e
    if raw is None:
        raise RuntimeError(f"{symbol} hist failed: {last_err}")
    if raw is not None and not raw.empty:
        g = raw.rename(columns={
            "日期": "date", "开盘": "O", "收盘": "C", "最高": "H",
            "最低": "L", "成交量": "V", "成交额": "amount"})[
            ["date", "O", "H", "L", "C", "V", "amount"]]
        g["date"] = pd.to_datetime(g["date"])
        out = g.set_index("date").sort_index()
        out.to_parquet(f)
        return out
    raise RuntimeError(f"no data for {symbol}")


def get_index(symbol: str = "000300", start: str = "20180101",
              end: str | None = None) -> pd.DataFrame:
    """基准指数日线（默认沪深300），列: C。带重试与缓存。"""
    import akshare as ak
    _ensure_dirs()
    end_s = end or today_str().replace("-", "")
    f = DAILY_DIR / f"index_{symbol}_{start}_{end_s}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    raw = None
    last_err = None
    for attempt in range(5):
        try:
            raw = ak.index_zh_a_hist(symbol=symbol, period="daily",
                                     start_date=start, end_date=end_s)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(3 + attempt * 4)
    if raw is None or getattr(raw, "empty", True):
        # 备用：新浪指数源
        sina_sym = {"000300": "sh000300", "000001": "sh000001"}.get(symbol)
        if sina_sym:
            try:
                raw2 = ak.stock_zh_index_daily(symbol=sina_sym)
                if raw2 is not None and not raw2.empty:
                    g2 = raw2.rename(columns={"date": "date", "close": "C"})
                    g2["date"] = pd.to_datetime(g2["date"])
                    out = g2[["date", "C"]].set_index("date").sort_index()
                    out = out[out.index >= pd.Timestamp(start)]
                    out.to_parquet(f)
                    return out
            except Exception as e:  # noqa: BLE001
                last_err = e
    if raw is None or getattr(raw, "empty", True):
        raise RuntimeError(f"index {symbol} failed: {last_err}")
    g = raw.rename(columns={"日期": "date", "收盘": "C"})[["date", "C"]]
    g["date"] = pd.to_datetime(g["date"])
    out = g.set_index("date").sort_index()
    out.to_parquet(f)
    return out


def load_universe(symbols: list[str] | None = None, start: str = "20150101",
                  max_stocks: int | None = None) -> dict[str, pd.DataFrame]:
    """批量加载全市场日线到内存字典。max_stocks 用于小规模试跑。"""
    lst = get_all_symbols()
    syms = symbols or lst["symbol"].tolist()
    if max_stocks:
        syms = syms[:max_stocks]
    data = {}
    failed = []
    for i, s in enumerate(syms):
        try:
            data[s] = get_daily(s, start=start)
        except Exception as e:  # noqa: BLE001
            failed.append((s, str(e)[:80]))
        if i % 50 == 0:
            print(f"  loaded {i}/{len(syms)}")
    if failed:
        (DATA_DIR / "failed.json").write_text(json.dumps(failed, ensure_ascii=False, indent=1))
    return data
