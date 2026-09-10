"""数据层：HiThink（主）+ akshare（备）混合数据源。

设计：
- HiThink Finance-API：同花顺官方，稳定、支持批量，需 API Key
- akshare：免费备用源，HiThink 失败时自动回退
- 本地 Parquet 缓存避免重复拉取
"""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from sps.paths import DATA_DIR, resolve_data_dir as _resolve_data_dir

DAILY_DIR = DATA_DIR / "daily"
META_DIR = DATA_DIR / "meta"

# 前复权拼接污染防护：重叠窗口收盘价相对偏差超过该阈值即判定
# 旧缓存复权基准过期，强制全量重拉该票
QFQ_MISMATCH_TOL = 0.005


def _ensure_dirs():
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)


_no_proxy_lock = threading.Lock()
_no_proxy_depth = 0
_no_proxy_saved: dict | None = None


@contextlib.contextmanager
def no_proxy():
    """临时屏蔽系统代理（akshare 等国内数据接口不走代理）。

    数据源均为国内站点；用户机器上的系统代理（常为国际线路）经常
    拒绝/断开这些请求，导致行情更新整体失败。
    用引用计数防止并发嵌套时把环境改坏（泄漏 NO_PROXY）：
    只有最外层进入时保存/设置、最外层退出时还原。
    已知限制：同一进程内与数据刷新并发的国际 API 调用（如走代理的
    OpenAI）在该窗口内也会绕过代理；国内大模型（DeepSeek 等）不受影响。
    """
    global _no_proxy_depth, _no_proxy_saved
    proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                  "http_proxy", "https_proxy", "all_proxy")
    with _no_proxy_lock:
        if _no_proxy_depth == 0:
            saved = {k: os.environ.pop(k) for k in proxy_keys if k in os.environ}
            saved["NO_PROXY"] = os.environ.get("NO_PROXY")
            _no_proxy_saved = saved
            os.environ["NO_PROXY"] = "*"
        _no_proxy_depth += 1
    try:
        yield
    finally:
        with _no_proxy_lock:
            _no_proxy_depth -= 1
            if _no_proxy_depth == 0 and _no_proxy_saved is not None:
                os.environ.pop("NO_PROXY", None)
                os.environ.update(_no_proxy_saved)
                _no_proxy_saved = None


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """原子落盘：先写临时文件再替换，避免并发读者读到半截 parquet。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, path)


# ================================================================ HiThink 检测

def _hithink_available() -> bool:
    """HiThink CLI 是否可用（已安装且已配置 API Key）"""
    if shutil.which("hithink-finance") is None:
        return False
    try:
        r = subprocess.run(
            "hithink-finance auth status --format json",
            capture_output=True, text=True, timeout=10, shell=True
        )
        if r.returncode != 0:
            return False
        # 解析 NDJSON（首行可能是 type:message）
        for line in r.stdout.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('{"type":"message"'):
                continue
            data = json.loads(line)
            return data.get("ok", False) and data.get("data", {}).get("configured", False)
        return False
    except Exception:
        return False


# 兼容保留的常量：不再在 import 时探测（那会拖慢冷启动并被并行扫描
# 的每个子进程重复执行）。实际可用性一律调用 hithink_available() 惰性探测。
HIT_HINK_AVAILABLE = False
LAST_BATCH_STATS = {
    "requested": 0,
    "hithink_refreshed": 0,
    "akshare_fallback": 0,
    "stale_cache_retained": 0,
}


def hithink_available() -> bool:
    """Re-check at refresh time so newly saved credentials take effect."""
    return _hithink_available()


# ================================================================ HiThink 数据获取

def _symbol_to_thscode(symbol: str) -> str:
    """600519 → 600519.SH, 000001 → 000001.SZ"""
    if symbol.endswith(('.SH', '.SZ', '.BJ')):
        return symbol
    if symbol.startswith(('6', '9', '5')):
        return f"{symbol}.SH"
    return f"{symbol}.SZ"


def _parse_hithink_history(payload: dict) -> pd.DataFrame | None:
    """Normalize the current HiThink CLI envelope into SPS OHLCV columns."""
    if not payload.get("ok"):
        return None
    data = payload.get("data") or {}
    items = data.get("item") or data.get("items") or []
    if not items:
        return None
    df = pd.DataFrame(items)
    date_column = "date_ms" if "date_ms" in df.columns else "timestamp"
    if date_column not in df.columns:
        return None
    df["date"] = (
        pd.to_datetime(df[date_column], unit="ms", utc=True)
        .dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    )
    df = df.rename(columns={
        "open_price": "O", "high_price": "H", "low_price": "L",
        "close_price": "C", "open": "O", "high": "H", "low": "L",
        "close": "C", "volume": "V",
    })
    required = ["O", "H", "L", "C", "V"]
    if any(column not in df.columns for column in required):
        return None
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    # HiThink uses shares; existing SPS/AkShare caches use 100-share lots.
    df["V"] = df["V"] / 100.0
    return df[["date", *required]].dropna().set_index("date").sort_index()


def _timestamp_ms(value: str) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Shanghai")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return str(int(timestamp.timestamp() * 1000))


def get_daily_hithink(symbol: str, start: str | None = None,
                      end: str | None = None) -> pd.DataFrame | None:
    """从 HiThink 获取单只股票日线。失败返回 None。"""
    thscode = _symbol_to_thscode(symbol)
    start_value = start or "20150101"
    end_value = end or dt.date.today().strftime("%Y%m%d")
    cmd = ["hithink-finance", "market", "history", "--thscode", thscode,
           "--start-ms", _timestamp_ms(start_value),
           "--end-ms", _timestamp_ms(end_value),
           "--adjust", "forward", "--format", "json"]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, shell=True)
        if r.returncode != 0:
            return None
        # 解析 NDJSON（首行可能是 type:message，后面才是 JSON）
        for line in r.stdout.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('{"type":"message"'):
                continue
            frame = _parse_hithink_history(json.loads(line))
            if frame is not None:
                return frame
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


# ================================================================ 快照快路径（每日增量）

def _trade_calendar() -> list[str]:
    """A股交易日历（HiThink market calendar），本地缓存 1 天。返回 ['YYYYMMDD', ...]。"""
    f = META_DIR / "trade_calendar.json"
    if f.exists():
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("dates") and time.time() - d.get("_ts", 0) < 1 * 86400:  # 1天缓存
                return d["dates"]
        except Exception:
            pass
    try:
        r = subprocess.run(["hithink-finance", "market", "calendar", "--format", "json"],
                           capture_output=True, text=True, timeout=60, shell=True)
        if r.returncode != 0:
            return []
        payload = None
        for line in r.stdout.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith('{"type":"message"'):
                continue
            payload = json.loads(line)
            break
        items = (payload or {}).get("data", {}).get("item") or []
        # 归一化为 YYYYMMDD，防御接口返回 "YYYY-MM-DD" 等格式
        dates = sorted(re.sub(r"\D", "", str(i["date"]))
                       for i in items
                       if i.get("date") and len(re.sub(r"\D", "", str(i["date"]))) == 8)
        if dates:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"_ts": time.time(), "dates": dates}),
                         encoding="utf-8")
        return dates
    except Exception:
        return []


def _latest_closed_trade_day(calendar: list[str]) -> str | None:
    """最近一个已收盘的交易日（YYYYMMDD）。15:05 前视为当日未收盘。"""
    if not calendar:
        return None
    today = dt.date.today().strftime("%Y%m%d")
    lt = time.localtime()
    closed_today = lt.tm_hour * 60 + lt.tm_min >= 15 * 60 + 5
    if today in calendar and closed_today:
        return today
    prev = [d for d in calendar if d < today]
    return prev[-1] if prev else None


def get_market_snapshot(max_pages: int = 10, page_size: int = 1000):
    """HiThink 全市场行情快照：几次子进程调用拿全市场最新日K。

    这是每日增量更新的快路径——替代"每票一个子进程"的逐票拉取。
    返回 ({symbol: 单行DataFrame(O,H,L,C,V)}, trade_day 'YYYYMMDD')，
    任一页失败返回 None（调用方回退慢路径）。
    """
    calendar = _trade_calendar()
    trade_day = _latest_closed_trade_day(calendar)
    if trade_day is None:
        return None
    # 交易时段（含集合竞价与收盘竞价）不使用快照：此时返回的是当日
    # 盘中半根K线，而"最近已收盘交易日"是前一交易日——直接合并会把
    # 昨日完成K线覆盖成盘中数据。回退慢路径（其日期标签正确）。
    lt = time.localtime()
    minutes = lt.tm_hour * 60 + lt.tm_min
    if time.strftime("%Y%m%d") in calendar and (9 * 60 + 15) <= minutes < (15 * 60 + 5):
        return None
    ts = pd.Timestamp(trade_day)
    bars: dict[str, pd.DataFrame] = {}
    offset, total = 0, None
    for _ in range(max_pages):
        cmd = ["hithink-finance", "market", "snapshot",
               "--limit", str(page_size), "--offset", str(offset),
               "--format", "json"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=60, shell=True)
        except Exception:
            return None
        if r.returncode != 0:
            return None
        payload = None
        for line in r.stdout.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith('{"type":"message"'):
                continue
            payload = json.loads(line)
            break
        if not payload or not payload.get("ok"):
            return None
        data = payload.get("data") or {}
        items = data.get("item") or []
        total = data.get("total")
        for it in items:
            sym = str(it.get("ticker") or "")[:6]
            o, h, l = it.get("open_price"), it.get("high_price"), it.get("low_price")
            c, v = it.get("last_price"), it.get("volume")
            if not sym or not all(isinstance(x, (int, float))
                                  for x in (o, h, l, c, v)):
                continue
            if o <= 0 or v <= 0:
                continue   # 停牌/无成交：保留旧缓存即可
            # HiThink 成交量为股数，与既有缓存（手）统一 /100；
            # prev_price（昨收，除权日为调整后昨收）用于复权基准变化检测
            prev = it.get("prev_price")
            bars[sym] = pd.DataFrame(
                {"O": [float(o)], "H": [float(h)], "L": [float(l)],
                 "C": [float(c)], "V": [float(v) / 100.0],
                 "PREV": [float(prev) if isinstance(prev, (int, float)) else 0.0]},
                index=[ts])
        offset += max(len(items), 1)   # 按实际返回条数步进，防上游截断跳行（A12）
        if not items or (total is not None and offset >= total):
            break
    if not bars:
        return None
    return bars, trade_day


# ================================================================ akshare 备用

def get_daily_akshare(symbol: str, start: str = "20150101",
                      end: str | None = None, kind: str = "stock",
                      refresh: bool = False, save: bool = True) -> pd.DataFrame:
    """akshare 备用数据获取（带 5 次退避重试）"""
    import akshare as ak

    _ensure_dirs()
    tag = f"{start}_{end or 'latest'}"
    f = DAILY_DIR / f"{symbol.replace('.', '_')}_{tag}_akshare.parquet"
    if f.exists() and not refresh:
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
            with no_proxy():
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
    if save:
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
    if hithink_available():
        df = get_daily_hithink(symbol, start, end)
        if df is not None and len(df) > 0:
            df[["O", "H", "L", "C", "V"]].to_parquet(cache_file)
            return df

    # 3. akshare 备用
    df = get_daily_akshare(symbol, start, end, kind=kind)
    df[["O", "H", "L", "C", "V"]].to_parquet(akshare_cache)
    return df


def _cache_last_date(path: Path) -> pd.Timestamp | None:
    """Read the last date from a parquet file without loading full data.
    Uses pyarrow metadata when possible, falls back to reading index.
    """
    try:
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(path)
        metadata = parquet.schema_arrow.metadata or {}
        import json
        pandas_meta = json.loads(metadata.get(b"pandas", b"{}").decode("utf-8"))
        index_cols = [v for v in pandas_meta.get("index_columns", []) if isinstance(v, str)]
        candidates = index_cols + ["date", "__index_level_0__"]
        for field in candidates:
            if field not in parquet.schema.names:
                continue
            col_idx = parquet.schema.names.index(field)
            maxima = []
            for grp_idx in range(parquet.metadata.num_row_groups):
                stats = parquet.metadata.row_group(grp_idx).column(col_idx).statistics
                if stats is not None and stats.has_min_max:
                    maxima.append(stats.max)
            if maxima:
                return pd.Timestamp(max(maxima))
    except Exception:
        pass
    # Fallback: read the file
    try:
        df = pd.read_parquet(path, columns=[])
        if len(df) > 0:
            return pd.Timestamp(df.index[-1])
    except Exception:
        pass
    return None


def batch_get_daily(symbols: list[str], start: str = "20150101",
                    end: str | None = None, max_workers: int = 5,
                    on_progress=None, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Load cache, or refresh it incrementally via HiThink then AkShare.

    Incremental logic:
    - If cache exists and last date is today/yesterday → skip entirely
    - If cache exists but stale → fetch only from last_date+1 to today
    - If no cache → full fetch from start date
    """
    global LAST_BATCH_STATS
    _ensure_dirs()
    results: dict[str, pd.DataFrame] = {}
    cached: dict[str, pd.DataFrame] = {}
    need_fetch = []
    cache_tag = f"{start}_{end or 'latest'}"
    today = pd.Timestamp.today().normalize()
    for sym in symbols:
        cache_file = DAILY_DIR / f"{sym.replace('.', '_')}_{cache_tag}.parquet"
        akshare_cache = DAILY_DIR / f"{sym.replace('.', '_')}_{cache_tag}_akshare.parquet"
        path = None
        if cache_file.exists():
            path = cache_file
        elif akshare_cache.exists():
            path = akshare_cache
        if path is not None:
            # Check if cache is up-to-date
            last_dt = _cache_last_date(path)
            if last_dt is not None and not refresh:
                age_days = (today - last_dt.normalize()).days
                if age_days <= 1:
                    # Cache is fresh (today or yesterday) → skip
                    cached[sym] = pd.read_parquet(path)
                    results[sym] = cached[sym]
                    continue
                else:
                    # Cache is stale → will fetch incrementally
                    need_fetch.append(sym)
            else:
                need_fetch.append(sym)
        else:
            need_fetch.append(sym)
    if not need_fetch:
        n_cached = len(results)
        if n_cached > 0:
            print(f"  {n_cached} 只股票缓存已是最新，无需下载")
        LAST_BATCH_STATS = {"requested": len(symbols), "hithink_refreshed": 0,
                            "akshare_fallback": 0, "stale_cache_retained": n_cached,
                            "qfq_refetched": 0, "snapshot_appended": 0,
                            "stale_sidecars_removed": 0}
        return results

    def _merge(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
        def _clean(df: pd.DataFrame) -> pd.DataFrame:
            """剔除供应商脏行（0/负价格、缺失 OHLC），避免污染指标与检测器。"""
            df = df.loc[:, ["O", "H", "L", "C", "V"]]
            return df[(df["C"] > 0) & (df["O"] > 0)].dropna(
                subset=["O", "H", "L", "C"])

        if old is None or old.empty:
            return _clean(new).sort_index()
        combined = pd.concat([_clean(old), _clean(new)])
        return combined.loc[~combined.index.duplicated(keep="last")].sort_index()

    def _overlap_mismatch(old: pd.DataFrame | None, new: pd.DataFrame) -> bool:
        """前复权基准变化检测：重叠区间收盘价偏差超阈值说明旧缓存已过期。

        qfq 价格在除权后会被供应商整条重写；旧缓存是旧基准，
        直接与新数据拼接会在除权日之前留下失真价格。
        """
        if old is None or old.empty or new is None or new.empty:
            return False
        common = old.index.intersection(new.index)
        if len(common) < 3:
            return False
        a = old.loc[common, "C"].astype(float)
        b = new.loc[common, "C"].astype(float).abs()
        mask = b > 0
        if not bool(mask.any()):
            return False
        dev = (a[mask] - b[mask]).abs() / b[mask]
        return bool((dev > QFQ_MISMATCH_TOL).any())

    fetch_starts = {}
    for sym in need_fetch:
        old = cached.get(sym)
        if old is not None and not old.empty:
            fetch_starts[sym] = (
                pd.Timestamp(old.index.max()) - pd.Timedelta(days=14)
            ).strftime("%Y%m%d")
        else:
            fetch_starts[sym] = start

    qfq_refetched = 0
    snapshot_appended = 0
    hithink_results: dict[str, pd.DataFrame] = {}
    if hithink_available():
        # ---- 快路径：一次快照拉全市场最新日K，缓存仅落后一天的票直接合并 ----
        # 逐票 history 每票起一个子进程，全市场 5000+ 次是更新慢的主因；
        # 快照只需几次分页调用，日常增量从几十分钟降到秒级。
        snap = get_market_snapshot()
        if snap is not None:
            bars, trade_day = snap
            cal = _trade_calendar()
            prev_day = None
            if trade_day in cal:
                i = cal.index(trade_day)
                prev_day = cal[i - 1] if i > 0 else None
            still_slow = []
            for fi, sym in enumerate(need_fetch):
                old = cached.get(sym)
                bar = bars.get(sym)
                if old is None or old.empty or bar is None:
                    still_slow.append(sym)   # 无缓存/停牌缺bar：走慢路径
                    continue
                cache_max = pd.Timestamp(old.index.max()).strftime("%Y%m%d")
                if cache_max == trade_day:
                    # 同日=用快照定稿/修正当日bar（同基准，安全）
                    merged = _merge(old, bar)
                    results[sym] = merged
                    _atomic_write_parquet(
                        merged, DAILY_DIR / f"{sym.replace('.', '_')}_{cache_tag}.parquet")
                    snapshot_appended += 1
                elif cache_max == prev_day:
                    # 追加最新收盘：先校验昨收是否对得上（除权日昨收会被
                    # 供应商按新基准改写，对不上=旧缓存基准过期 → 慢路径）
                    prev = float(bar["PREV"].iloc[-1]) if "PREV" in bar.columns else 0.0
                    old_last = float(old["C"].iloc[-1])
                    if (prev > 0 and old_last > 0
                            and abs(prev / old_last - 1) > QFQ_MISMATCH_TOL):
                        still_slow.append(sym)
                        continue
                    merged = _merge(old, bar)
                    results[sym] = merged
                    _atomic_write_parquet(
                        merged, DAILY_DIR / f"{sym.replace('.', '_')}_{cache_tag}.parquet")
                    snapshot_appended += 1
                else:
                    still_slow.append(sym)   # 缺口或落后过多：补历史窗口
                if on_progress and fi % 500 == 0:
                    on_progress(fi, len(need_fetch))
            need_fetch = still_slow
        # ---- 慢路径：快照覆盖不到的票逐票补历史 ----
        if need_fetch:
            groups: dict[str, list[str]] = {}
            for sym in need_fetch:
                groups.setdefault(fetch_starts[sym], []).append(sym)
            for group_start, group_symbols in groups.items():
                hithink_results.update(get_batch_hithink(
                    group_symbols, group_start, end, max_workers=max_workers,
                    on_progress=on_progress,
                ))
        # 除权后整条历史被重写：重叠窗口对不上的票全量重拉，禁止拼接
        mismatched = [sym for sym, fresh in hithink_results.items()
                      if _overlap_mismatch(cached.get(sym), fresh)]
        if mismatched:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(get_daily_hithink, sym, start, end): sym
                           for sym in mismatched}
                for future in as_completed(futures):
                    sym = futures[future]
                    try:
                        full = future.result(timeout=120)
                    except Exception:
                        full = None
                    if full is not None and len(full) > 0:
                        hithink_results[sym] = full
                        qfq_refetched += 1
        for sym, fresh in hithink_results.items():
            merged = _merge(cached.get(sym), fresh)
            results[sym] = merged
            _atomic_write_parquet(
                merged, DAILY_DIR / f"{sym.replace('.', '_')}_{cache_tag}.parquet")
        need_fetch = [sym for sym in need_fetch if sym not in hithink_results]

    akshare_count = 0
    stale_retained = 0
    if need_fetch:
        def _fetch_ak(sym):
            try:
                df = get_daily_akshare(
                    sym, fetch_starts[sym], end, refresh=True, save=False
                )
                if _overlap_mismatch(cached.get(sym), df):
                    full = get_daily_akshare(
                        sym, start, end, refresh=True, save=False
                    )
                    if full is not None and len(full) > 0:
                        return sym, full, True
                return sym, df, False
            except Exception:
                return sym, None, False

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_ak, sym): sym for sym in need_fetch}
            for future in as_completed(futures):
                try:
                    sym, df, refetched = future.result(timeout=120)
                    qfq_refetched += int(refetched)
                    if df is not None and len(df) > 0:
                        merged = _merge(cached.get(sym), df)
                        results[sym] = merged
                        _atomic_write_parquet(
                            merged,
                            DAILY_DIR / f"{sym.replace('.', '_')}_{cache_tag}_akshare.parquet")
                        akshare_count += 1
                    elif sym in cached:
                        results[sym] = cached[sym]
                        stale_retained += 1
                except Exception:
                    sym = futures[future]
                    if sym in cached:
                        results[sym] = cached[sym]
                        stale_retained += 1

    # 清理过期 akshare 残留：主缓存数据更新时，旧的备用文件只会被
    # 按 mtime 选文件的读取方误读（K线显示旧日期），直接删除。
    # 缓存可再生，不影响真实数据；akshare 仍是唯一来源的票不受影响。
    removed_sidecars = 0
    try:
        from sps.health import _parquet_last_date
        for sym in list(results):
            main_file = DAILY_DIR / f"{sym.replace('.', '_')}_{cache_tag}.parquet"
            if not main_file.exists():
                continue
            main_last = _parquet_last_date(main_file)
            side = DAILY_DIR / f"{sym.replace('.', '_')}_{cache_tag}_akshare.parquet"
            for side in [side] if side.exists() else []:
                try:
                    if _parquet_last_date(side) <= main_last:
                        side.unlink()
                        removed_sidecars += 1
                except (OSError, ValueError):
                    continue
    except Exception as e:  # noqa: BLE001
        print(f"[warn] akshare 残留清理失败(不影响数据): {e}")

    LAST_BATCH_STATS = {
        "requested": len(symbols),
        "hithink_refreshed": len(hithink_results),
        "akshare_fallback": akshare_count,
        "stale_cache_retained": stale_retained,
        "qfq_refetched": qfq_refetched,
        "snapshot_appended": snapshot_appended,
        "stale_sidecars_removed": removed_sidecars,
    }
    return results


# ================================================================ 其他辅助函数

def get_index(symbol: str = "000300", start: str = "20180101",
              end: str | None = None) -> pd.DataFrame:
    """基准指数日线（默认沪深300）。

    链路：HiThink 指数历史（不走系统代理）→ akshare（临时屏蔽代理）
    → 最近旧缓存兜底。指数是环境评分的输入，宁可旧一天也不能让
    整个扫描因代理抖动而失败。
    """
    import akshare as ak
    _ensure_dirs()
    end_s = end or dt.date.today().strftime("%Y%m%d")
    f = DATA_DIR / "meta" / f"index_{symbol}_{start}_{end_s}.parquet"
    if f.exists():
        return pd.read_parquet(f)

    # 1) HiThink 指数历史（不走系统代理，最稳）。
    #    注意：上游对超长区间返回空（实测 2018 年起为 0 条，约 3 年内正常），
    #    因此按请求起始 → 3年前 → 1年前 渐进重试；MA200 口径 1 年足够。
    thscode = (f"{symbol}.SZ" if symbol.startswith("399") else f"{symbol}.SH") \
        if not symbol.endswith((".SH", ".SZ", ".BJ")) else symbol
    this_year = dt.date.today().year
    starts = [start, f"{this_year - 3}0101", f"{this_year - 1}0101"]
    for si, start_try in enumerate(starts):
        try:
            r = subprocess.run(
                ["hithink-finance", "index", "history", "--thscode", thscode,
                 "--start-ms", _timestamp_ms(start_try),
                 "--end-ms", _timestamp_ms(end_s),
                 "--format", "json"],
                capture_output=True, text=True, timeout=60, shell=True)
            if r.returncode != 0:
                continue
            payload = None
            for line in r.stdout.strip().split("\n"):
                line = line.strip()
                if not line or line.startswith('{"type":"message"'):
                    continue
                payload = json.loads(line)
                break
            items = (payload or {}).get("data", {}).get("item") or []
            if not items:
                continue
            df = pd.DataFrame(items)
            df["date"] = (pd.to_datetime(df["date_ms"], unit="ms", utc=True)
                          .dt.tz_convert("Asia/Shanghai").dt.tz_localize(None))
            df["C"] = pd.to_numeric(df["close_price"], errors="coerce")
            out = df.dropna(subset=["C"]).set_index("date").sort_index()[["C"]]
            if len(out):
                if si > 0:
                    print(f"[warn] index {symbol}: 起始 {start} 区间超上游上限，"
                          f"改用 {start_try} 起（{len(out)} 根，MA200 口径足够）")
                out.to_parquet(f)
                _cleanup_old_index_files(symbol, keep=3)
                return out
        except Exception as e:  # noqa: BLE001
            if si == len(starts) - 1:
                print(f"[warn] HiThink index {symbol} failed: {e}")

    # 2) akshare 备用（国内接口，屏蔽系统代理）
    raw = None
    last_err = None
    with no_proxy():
        for attempt in range(5):
            try:
                raw = ak.index_zh_a_hist(symbol=symbol, period="daily",
                                         start_date=start, end_date=end_s)
                break
            except Exception as e:
                last_err = e
                time.sleep(3 + attempt * 4)
    if raw is not None and not raw.empty:
        g = raw.rename(columns={"日期": "date", "收盘": "C"})
        g["date"] = pd.to_datetime(g["date"])
        out = g.set_index("date").sort_index()
        out[["C"]].to_parquet(f)
        _cleanup_old_index_files(symbol, keep=3)
        return out

    # 3) 最近旧缓存兜底
    older = sorted(DATA_DIR.glob(f"meta/index_{symbol}_*.parquet"),
                   key=lambda p: p.stat().st_mtime)
    if older:
        print(f"[warn] index {symbol} 拉取失败({last_err})，使用旧缓存 {older[-1].name}")
        return pd.read_parquet(older[-1])
    raise RuntimeError(f"index {symbol} failed: {last_err}")


def _cleanup_old_index_files(symbol: str, keep: int = 3) -> None:
    """指数缓存按日累积，只保留最近 keep 份。"""
    files = sorted(DATA_DIR.glob(f"meta/index_{symbol}_*.parquet"),
                   key=lambda p: p.stat().st_mtime)
    for p in files[:-keep]:
        try:
            p.unlink()
        except OSError:
            pass


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
            with no_proxy():
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
