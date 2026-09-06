"""Fast, read-only health summaries for local SPS data artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def _parquet_last_date(path: Path) -> pd.Timestamp:
    parquet = pq.ParquetFile(path)
    metadata = parquet.schema_arrow.metadata or {}
    pandas_meta = json.loads(metadata.get(b"pandas", b"{}").decode("utf-8"))
    index_fields = [
        value for value in pandas_meta.get("index_columns", [])
        if isinstance(value, str)
    ]
    candidates = index_fields + ["date", "__index_level_0__"]
    for field in candidates:
        if field not in parquet.schema.names:
            continue
        column_index = parquet.schema.names.index(field)
        maxima = []
        for group_index in range(parquet.metadata.num_row_groups):
            stats = parquet.metadata.row_group(group_index).column(column_index).statistics
            if stats is not None and stats.has_min_max:
                maxima.append(stats.max)
        if maxima:
            return pd.Timestamp(max(maxima))

    frame = pd.read_parquet(path, columns=[])
    if frame.empty:
        raise ValueError("empty parquet")
    return pd.Timestamp(frame.index[-1])


def summarize_daily_cache(daily_dir: Path) -> dict:
    """Report freshness across every stock cache, using Parquet metadata."""
    by_symbol: dict[str, list[Path]] = {}
    for path in daily_dir.glob("*.parquet") if daily_dir.exists() else []:
        if path.name.startswith("index_"):
            continue
        by_symbol.setdefault(path.stem.split("_")[0], []).append(path)

    # 同一票可能有多个来源文件（HiThink/akshare 残留、不同起始年标签）。
    # 必须按「数据最后日期」取最新的那个，按 mtime 选会把最新数据判成过期。
    selected: dict[str, Path] = {}
    for symbol, paths in by_symbol.items():
        if len(paths) == 1:
            selected[symbol] = paths[0]
            continue
        try:
            selected[symbol] = max(paths, key=_parquet_last_date)
        except Exception:
            selected[symbol] = max(paths, key=lambda p: p.stat().st_mtime_ns)

    dates: dict[str, str] = {}
    invalid = 0
    hithink_count = sum("_akshare" not in path.stem for path in selected.values())
    akshare_count = len(selected) - hithink_count
    for symbol, path in selected.items():
        try:
            dates[symbol] = _parquet_last_date(path).strftime("%Y-%m-%d")
        except Exception:
            invalid += 1

    if not dates:
        return {
            "ready": False,
            "total_symbols": len(selected),
            "current_symbols": 0,
            "stale_symbols": len(selected),
            "coverage_pct": 0.0,
            "invalid_files": invalid,
            "hithink_cache_symbols": hithink_count,
            "akshare_cache_symbols": akshare_count,
        }

    last_date = max(dates.values())
    oldest_date = min(dates.values())
    current = sum(value == last_date for value in dates.values())
    total = len(selected)
    # 长期停牌/退市整理/池外ETF残留的票永远追不上最新日期，单独分桶，
    # 不让它们把"未同步"告警撑成一个永远消不掉的数字。
    # 损坏/不可读的缓存计入未同步（无法读取 = 需要处理）。
    stale_symbols = suspended_symbols = invalid
    for value in dates.values():
        if value == last_date:
            continue
        if (pd.Timestamp(last_date) - pd.Timestamp(value)).days <= 30:
            stale_symbols += 1
        else:
            suspended_symbols += 1
    age_days = (
        pd.Timestamp.today().normalize() - pd.Timestamp(last_date).normalize()
    ).days
    return {
        "ready": True,
        "last_date": last_date,
        "oldest_date": oldest_date,
        "age_days": int(age_days),
        "total_symbols": total,
        "current_symbols": current,
        "stale_symbols": stale_symbols,
        "suspended_symbols": suspended_symbols,
        "coverage_pct": round(current / total * 100, 2),
        "invalid_files": invalid,
        "hithink_cache_symbols": hithink_count,
        "akshare_cache_symbols": akshare_count,
    }
