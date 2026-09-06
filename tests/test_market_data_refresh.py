"""Regression coverage for HiThink-first market data refreshes."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def _frame(dates: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "O": closes,
            "H": [value + 0.2 for value in closes],
            "L": [value - 0.2 for value in closes],
            "C": closes,
            "V": [1000.0] * len(dates),
        },
        index=pd.to_datetime(dates),
    )


def test_hithink_history_parser_accepts_current_cli_contract_and_normalizes_volume():
    from sps.data import _parse_hithink_history

    payload = {
        "ok": True,
        "data": {
            "item": [{
                "date_ms": 1787155200000,
                "open_price": 11.2,
                "high_price": 11.4,
                "low_price": 11.19,
                "close_price": 11.4,
                "volume": 118357823,
            }],
            "adjust": "forward",
        },
    }

    frame = _parse_hithink_history(payload)

    assert frame.index.strftime("%Y-%m-%d").tolist() == ["2026-08-20"]
    assert frame.iloc[0].to_dict() == {
        "O": 11.2, "H": 11.4, "L": 11.19, "C": 11.4, "V": 1183578.23
    }


def test_hithink_request_always_supplies_required_window(monkeypatch):
    import sps.data as data_module

    seen: list[str] = []
    response = {
        "ok": True,
        "data": {"item": [{
            "date_ms": 1788364800000,
            "open_price": 11.88,
            "high_price": 12.08,
            "low_price": 11.83,
            "close_price": 11.88,
            "volume": 110513439,
        }]},
    }

    def fake_run(command, **_kwargs):
        seen.extend(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps(response), stderr="")

    monkeypatch.setattr(data_module.subprocess, "run", fake_run)

    frame = data_module.get_daily_hithink("000001", start="20260820")

    assert frame is not None and len(frame) == 1
    assert "--start-ms" in seen
    assert "--end-ms" in seen
    assert seen[seen.index("--adjust") + 1] == "forward"


def test_refresh_replaces_stale_akshare_cache_with_merged_hithink_cache(
    tmp_path: Path, monkeypatch
):
    import sps.data as data_module

    daily = tmp_path / "daily"
    daily.mkdir()
    stale = _frame(["2026-09-01", "2026-09-02"], [10.0, 10.1])
    stale.to_parquet(daily / "000001_20190101_latest_akshare.parquet")
    fresh = _frame(["2026-09-02", "2026-09-03"], [10.1, 10.3])

    monkeypatch.setattr(data_module, "DAILY_DIR", daily)
    monkeypatch.setattr(data_module, "META_DIR", tmp_path / "meta")
    monkeypatch.setattr(data_module, "hithink_available", lambda: True)
    monkeypatch.setattr(data_module, "get_market_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(
        data_module, "get_batch_hithink", lambda *a, **k: {"000001": fresh}
    )
    monkeypatch.setattr(
        data_module, "get_daily_akshare",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("unexpected fallback")),
    )

    result = data_module.batch_get_daily(["000001"], start="20190101", refresh=True)

    assert result["000001"].index.strftime("%Y-%m-%d").tolist() == [
        "2026-09-01", "2026-09-02", "2026-09-03"
    ]
    assert result["000001"].loc["2026-09-02", "C"] == 10.1
    assert (daily / "000001_20190101_latest.parquet").exists()
    assert data_module.LAST_BATCH_STATS == {
        "requested": 1,
        "hithink_refreshed": 1,
        "akshare_fallback": 0,
        "stale_cache_retained": 0,
        "qfq_refetched": 0,
        "snapshot_appended": 0,
        "stale_sidecars_removed": 1,   # 测试装置里的过期 akshare 残留被顺手清理
    }


def test_refresh_falls_back_to_akshare_and_keeps_history(tmp_path: Path, monkeypatch):
    import sps.data as data_module

    daily = tmp_path / "daily"
    daily.mkdir()
    stale = _frame(["2026-09-01", "2026-09-02"], [10.0, 10.1])
    stale.to_parquet(daily / "000001_20190101_latest_akshare.parquet")
    fresh = _frame(["2026-09-02", "2026-09-03"], [10.1, 10.25])

    monkeypatch.setattr(data_module, "DAILY_DIR", daily)
    monkeypatch.setattr(data_module, "META_DIR", tmp_path / "meta")
    monkeypatch.setattr(data_module, "hithink_available", lambda: True)
    monkeypatch.setattr(data_module, "get_market_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(data_module, "get_batch_hithink", lambda *a, **k: {})
    monkeypatch.setattr(data_module, "get_daily_akshare", lambda *a, **k: fresh)

    result = data_module.batch_get_daily(["000001"], start="20190101", refresh=True)

    assert result["000001"].index.strftime("%Y-%m-%d").tolist() == [
        "2026-09-01", "2026-09-02", "2026-09-03"
    ]
    assert result["000001"].loc["2026-09-02", "C"] == 10.1
    assert data_module.LAST_BATCH_STATS["akshare_fallback"] == 1


def test_refresh_refetches_full_history_when_qfq_basis_changed(
    tmp_path: Path, monkeypatch
):
    """回归：除权后前复权整条历史被重写，重叠窗口对不上必须全量重拉，禁止拼接。"""
    import sps.data as data_module

    daily = tmp_path / "daily"
    daily.mkdir()
    # 旧缓存：除权前基准，10 元
    stale = _frame(
        pd.bdate_range("2026-06-01", periods=70).strftime("%Y-%m-%d").tolist(),
        [10.0] * 70,
    )
    stale.to_parquet(daily / "000001_20190101_latest.parquet")
    # 增量新数据：除权后新基准 9 元（10 派 1 相当于 -10%）
    fresh = _frame(
        pd.bdate_range("2026-08-20", periods=14).strftime("%Y-%m-%d").tolist(),
        [9.0] * 14,
    )
    full = _frame(
        pd.bdate_range("2026-01-05", periods=180).strftime("%Y-%m-%d").tolist(),
        [9.0] * 180,
    )

    monkeypatch.setattr(data_module, "DAILY_DIR", daily)
    monkeypatch.setattr(data_module, "META_DIR", tmp_path / "meta")
    monkeypatch.setattr(data_module, "hithink_available", lambda: True)
    monkeypatch.setattr(data_module, "get_market_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(
        data_module, "get_batch_hithink", lambda *a, **k: {"000001": fresh}
    )
    monkeypatch.setattr(
        data_module, "get_daily_hithink", lambda *a, **k: full.copy()
    )

    result = data_module.batch_get_daily(["000001"], start="20190101", refresh=True)

    merged = result["000001"]
    assert merged.index.strftime("%Y-%m-%d").tolist() == (
        full.index.strftime("%Y-%m-%d").tolist()
    )
    # 历史段必须是新基准，不允许残留旧基准的 10 元价格
    assert (merged["C"] == 9.0).all()
    assert data_module.LAST_BATCH_STATS["qfq_refetched"] == 1


def test_cache_health_reports_hithink_and_akshare_provenance(tmp_path: Path):
    from sps.health import summarize_daily_cache

    daily = tmp_path / "daily"
    daily.mkdir()
    _frame(["2026-09-03"], [10.0]).to_parquet(
        daily / "000001_20190101_latest.parquet"
    )
    _frame(["2026-09-03"], [20.0]).to_parquet(
        daily / "000002_20190101_latest_akshare.parquet"
    )

    status = summarize_daily_cache(daily)

    assert status["hithink_cache_symbols"] == 1
    assert status["akshare_cache_symbols"] == 1


def test_hithink_connection_test_uses_current_required_history_window(
    tmp_path: Path, monkeypatch
):
    import sps.hithink_cfg as config

    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout='{"ok":true}', stderr="")

    monkeypatch.setattr(config, "CFG_FILE", tmp_path / "hithink_config.json")
    monkeypatch.setattr(config.subprocess, "run", fake_run)

    result = config.test_hithink("secret-key")

    history_command = commands[-1]
    assert result["ok"] is True
    assert "--start-ms" in history_command
    assert "--end-ms" in history_command


def test_scan_refreshes_even_when_latest_cache_file_exists(tmp_path: Path, monkeypatch):
    import scripts.run_scan as run_scan
    import sps.data as data_module

    data_dir = tmp_path / "data"
    daily = data_dir / "daily"
    daily.mkdir(parents=True)
    _frame(["2026-09-01", "2026-09-02"], [10.0, 10.1]).to_parquet(
        daily / "000001_20190101_latest_akshare.parquet"
    )
    refreshed = _frame(
        pd.bdate_range(end="2026-09-03", periods=80).strftime("%Y-%m-%d").tolist(),
        [10.0] * 80,
    )

    monkeypatch.setattr(run_scan, "OUT_DIR", tmp_path / "runs")
    monkeypatch.setattr(run_scan, "DATA_DIR", data_dir)
    monkeypatch.setattr(run_scan, "get_index", lambda *a, **k: refreshed[["C"]])
    monkeypatch.setattr(
        run_scan, "market_regime", lambda close: pd.Series("bull", index=close.index)
    )
    monkeypatch.setattr(run_scan, "get_fundamental", lambda _symbol: {"roe": 1.0})
    monkeypatch.setattr(
        run_scan, "check_redlines", lambda *a, **k: (False, ["R4:roe1.0"])
    )

    def fake_batch(*_args, **kwargs):
        return {"000001": refreshed} if kwargs.get("refresh") is True else {}

    monkeypatch.setattr(data_module, "batch_get_daily", fake_batch)
    monkeypatch.setattr(data_module, "hithink_available", lambda: True)
    monkeypatch.setattr(data_module, "get_market_snapshot", lambda *a, **k: None)

    run_scan.run(["000001"], kind_map={"000001": "stock"})
    meta = json.loads(
        (tmp_path / "runs" / "candidates_meta.json").read_text(encoding="utf-8")
    )

    assert meta["filtered_symbols"] == 1
    assert meta["insufficient_history_symbols"] == 0


# ---------------- 快照快路径（每日增量） ----------------

def test_snapshot_parser_normalizes_fields_and_volume(monkeypatch):
    import sps.data as data_module

    page = {
        "ok": True,
        "data": {"total": 2, "item": [
            {"ticker": "000001", "open_price": 11.86, "high_price": 12.0,
             "low_price": 11.85, "last_price": 11.89, "volume": 81437295,
             "prev_price": 11.88},
            {"ticker": "000002", "open_price": 0, "high_price": 0,
             "low_price": 0, "last_price": 0, "volume": 0},   # 停牌：应剔除
        ]},
    }

    def fake_run(command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(page), stderr="")

    monkeypatch.setattr(data_module.subprocess, "run", fake_run)
    monkeypatch.setattr(data_module, "_trade_calendar",
                        lambda: ["20260903", "20260904"])

    snap = data_module.get_market_snapshot()
    assert snap is not None
    bars, trade_day = snap
    assert trade_day == "20260904"
    assert set(bars) == {"000001"}
    row = bars["000001"]
    assert row.index[0] == pd.Timestamp("2026-09-04")
    assert row.iloc[0]["V"] == 814372.95   # 股 → 手
    assert row.iloc[0]["C"] == 11.89
    assert row.iloc[0]["PREV"] == 11.88   # 昨收，用于复权基准变化检测


def test_snapshot_fast_path_appends_latest_bar(tmp_path, monkeypatch):
    import sps.data as data_module

    daily = tmp_path / "daily"
    daily.mkdir()
    # 缓存最新到前一交易日 09-03
    stale = _frame(["2026-09-02", "2026-09-03"], [10.0, 10.1])
    stale.to_parquet(daily / "000001_20190101_latest.parquet")
    bar = _frame(["2026-09-04"], [10.5])
    calendar = ["20260901", "20260902", "20260903", "20260904"]

    monkeypatch.setattr(data_module, "DAILY_DIR", daily)
    monkeypatch.setattr(data_module, "META_DIR", tmp_path / "meta")
    monkeypatch.setattr(data_module, "hithink_available", lambda: True)
    monkeypatch.setattr(data_module, "_trade_calendar", lambda: calendar)
    monkeypatch.setattr(data_module, "get_market_snapshot",
                        lambda *a, **k: ({"000001": bar}, "20260904"))
    monkeypatch.setattr(
        data_module, "get_batch_hithink",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("快路径不应触发慢路径")))

    result = data_module.batch_get_daily(["000001"], start="20190101", refresh=True)

    merged = result["000001"]
    assert merged.index.strftime("%Y-%m-%d").tolist() == [
        "2026-09-02", "2026-09-03", "2026-09-04"]
    assert merged.loc["2026-09-04", "C"] == 10.5
    assert (daily / "000001_20190101_latest.parquet").exists()
    assert data_module.LAST_BATCH_STATS["snapshot_appended"] == 1
    assert data_module.LAST_BATCH_STATS["hithink_refreshed"] == 0


def test_snapshot_same_day_overwrites_intraday_value(tmp_path, monkeypatch):
    import sps.data as data_module

    daily = tmp_path / "daily"
    daily.mkdir()
    # 缓存已有当日盘中旧值
    intraday = _frame(["2026-09-03", "2026-09-04"], [10.0, 10.2])
    intraday.to_parquet(daily / "000001_20190101_latest.parquet")
    bar = _frame(["2026-09-04"], [10.5])   # 收盘快照
    calendar = ["20260903", "20260904"]

    monkeypatch.setattr(data_module, "DAILY_DIR", daily)
    monkeypatch.setattr(data_module, "META_DIR", tmp_path / "meta")
    monkeypatch.setattr(data_module, "hithink_available", lambda: True)
    monkeypatch.setattr(data_module, "_trade_calendar", lambda: calendar)
    monkeypatch.setattr(data_module, "get_market_snapshot",
                        lambda *a, **k: ({"000001": bar}, "20260904"))

    result = data_module.batch_get_daily(["000001"], start="20190101", refresh=True)

    assert result["000001"].loc["2026-09-04", "C"] == 10.5
    assert data_module.LAST_BATCH_STATS["snapshot_appended"] == 1


def test_snapshot_gap_falls_back_to_history(tmp_path, monkeypatch):
    import sps.data as data_module

    daily = tmp_path / "daily"
    daily.mkdir()
    # 缓存落后两个交易日（09-01），快照只够补一天 → 必须走慢路径补历史
    stale = _frame(["2026-09-01"], [10.0])
    stale.to_parquet(daily / "000001_20190101_latest.parquet")
    bar = _frame(["2026-09-04"], [10.5])
    fresh_hist = _frame(["2026-09-02", "2026-09-03", "2026-09-04"],
                        [10.1, 10.3, 10.5])
    calendar = ["20260901", "20260902", "20260903", "20260904"]

    monkeypatch.setattr(data_module, "DAILY_DIR", daily)
    monkeypatch.setattr(data_module, "META_DIR", tmp_path / "meta")
    monkeypatch.setattr(data_module, "hithink_available", lambda: True)
    monkeypatch.setattr(data_module, "_trade_calendar", lambda: calendar)
    monkeypatch.setattr(data_module, "get_market_snapshot",
                        lambda *a, **k: ({"000001": bar}, "20260904"))
    monkeypatch.setattr(data_module, "get_batch_hithink",
                        lambda *a, **k: {"000001": fresh_hist})
    monkeypatch.setattr(data_module, "get_daily_akshare", lambda *a, **k: fresh_hist)

    result = data_module.batch_get_daily(["000001"], start="20190101", refresh=True)

    assert result["000001"].index.strftime("%Y-%m-%d").tolist() == [
        "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"]
    assert data_module.LAST_BATCH_STATS["snapshot_appended"] == 0
    assert data_module.LAST_BATCH_STATS["hithink_refreshed"] == 1


def test_score_event_tolerates_nan_rps():
    """回归：次新股 RPS 分位为 NaN 时不得崩溃（此前 int(min(nan,1.0)*30) 抛 ValueError）。"""
    import scripts.run_scan as run_scan

    wide = pd.DataFrame({"000001": [0.5, float("nan")], "000002": [0.3, 0.7]},
                        index=pd.to_datetime(["2026-08-03", "2026-08-04"]))
    rps_pct = wide.rank(axis=1, pct=True)
    e = {"symbol": "000001", "signal_date": "2026-08-04",
         "pattern": "FLAT_BREAKOUT", "regime": "bull"}
    run_scan.score_event(e, rps_pct)

    assert e["rps50"] is None
    assert e["score"] == 50          # neutral 基础分，无 RPS 加成
    # 有正常 RPS 的票仍获得加成
    e2 = {"symbol": "000002", "signal_date": "2026-08-04",
          "pattern": "FLAT_BREAKOUT", "regime": "bull"}
    run_scan.score_event(e2, rps_pct)
    assert e2["rps50"] is not None and e2["score"] >= 50


# ---------------- K线读取按数据日期选文件 + 残留清理 ----------------

def test_candle_reader_picks_freshest_data_not_newest_mtime(tmp_path, monkeypatch):
    """回归：K线读取必须按「数据最后日期」选文件，不能按 mtime。

    此前按 mtime 升序取第一个，导致过期 akshare 残留（写入更早）
    掩盖了已更新到最新交易日的主缓存，K线停在旧日期。
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import app as app_mod

    daily = tmp_path / "daily"
    daily.mkdir()
    main = _frame(["2026-09-03", "2026-09-04"], [10.0, 10.5])     # 新数据
    main.to_parquet(daily / "600000_20190101_latest.parquet")
    side = _frame(["2026-09-02", "2026-09-03"], [9.0, 9.1])       # 旧残留
    side.to_parquet(daily / "600000_20190101_latest_akshare.parquet")
    # 人为让旧残留的 mtime 更新，模拟真实事故场景
    os.utime(daily / "600000_20190101_latest.parquet", (1_000_000, 1_000_000))
    os.utime(daily / "600000_20190101_latest_akshare.parquet", (2_000_000, 2_000_000))

    monkeypatch.setattr(app_mod, "DATA_DIR", tmp_path)
    raw = app_mod.candle_raw("600000")
    assert raw["dates"][-1] == "2026-09-04"
    assert raw["C"][-1] == 10.5

    # health 统计同样按数据日期选文件
    from sps.health import summarize_daily_cache
    status = summarize_daily_cache(daily)
    assert status["last_date"] == "2026-09-04"
    assert status["stale_symbols"] == 0


def test_refresh_removes_stale_akshare_sidecar(tmp_path, monkeypatch):
    """刷新后主缓存更新时，过期的 akshare 残留文件应被自动删除。"""
    import sps.data as data_module

    daily = tmp_path / "daily"
    daily.mkdir()
    _frame(["2026-09-03"], [10.0]).to_parquet(daily / "000001_20190101_latest_akshare.parquet")
    fresh = _frame(["2026-09-03", "2026-09-04"], [10.1, 10.2])

    monkeypatch.setattr(data_module, "DAILY_DIR", daily)
    monkeypatch.setattr(data_module, "META_DIR", tmp_path / "meta")
    monkeypatch.setattr(data_module, "hithink_available", lambda: True)
    monkeypatch.setattr(data_module, "get_market_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(data_module, "get_batch_hithink",
                        lambda *a, **k: {"000001": fresh})

    result = data_module.batch_get_daily(["000001"], start="20190101", refresh=True)

    assert result["000001"].index[-1] == pd.Timestamp("2026-09-04")
    assert not (daily / "000001_20190101_latest_akshare.parquet").exists()
    assert data_module.LAST_BATCH_STATS["stale_sidecars_removed"] == 1


def test_snapshot_skipped_during_market_hours(monkeypatch):
    """回归：交易时段不得使用快照——此时快照是当日盘中半根K线，
    而最近已收盘日是前一交易日，合并会把昨日完成K线覆盖成盘中数据。"""
    from types import SimpleNamespace
    import sps.data as data_module

    class FakeTime:
        @staticmethod
        def localtime():
            return SimpleNamespace(tm_hour=10, tm_min=0)   # 周一 10:00 盘中
        @staticmethod
        def strftime(_fmt):
            return "20260907"

    monkeypatch.setattr(data_module, "time", FakeTime())
    monkeypatch.setattr(data_module, "_trade_calendar",
                        lambda: ["20260904", "20260907"])

    assert data_module.get_market_snapshot() is None


def test_snapshot_dividend_day_falls_back_to_history(tmp_path, monkeypatch):
    """回归：除权日昨收（新基准）与旧缓存收盘（旧基准）对不上时，
    快路径必须放弃该票、交给慢路径全量校验，禁止拼接污染。"""
    import sps.data as data_module

    daily = tmp_path / "daily"
    daily.mkdir()
    # 旧缓存（旧基准）：昨日收盘 10.0
    stale = _frame(["2026-09-03"], [10.0])
    stale.to_parquet(daily / "000001_20190101_latest.parquet")
    # 快照（新基准，10派1后昨收被调整为 9.0）：与旧缓存差 10%
    bar = _frame(["2026-09-04"], [9.0])
    bar["PREV"] = [9.0]
    calendar = ["20260903", "20260904"]

    monkeypatch.setattr(data_module, "DAILY_DIR", daily)
    monkeypatch.setattr(data_module, "META_DIR", tmp_path / "meta")
    monkeypatch.setattr(data_module, "hithink_available", lambda: True)
    monkeypatch.setattr(data_module, "_trade_calendar", lambda: calendar)
    monkeypatch.setattr(
        data_module, "get_market_snapshot",
        lambda *a, **k: ({"000001": bar}, "20260904"))
    # 慢路径返回全量新基准历史（含重叠日 09-03）
    full = _frame(pd.bdate_range("2026-05-01", "2026-09-04").strftime("%Y-%m-%d"),
                  [9.0] * len(pd.bdate_range("2026-05-01", "2026-09-04")))
    monkeypatch.setattr(data_module, "get_batch_hithink",
                        lambda *a, **k: {"000001": full})

    result = data_module.batch_get_daily(["000001"], start="20190101", refresh=True)

    # 合并结果全部为新基准，不残留旧基准的 10.0
    assert (result["000001"]["C"] == 9.0).all()
    assert data_module.LAST_BATCH_STATS["snapshot_appended"] == 0
    assert data_module.LAST_BATCH_STATS["hithink_refreshed"] == 1


def test_daily_cache_detail_buckets_with_gap_days(tmp_path):
    """明细接口：未同步/长期停牌分桶正确，带落后天数与排序。"""
    from sps.health import daily_cache_detail

    daily = tmp_path / "daily"
    daily.mkdir()
    _frame(["2026-09-04"], [10.0]).to_parquet(daily / "000001_latest.parquet")   # 最新
    _frame(["2026-09-01"], [10.0]).to_parquet(daily / "000002_latest.parquet")   # 未同步(3天)
    _frame(["2026-06-01"], [10.0]).to_parquet(daily / "000003_latest.parquet")   # 长期停牌

    d = daily_cache_detail(daily)

    assert d["ready"] is True
    assert [x["symbol"] for x in d["stale"]] == ["000002"]
    assert d["stale"][0]["gap_days"] == 3
    assert [x["symbol"] for x in d["suspended"]] == ["000003"]
