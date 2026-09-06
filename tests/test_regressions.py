"""Regression tests for the user-visible failures found in the project audit."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


def _daily_frame(last_date: str, periods: int = 80) -> pd.DataFrame:
    idx = pd.bdate_range(end=last_date, periods=periods)
    return pd.DataFrame(
        {
            "O": [100.0] * periods,
            "H": [102.0] * periods,
            "L": [98.0] * periods,
            "C": [101.0] * periods,
            "V": [1_000_000] * periods,
        },
        index=idx,
    )


def test_stop_loss_contract_separates_thresholds_from_realized_exits():
    from sps import stats

    df = _daily_frame("2026-09-03", periods=3)
    df.loc[:, "O"] = [100.0, 101.0, 110.0]
    df.loc[:, "L"] = [99.0, 100.0, 109.0]
    df.loc[:, "C"] = [100.0, 105.0, 110.0]

    assert stats.stop_prices(100.0) == {
        -0.05: 95.0,
        -0.07: 93.0,
        -0.10: 90.0,
    }
    assert stats.stop_exit_sim(df, 0, 100.0) == {
        -0.05: 110.0,
        -0.07: 110.0,
        -0.10: 110.0,
    }


def test_scan_counts_filtered_symbols_and_writes_coverage_metadata(
    tmp_path: Path, monkeypatch, capsys
):
    import scripts.run_scan as run_scan
    import sps.data as data_module

    df = _daily_frame("2026-09-03")
    monkeypatch.setattr(run_scan, "OUT_DIR", tmp_path)
    monkeypatch.setattr(run_scan, "get_index", lambda *a, **k: df[["C"]])
    monkeypatch.setattr(
        run_scan, "market_regime", lambda close: pd.Series("bull", index=close.index)
    )
    monkeypatch.setattr(run_scan, "get_fundamental", lambda symbol: {"roe": 1.0})
    monkeypatch.setattr(
        run_scan, "check_redlines", lambda *a, **k: (False, ["R4:roe1.0"])
    )
    monkeypatch.setattr(data_module, "HIT_HINK_AVAILABLE", False)
    monkeypatch.setattr(
        data_module, "batch_get_daily", lambda *a, **k: {"000001": df}
    )

    assert run_scan.run(["000001"], kind_map={"000001": "stock"}) == []
    assert "基本面红线过滤：淘汰 1 只" in capsys.readouterr().out

    meta = json.loads((tmp_path / "candidates_meta.json").read_text(encoding="utf-8"))
    assert meta["requested_symbols"] == 1
    assert meta["loaded_symbols"] == 1
    assert meta["filtered_symbols"] == 1
    assert meta["failed_load_symbols"] == 0
    assert meta["insufficient_history_symbols"] == 0
    assert meta["processed_symbols"] == 0
    assert meta["detector_failures"] == 0
    assert meta["candidate_count"] == 0


def test_data_status_reports_full_cache_coverage(tmp_path: Path, monkeypatch):
    import scripts.app as app_module

    daily = tmp_path / "daily"
    daily.mkdir()
    _daily_frame("2026-09-03").to_parquet(daily / "000001_latest.parquet")
    _daily_frame("2026-09-03").to_parquet(daily / "000002_latest.parquet")
    _daily_frame("2026-09-02").to_parquet(daily / "000003_latest.parquet")
    monkeypatch.setattr(app_module, "DATA_DIR", tmp_path)

    payload = app_module.app.test_client().get("/api/data_status").get_json()

    assert payload["last_date"] == "2026-09-03"
    assert payload["total_symbols"] == 3
    assert payload["current_symbols"] == 2
    assert payload["stale_symbols"] == 1
    assert payload["coverage_pct"] == 66.67


def test_packaged_data_directory_can_be_pinned_by_launcher(tmp_path: Path, monkeypatch):
    import sps.data as data_module

    monkeypatch.setenv("SPS_DATA_DIR", str(tmp_path / "data"))

    assert data_module._resolve_data_dir() == (tmp_path / "data").resolve()


def test_invalid_cache_counts_against_coverage(tmp_path: Path):
    from sps.health import summarize_daily_cache

    daily = tmp_path / "daily"
    daily.mkdir()
    _daily_frame("2026-09-03").to_parquet(daily / "000001_latest.parquet")
    _daily_frame("2026-09-03").to_parquet(daily / "000002_latest.parquet")
    (daily / "000003_latest.parquet").write_bytes(b"not parquet")

    payload = summarize_daily_cache(daily)

    assert payload["total_symbols"] == 3
    assert payload["current_symbols"] == 2
    assert payload["stale_symbols"] == 1
    assert payload["invalid_files"] == 1
    assert payload["coverage_pct"] == 66.67


def test_all_invalid_cache_still_reports_zero_coverage(tmp_path: Path):
    from sps.health import summarize_daily_cache

    daily = tmp_path / "daily"
    daily.mkdir()
    (daily / "000001_latest.parquet").write_bytes(b"not parquet")

    payload = summarize_daily_cache(daily)

    assert payload == {
        "ready": False,
        "total_symbols": 1,
        "current_symbols": 0,
        "stale_symbols": 1,
        "coverage_pct": 0.0,
        "invalid_files": 1,
        "hithink_cache_symbols": 1,
        "akshare_cache_symbols": 0,
    }


def test_candidates_api_exposes_generation_and_coverage_metadata(
    tmp_path: Path, monkeypatch
):
    import scripts.app as app_module

    candidate = {
        "symbol": "000001",
        "signal_date": "2026-09-02",
        "entry": {"price": 10.0, "stops": {"-0.07": 9.3}},
    }
    from sps.candidates import write_candidate_artifacts

    write_candidate_artifacts(
        tmp_path / "candidates.json",
        [candidate],
        {
            "generated_at": "2026-09-03T16:00:00+08:00",
            "requested_symbols": 5015,
            "loaded_symbols": 5000,
            "filtered_symbols": 15,
            "candidate_count": 1,
        },
    )
    monkeypatch.setattr(app_module, "RUN_DIR", tmp_path)
    monkeypatch.setattr(app_module, "symbol_names", lambda: {"000001": "平安银行"})

    payload = app_module.app.test_client().get("/api/candidates").get_json()

    assert payload["candidates"][0]["_name"] == "平安银行"
    assert payload["meta"]["generated_at"] == "2026-09-03T16:00:00+08:00"
    assert payload["meta"]["requested_symbols"] == 5015
    assert payload["meta"]["candidate_symbols"] == 1
    assert payload["meta"]["coverage_known"] is True


def test_candidates_api_rejects_mismatched_sidecar_metadata(tmp_path: Path, monkeypatch):
    import scripts.app as app_module

    path = tmp_path / "candidates.json"
    path.write_text(json.dumps([{"symbol": "000001"}]), encoding="utf-8")
    (tmp_path / "candidates_meta.json").write_text(
        json.dumps({
            "schema_version": 2,
            "candidates_sha256": "stale",
            "requested_symbols": 9999,
            "generated_at": "2099-01-01T00:00:00+08:00",
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(app_module, "RUN_DIR", tmp_path)
    monkeypatch.setattr(app_module, "symbol_names", lambda: {})

    meta = app_module.app.test_client().get("/api/candidates").get_json()["meta"]

    assert meta["coverage_known"] is False
    assert "requested_symbols" not in meta
    assert not meta["generated_at"].startswith("2099-")


def test_browser_opens_only_after_local_server_is_ready():
    import scripts.app as app_module

    checks = iter([False, False, True])
    opened: list[str] = []

    ok = app_module.open_browser_when_ready(
        "http://127.0.0.1:5000",
        attempts=3,
        interval=0,
        probe=lambda _url: next(checks),
        browser_open=opened.append,
        sleep=lambda _seconds: None,
    )

    assert ok is True
    assert opened == ["http://127.0.0.1:5000"]


def test_server_identity_rejects_foreign_http_service():
    import scripts.app as app_module

    class Response:
        status = 200

        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    assert app_module.server_is_ready(
        "http://127.0.0.1:5000", opener=lambda *_a, **_k: Response({"service": "other"})
    ) is False
    assert app_module.server_is_ready(
        "http://127.0.0.1:5000", opener=lambda *_a, **_k: Response({"service": "SPS", "ok": True})
    ) is True


def test_health_endpoint_identifies_sps():
    import scripts.app as app_module

    assert app_module.app.test_client().get("/api/health").get_json() == {
        "service": "SPS", "ok": True, "version": "1.1"
    }


def test_legacy_candidate_stop_values_are_normalized_idempotently():
    from sps import candidates

    legacy = {
        "symbol": "000001",
        "entry": {
            "price": 100.0,
            "stops": {"-0.05": 120.0, "-0.07": 120.0, "-0.1": 90.0},
        },
    }

    normalized = candidates.normalize_candidate(legacy)
    assert normalized["entry"]["stops"] == {
        "-0.05": 95.0,
        "-0.07": 93.0,
        "-0.1": 90.0,
    }
    assert normalized["entry"]["stop_exits"] == {
        "-0.05": 120.0,
        "-0.07": 120.0,
        "-0.1": 90.0,
    }
    assert normalized["entry"]["stop_contract"] == "thresholds-v2"
    assert candidates.normalize_candidate(normalized) == normalized


def test_modern_threshold_only_candidate_does_not_invent_realized_exits():
    from sps.candidates import normalize_candidate

    modern = {
        "symbol": "000001",
        "entry": {
            "price": 100.0,
            "stops": {"-0.07": 93.0},
            "stop_contract": "thresholds-v2",
        },
    }

    normalized = normalize_candidate(modern)
    assert normalized == modern
    assert "stop_exits" not in normalized["entry"]


def test_candidate_file_migration_keeps_backup(tmp_path: Path):
    from sps.candidates import load_candidates_metadata, migrate_candidates_file

    path = tmp_path / "candidates.json"
    path.write_text(
        json.dumps(
            [{"symbol": "000001", "entry": {"price": 10.0,
              "stops": {"-0.07": 12.0}}}]
        ),
        encoding="utf-8",
    )
    original_mtime = 1_788_166_800
    os.utime(path, (original_mtime, original_mtime))

    result = migrate_candidates_file(path)

    assert result["changed"] is True
    assert result["backup"] is not None
    assert Path(result["backup"]).exists()
    assert int(path.stat().st_mtime) == original_mtime
    migrated = json.loads(path.read_text(encoding="utf-8"))
    assert migrated[0]["entry"]["stops"]["-0.07"] == 9.3
    assert migrated[0]["entry"]["stop_contract"] == "thresholds-v2"
    meta_path = tmp_path / "candidates_meta.json"
    assert meta_path.exists()
    metadata = load_candidates_metadata(path, migrated)
    assert metadata["schema_version"] == 2
    assert metadata["coverage_known"] is False
    assert metadata["artifact_valid"] is True


def test_candidate_load_invokes_one_time_migration(tmp_path: Path):
    from sps.candidates import load_candidates_file

    path = tmp_path / "candidates.json"
    path.write_text(json.dumps([{
        "symbol": "000001",
        "entry": {"price": 10.0, "stops": {"-0.07": 12.0}},
    }]), encoding="utf-8")

    loaded = load_candidates_file(path)

    assert loaded[0]["entry"]["stop_contract"] == "thresholds-v2"
    assert (tmp_path / "candidates_meta.json").exists()
    assert len(list(tmp_path.glob("candidates.legacy-*.json"))) == 1
    load_candidates_file(path)
    assert len(list(tmp_path.glob("candidates.legacy-*.json"))) == 1


def test_scan_accounting_reconciles_missing_and_short_histories(
    tmp_path: Path, monkeypatch
):
    import scripts.run_scan as run_scan
    import sps.data as data_module

    index = _daily_frame("2026-09-03")
    short = _daily_frame("2026-09-03", periods=20)
    monkeypatch.setattr(run_scan, "OUT_DIR", tmp_path)
    monkeypatch.setattr(run_scan, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(run_scan, "get_index", lambda *a, **k: index[["C"]])
    monkeypatch.setattr(
        run_scan, "market_regime", lambda close: pd.Series("bull", index=close.index)
    )
    monkeypatch.setattr(data_module, "HIT_HINK_AVAILABLE", False)
    monkeypatch.setattr(
        data_module, "batch_get_daily", lambda *a, **k: {"000001": short}
    )

    run_scan.run(["000001", "000002"])
    meta = json.loads((tmp_path / "candidates_meta.json").read_text(encoding="utf-8"))

    assert meta["failed_load_symbols"] == 1
    assert meta["insufficient_history_symbols"] == 1
    assert meta["processed_symbols"] == 0
    assert meta["accounted_symbols"] == meta["requested_symbols"] == 2


def test_fundamental_health_uses_the_same_five_checks_as_detail_view():
    from sps.fundamental import summarize_fundamental_health

    health = summarize_fundamental_health({
        "profit_yoy": 12.0,
        "revenue_yoy": -1.0,
        "roe": 15.0,
        "gross_margin": 25.0,
        "debt_ratio": 60.0,
    })

    assert health == {
        "passed": 4,
        "known": 5,
        "total": 5,
        "ratio": 0.8,
        "grade": "优秀",
    }


def test_triggered_fundamental_sort_prefers_ratio_then_completeness():
    from sps.fundamental import rank_triggered_by_fundamentals

    records = [
        {"symbol": "UNKNOWN", "score": 100},
        {"symbol": "PARTIAL", "score": 90},
        {"symbol": "FULL", "score": 80},
        {"symbol": "WEAK", "score": 70},
    ]
    snapshots = {
        "UNKNOWN": {},
        "PARTIAL": {"profit_yoy": 1.0, "roe": 12.0},
        "FULL": {
            "profit_yoy": 1.0, "revenue_yoy": 2.0, "roe": 12.0,
            "gross_margin": 30.0, "debt_ratio": 50.0,
        },
        "WEAK": {
            "profit_yoy": -1.0, "revenue_yoy": -2.0, "roe": 4.0,
            "gross_margin": 10.0, "debt_ratio": 90.0,
        },
    }

    ranked = rank_triggered_by_fundamentals(
        records, lambda symbol: snapshots[symbol]
    )

    assert [r["symbol"] for r in ranked] == ["FULL", "PARTIAL", "WEAK", "UNKNOWN"]
    assert ranked[0]["fundamental_health"]["passed"] == 5
    assert ranked[-1]["fundamental_health"]["known"] == 0


def test_screen_page_offers_fundamental_health_sort_and_card_summary():
    import scripts.app as app_module

    page = app_module.app.test_client().get("/").get_data(as_text=True)

    assert 'option value="health"' in page
    assert "基本面体检 高→低" in page
    assert "fundamental_health" in page
