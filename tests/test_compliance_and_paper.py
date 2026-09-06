"""合规与模拟盘功能的回归测试：免责声明、应用配置、反馈、新手门槛、复盘报告。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """隔离 data 目录的 Flask test client。"""
    import app as app_mod
    import sps.positions as pos_mod

    data_dir = tmp_path / "data"
    (data_dir / "daily").mkdir(parents=True)
    monkeypatch.setattr(app_mod, "DISCLAIMER_FILE", data_dir / "meta" / "disclaimer_accepted.json")
    monkeypatch.setattr(app_mod, "APP_CFG_FILE", data_dir / "meta" / "app_config.json")
    monkeypatch.setattr(app_mod, "RUN_DIR", data_dir / "runs")
    monkeypatch.setattr(pos_mod, "POS_FILE", data_dir / "positions.json")
    return app_mod.app.test_client()


def test_disclaimer_flow(client):
    r = client.get("/api/disclaimer")
    assert r.status_code == 200
    body = r.get_json()
    assert body["accepted"] is False
    assert "不构成任何证券投资建议" in body["text"]

    r = client.post("/api/disclaimer")
    assert r.status_code == 200

    body = client.get("/api/disclaimer").get_json()
    assert body["accepted"] is True


def test_app_config_roundtrip_and_validation(client):
    body = client.post("/api/app_config", json={"auto_update_enabled": True,
                                                "auto_update_time": "18:00"}).get_json()
    assert body["auto_update_enabled"] is True
    assert body["auto_update_time"] == "18:00"
    # 非法字段被忽略，白名单外不落盘
    body = client.post("/api/app_config", json={"evil": "x"}).get_json()
    assert "evil" not in body
    body = client.get("/api/app_config").get_json()
    assert body["auto_update_enabled"] is True


def test_feedback_requires_text_and_persists(client):
    r = client.post("/api/feedback", json={"text": "  "})
    assert r.status_code == 400

    r = client.post("/api/feedback", json={"text": "筛选结果为空", "contact": "wx-123"})
    assert r.status_code == 200
    import app as app_mod
    log = app_mod.RUN_DIR / "feedback.log"
    assert log.exists()
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert entry["text"] == "筛选结果为空"
    assert entry["contact"] == "wx-123"
    assert "data_status" in entry


def test_paper_gate_and_review_report(client):
    import app as app_mod
    from sps.positions import add_position, close_position

    body = client.get("/api/paper_gate").get_json()
    assert body == {"paper_records": 0, "novice": True}

    add_position("600519", "贵州茅台", 1600.0, "2026-08-01",
                 stop_pct=7.0, rules={"stop_loss": True}, mode="paper")
    add_position("300750", "宁德时代", 180.0, "2026-08-02",
                 stop_pct=7.0, rules={"break_ma": 10}, mode="live")
    close_position("600519", 1700.0, "2026-08-15", reason="manual")

    body = client.get("/api/paper_gate").get_json()
    assert body == {"paper_records": 1, "novice": True}   # 实盘仓不计入门槛；满20笔前保持新手模式

    rep = client.get("/api/review_report?days=30").get_json()
    # 复盘只统计模拟仓：实盘的宁德时代不进入
    assert rep["n_closed"] == 1
    assert rep["n_open"] == 0
    assert rep["n_total"] == 1   # n_total 只含模拟仓，实盘的宁德时代被排除
    assert rep["win_rate"] == 100.0
    assert rep["exit_reasons"] == {"manual": 1}
    # 实盘仓被排除在复盘之外
    add_position("000001", "平安银行", 10.0, "2026-08-20",
                 stop_pct=5.0, rules={"stop_loss": True}, mode="paper")
    rep = client.get("/api/review_report").get_json()
    assert rep["n_open"] == 1
    assert rep["open_positions"][0]["symbol"] == "000001"


def test_add_position_rejects_unknown_mode(client):
    r = client.post("/api/positions", json={
        "symbol": "600519", "entry_price": 1600.0, "entry_date": "2026-08-01",
        "stop_pct": 7.0, "rules": {"stop_loss": True}, "mode": "magic"})
    assert r.status_code == 400
    assert "未知持仓模式" in r.get_json()["error"]
