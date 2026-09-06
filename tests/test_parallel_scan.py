"""并行形态检测的等价性回归：分片多进程结果必须与单进程完全一致。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.run_scan as run_scan


def _make_universe(data_dir: Path, n_syms: int = 42, periods: int = 260):
    """合成日线缓存 + 基本面缓存（子进程读缓存，不碰网络）。"""
    daily = data_dir / "daily"
    daily.mkdir(parents=True)
    (data_dir / "fundamental").mkdir(exist_ok=True)
    (data_dir / "runs").mkdir(exist_ok=True)
    idx = pd.bdate_range("2024-01-01", periods=periods)
    rng = np.random.default_rng(7)
    syms = []
    for i in range(n_syms):
        sym = f"{600000 + i}"
        syms.append(sym)
        drift = 0.001 if i % 5 == 0 else -0.0002
        ret = rng.normal(drift, 0.02, periods)
        c = 10 * np.exp(np.cumsum(ret))
        o = c * (1 + rng.normal(0, 0.005, periods))
        h = np.maximum(o, c) * (1 + abs(rng.normal(0, 0.006, periods)))
        l = np.minimum(o, c) * (1 - abs(rng.normal(0, 0.006, periods)))
        v = rng.lognormal(14, 0.3, periods)
        pd.DataFrame({"O": o, "H": h, "L": l, "C": c, "V": v},
                     index=idx).to_parquet(daily / f"{sym}_20190101_latest.parquet")
        (data_dir / "fundamental" / f"{sym}.json").write_text(
            json.dumps({"roe": 12.0, "debt_ratio": 40.0, "net_profit": 1e8,
                        "deduct_np": 5e7}),
            encoding="utf-8")
    return syms, idx


def _run_scan(syms, idx, monkeypatch, data_dir: Path, workers: str):
    monkeypatch.setenv("SPS_SCAN_WORKERS", workers)
    kind_map = {s: "stock" for s in syms}

    def fake_batch(*_a, **_k):
        return {s: pd.read_parquet(data_dir / "daily" / f"{s}_20190101_latest.parquet")
                for s in syms}

    import sps.data as data_module
    import sps.fundamental as fundamental_module
    monkeypatch.setattr(data_module, "batch_get_daily", fake_batch)
    monkeypatch.setattr(data_module, "hithink_available", lambda: False)
    # 父进程也要指向合成基本面缓存，否则串行轮读真实数据、并行轮读合成数据
    monkeypatch.setattr(fundamental_module, "FUND_DIR", data_dir / "fundamental")
    monkeypatch.setattr(run_scan, "DATA_DIR", data_dir)
    monkeypatch.setattr(run_scan, "OUT_DIR", data_dir / "runs")
    monkeypatch.setattr(run_scan, "get_index", lambda *a, **k: pd.DataFrame(
        {"C": np.linspace(3000, 3500, len(idx))}, index=idx))

    monkeypatch.setenv("SPS_DATA_DIR", str(data_dir))   # 子进程据此解析数据目录
    return run_scan.run(list(syms), kind_map=kind_map)


def _canonical(cands):
    return sorted(json.dumps(e, sort_keys=True, default=str) for e in cands)


def test_parallel_scan_matches_serial(tmp_path, monkeypatch):
    syms, idx = _make_universe(tmp_path)

    monkeypatch.setenv("SPS_SCAN_WORKERS", "0")
    serial = _run_scan(syms, idx, monkeypatch, tmp_path, workers="0")
    parallel = _run_scan(syms, idx, monkeypatch, tmp_path, workers="3")

    # 两条路径产出的候选（含评分、进场价、止损、环境适配）必须完全一致
    assert _canonical(serial) == _canonical(parallel)
    assert len(serial) > 0, "合成数据应至少产生一个候选，否则对比无意义"

    # 并行路径的覆盖元数据必须完整（哈希 sidecar 校验通过）
    meta_file = tmp_path / "runs" / "candidates_meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["coverage_known"] is True
    assert meta["accounted_symbols"] == len(syms)
    assert meta["candidate_count"] == len(parallel)


def test_run_cancelled_between_phases(tmp_path, monkeypatch):
    """协作式取消：阶段边界触发 should_cancel 时抛 JobCancelled，
    不产出候选文件（已完成的数据写入保持有效）。"""
    import pytest
    syms, idx = _make_universe(tmp_path)
    monkeypatch.setenv("SPS_SCAN_WORKERS", "0")

    def fake_batch(*_a, **_k):
        return {s_: pd.read_parquet(tmp_path / "daily" / f"{s_}_20190101_latest.parquet")
                for s_ in syms}

    import sps.data as data_module
    monkeypatch.setattr(data_module, "batch_get_daily", fake_batch)
    monkeypatch.setattr(data_module, "hithink_available", lambda: False)
    monkeypatch.setattr(run_scan, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_scan, "OUT_DIR", tmp_path / "runs")
    monkeypatch.setattr(run_scan, "get_index", lambda *a, **k: pd.DataFrame(
        {"C": np.linspace(3000, 3500, len(idx))}, index=idx))

    with pytest.raises(run_scan.JobCancelled):
        run_scan.run(list(syms), kind_map={s_: "stock" for s_ in syms},
                     should_cancel=lambda: True)
    assert not (tmp_path / "runs" / "candidates.json").exists()
