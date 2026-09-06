"""Candidate artifact publication, validation, and contract migration."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = 2
STOP_CONTRACT = "thresholds-v2"


def normalize_candidate(candidate: dict) -> dict:
    """Convert an unversioned legacy stop contract without changing modern data."""
    normalized = copy.deepcopy(candidate)
    entry = normalized.get("entry")
    if not isinstance(entry, dict):
        return normalized
    if entry.get("stop_contract") == STOP_CONTRACT:
        return normalized
    if entry.get("stop_contract") is not None:
        return normalized
    try:
        price = float(entry.get("price"))
    except (TypeError, ValueError):
        return normalized
    stops = entry.get("stops")
    if price <= 0 or not isinstance(stops, dict):
        return normalized

    if "stop_exits" not in entry:
        entry["stop_exits"] = copy.deepcopy(stops)
    fixed = {}
    for key in stops:
        try:
            level = float(key)
        except (TypeError, ValueError):
            continue
        fixed[str(key)] = round(price * (1 + level), 4)
    entry["stops"] = fixed
    entry["stop_contract"] = STOP_CONTRACT
    return normalized


def _modernize_for_write(candidate: dict) -> dict:
    record = copy.deepcopy(candidate)
    entry = record.get("entry")
    if isinstance(entry, dict) and isinstance(entry.get("stops"), dict):
        entry.setdefault("stop_contract", STOP_CONTRACT)
    return record


def _encoded(records: list[dict]) -> bytes:
    return json.dumps(records, ensure_ascii=False, indent=1).encode("utf-8")


def _meta_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_meta.json")


def write_candidate_artifacts(path: Path, records: list[dict], meta: dict) -> dict:
    """Publish candidates plus a hash-bound sidecar; readers reject torn pairs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [_modernize_for_write(record) for record in records]
    payload = _encoded(records)
    digest = hashlib.sha256(payload).hexdigest()
    result = dict(meta)
    result.update({
        "schema_version": SCHEMA_VERSION,
        "run_id": result.get("run_id") or uuid.uuid4().hex,
        "candidates_sha256": digest,
        "coverage_known": bool(result.get("coverage_known", True)),
        "candidate_count": len(records),
        "candidate_symbols": len({r.get("symbol", "") for r in records}),
        "latest_signal_date": max(
            (r.get("signal_date") or "" for r in records), default=""
        ),
    })
    meta_payload = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
    candidate_tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    sidecar = _meta_path(path)
    meta_tmp = sidecar.with_suffix(sidecar.suffix + f".{uuid.uuid4().hex}.tmp")
    candidate_tmp.write_bytes(payload)
    meta_tmp.write_bytes(meta_payload)
    candidate_tmp.replace(path)
    meta_tmp.replace(sidecar)
    return result


def load_candidates_metadata(path: Path, records: list[dict]) -> dict:
    """Return trusted sidecar metadata or a conservative file-derived fallback."""
    sidecar = _meta_path(path)
    meta = {}
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        pass
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""
    valid = (
        meta.get("schema_version") == SCHEMA_VERSION
        and meta.get("candidates_sha256") == digest
    )
    if not valid:
        meta = {
            "generated_at": (
                datetime.fromtimestamp(path.stat().st_mtime).astimezone()
                .isoformat(timespec="seconds") if path.exists() else ""
            ),
            "coverage_known": False,
        }
    meta.update({
        "artifact_valid": valid,
        "candidate_count": len(records),
        "candidate_symbols": len({r.get("symbol", "") for r in records}),
        "latest_signal_date": max(
            (r.get("signal_date") or "" for r in records), default=""
        ),
    })
    return meta


def migrate_candidates_file(path: Path) -> dict:
    """Upgrade an old artifact once, retaining its original and timestamp."""
    if not path.exists():
        return {"changed": False, "backup": None, "count": 0}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("candidates artifact must be a JSON list")
    original_stat = path.stat()
    normalized = [normalize_candidate(record) for record in raw]
    sidecar = _meta_path(path)
    sidecar_version = None
    try:
        sidecar_version = json.loads(sidecar.read_text(encoding="utf-8")).get(
            "schema_version"
        )
    except (OSError, ValueError, TypeError):
        pass
    candidate_changed = normalized != raw
    needs_schema = sidecar_version != SCHEMA_VERSION
    if not candidate_changed and not needs_schema:
        return {"changed": False, "backup": None, "count": len(raw)}

    backup = None
    if candidate_changed:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = path.with_name(f"{path.stem}.legacy-{stamp}{path.suffix}")
        shutil.copy2(path, backup)
    write_candidate_artifacts(path, normalized, {
        "generated_at": datetime.fromtimestamp(original_stat.st_mtime).astimezone()
        .isoformat(timespec="seconds"),
        "coverage_known": False,
    })
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    return {"changed": candidate_changed, "backup": str(backup) if backup else None,
            "count": len(normalized)}


def load_candidates_file(path: Path) -> list[dict]:
    if not path.exists():
        return []
    migrate_candidates_file(path)
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("candidates artifact must be a JSON list")
    return [normalize_candidate(record) for record in records]
