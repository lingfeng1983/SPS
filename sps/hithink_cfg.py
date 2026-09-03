# -*- coding: utf-8 -*-
"""HiThink Financial API 配置管理。

存储：data/meta/hithink_config.json
API Key 同时写入 HiThink CLI keychain（auth login --replace --api-key-stdin），
让 sps/data.py 的混合数据源能自动识别。
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from sps.data import DATA_DIR

CFG_FILE = DATA_DIR / "meta" / "hithink_config.json"


def get_hithink_cfg() -> dict:
    """返回 HiThink 配置（带 key_set 标记）。"""
    cfg: dict = {"api_key_set": False}
    if CFG_FILE.exists():
        try:
            cfg = {**cfg, **json.loads(CFG_FILE.read_text(encoding="utf-8"))}
        except Exception:
            pass
    key = cfg.get("api_key", "")
    cfg["api_key_set"] = bool(key)
    # 不返回明文 key，只返回标记
    if key:
        cfg["api_key_masked"] = (key[:6] + "…" + key[-4:]) if len(key) > 12 else "已设置"
    else:
        cfg["api_key_masked"] = ""
    cfg.pop("api_key", None)
    return cfg


def save_hithink_cfg(body: dict) -> None:
    """保存配置到文件 + 同步到 HiThink CLI keychain。"""
    api_key = (body.get("api_key") or "").strip()
    if not api_key:
        return

    # 1. 保存到本地 json
    CFG_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"api_key": api_key}
    CFG_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    # 2. 同步到 HiThink CLI keychain
    try:
        subprocess.run(
            ["hithink-finance", "auth", "login", "--replace", "--api-key-stdin"],
            input=api_key.encode("utf-8"),
            timeout=15,
            check=False,
        )
    except Exception:
        pass  # CLI 不可用时忽略


def test_hithink(api_key: str = "") -> dict:
    """测试 HiThink API Key 是否可用。"""
    if not api_key:
        # 用已存储的 key 测试
        if CFG_FILE.exists():
            try:
                stored = json.loads(CFG_FILE.read_text(encoding="utf-8"))
                api_key = stored.get("api_key", "")
            except Exception:
                pass
    if not api_key:
        return {"ok": False, "error": "API Key 为空"}

    # 临时写入 CLI 测试
    try:
        subprocess.run(
            ["hithink-finance", "auth", "login", "--replace", "--api-key-stdin"],
            input=api_key.encode("utf-8"),
            timeout=15,
            check=False,
        )
    except Exception:
        pass

    # 调用一次简单查询验证
    try:
        import time as _t
        t0 = _t.time()
        result = subprocess.run(
            ["hithink-finance", "market", "history",
             "--thscode", "000001.SZ", "--format", "json"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        elapsed = round(_t.time() - t0, 1)
        if result.returncode == 0 and "1991" in result.stdout:
            # 能找到 1991 年数据说明 key 有效
            return {"ok": True, "elapsed": elapsed,
                    "reply": "可获取行情数据（1991 至今）"}
        if result.returncode != 0:
            err = result.stderr.strip().split("\n")[-1][:100]
            return {"ok": False, "error": err or "返回非零状态码"}
        return {"ok": True, "elapsed": elapsed, "reply": "连接正常"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "超时（30s）"}
    except FileNotFoundError:
        return {"ok": False, "error": "HiThink CLI 未安装（npm install -g @hithink-tech/hithink-finance-cli）"}
    except Exception as e:
        return {"ok": False, "error": str(e)[:100]}
