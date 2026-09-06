"""数据目录解析（全项目唯一来源）。

优先级：SPS_DATA_DIR 环境变量 → PyInstaller exe 同目录/data → 开发模式项目根/data。
其余模块一律从这里 import DATA_DIR，禁止各自用 Path(__file__) 硬编码——
否则打包版数据会分裂在 _internal/ 与 exe 同目录两处。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def resolve_data_dir() -> Path:
    configured = os.environ.get("SPS_DATA_DIR")
    if configured:
        return Path(configured).resolve()
    if getattr(sys, "frozen", False):          # PyInstaller 打包环境
        return Path(sys.executable).resolve().parent / "data"
    return Path(__file__).resolve().parent.parent / "data"


DATA_DIR = resolve_data_dir()
