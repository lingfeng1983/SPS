# -*- mode: python ; coding: utf-8 -*-
# SPS 桌面版打包配置：PyInstaller onedir 模式
# 构建：.venv/Scripts/python.exe -m PyInstaller SPS.spec --noconfirm
# 产物：dist/SPS/SPS.exe（整个 dist/SPS 文件夹即交付物）

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# akshare 内置交易日历等 json 资源必须随包分发，否则运行时报 calendar.json 缺失
akshare_datas = collect_data_files('akshare')
# py_mini_racer(akshare 指数接口依赖) 的 mini_racer.dll 必须收集，否则 Native library 报错
mini_racer_bins = collect_dynamic_libs('py_mini_racer')
# py_mini_racer 还需要 icudtl.dat 与 snapshot_blob.bin（不在 dynamic_libs 里）
mini_racer_datas = collect_data_files('py_mini_racer')

a = Analysis(
    ['scripts/app.py'],
    pathex=[str(Path.cwd())],
    binaries=[
        *mini_racer_bins,
    ],
    datas=[
        # 资源文件：akshare、py_mini_racer
        *akshare_datas,
        *collect_data_files('py_mini_racer'),
        # 模板与静态资源随包分发；用户数据(data/)运行时在 exe 同目录生成
        ('sps', 'sps'),
        ('scripts', 'scripts'),
    ],
    hiddenimports=[
        'flask', 'jinja2', 'pandas', 'numpy', 'pyarrow',
        'akshare', 'requests', 'urllib3', 'certifi', 'openpyxl',
        'akshare.fund.fund_etf_fund_em', 'akshare.stock_feature',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'scipy', 'torch', 'transformers'],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SPS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='SPS',
)