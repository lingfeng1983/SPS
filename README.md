# SPS — Stock Pattern System（牛股形态识别系统）

基于《形态识别规则规格书 V1.1》实现的可回测、防未来数据泄漏的 A 股形态识别 + 参数化选股桌面系统。

**定位**：面向个人投资者的本地桌面工具 —— 形态引擎做研究回测，参数筛选器做日常选股；用户自带 LLM API Key 获取 AI 解读。

---

## 两条路线（并行存在）

| 路线 | 模块 | 用途 | 状态 |
|---|---|---|---|
| **形态引擎** | `sps/patterns.py` + `scripts/run_scan.py` | 研究/回测/分层统计 | 4 个形态已实现，原型级 |
| **参数筛选器** | `sps/screener.py` + `scripts/app.py` | 日常选股/UI/打包交付 | 18 个因子，8 个模板，已商业化交付 |

> 形态引擎的输出（`Event` 列表）与筛选器的输出（`triggered/near` 列表）是两种独立格式，不互通。当前用户产品（Web UI + 打包 exe）走筛选器路线。

---

## 目录结构

```
D:\SPS
├── docs/
│   ├── 形态识别规则规格书_v1.1.md   # 规则定义（唯一权威来源）
│   └── 使用说明-用户版.md            # 桌面版用户文档
├── sps/                              # 核心包
│   ├── data.py                       # 数据层：AKShare 拉取 + parquet 缓存
│   ├── indicators.py                 # Pivot/ATR/RPS/词典
│   ├── events.py                     # 事件契约（JSON schema）
│   ├── patterns.py                   # 形态检测器（W_BOTTOM/FLAT_BREAKOUT/CUP_HANDLE/POCKET_PIVOT）
│   ├── stats.py                      # 统计口径（含防泄漏）
│   ├── stratify.py                   # 牛熊分层
│   ├── industry.py                   # 行业映射
│   ├── positions.py                  # 持仓管理与卖出规则
│   ├── screener.py                   # 参数化指标筛选引擎（18 因子）
│   ├── ai_interpret.py               # 用户 LLM API 解读
│   ├── resume.py                     # 断点续跑
│   └── stringify.py                  # 报告序列化
├── scripts/
│   ├── app.py                        # Flask Web UI（主程序，含 PyInstaller 打包入口）
│   ├── run_scan.py                   # 命令行扫描器
│   └── build_report.py               # HTML 报告生成
├── tests/
│   ├── test_patterns.py              # 形态合成 K 线正例/反例 + 无泄漏测试
│   └── test_screener.py              # 筛选器单元测试
├── data/                             # 运行时数据（git ignored）
│   ├── daily/                        # 日线缓存（~5700 parquet）
│   ├── meta/                         # 股票列表/行业映射/指数
│   └── runs/                         # 扫描结果/报告
├── dist/SPS/                         # PyInstaller 打包产物（git ignored）
├── requirements.txt                  # pip freeze 依赖
├── SPS.spec                          # PyInstaller 配置
├── 启动SPS.bat                       # 一键启动器
└── README.md                         # 本文件
```

---

## 快速启动

### 方式 A：开发环境（推荐调试用）

```bash
cd D:\SPS
.venv\Scripts\python.exe scripts\app.py
# 浏览器访问 http://127.0.0.1:5000
# 首次使用点击右上角「⬇ 更新数据」下载行情（约 30-60 分钟）
```

### 方式 B：打包 exe（交付用户）

```bash
cd D:\SPS
.venv\Scripts\python.exe -m PyInstaller SPS.spec --noconfirm
# 产物：dist/SPS/SPS.exe（整个 dist/SPS 文件夹拷贝给用户）
# 用户双击 SPS.exe 即可，数据在 exe 同目录的 data/ 下自动生成
```

---

## 已实现功能清单

### 形态引擎（研究用）
- [x] W 底、平台突破、杯柄、口袋支点 4 个形态检测
- [x] Pivot p+k 延迟确认（防泄漏）
- [x] 可执行进场价（次日开盘 + 停牌/一字板顺延）
- [x] 同标签 20 日去重
- [x] 牛熊分层统计（`stratify.py`）
- [x] 全市场扫描 + HTML 报告

### 参数筛选器（用户产品）
- [x] 18 个技术指标 + 4 类选股条件（趋势/强势/量能/位置）
- [x] 8 个风格模板（一键选股 + 多选组合）
- [x] 模板冲突检测（互斥条件自动禁用筛选）
- [x] 参数回测徽章（`↯` 显示历史胜率/样本数）
- [x] 行业分布 Top30 + 点击筛选
- [x] 持仓管理 + 健康度红绿灯 + 卖出规则回测
- [x] 历史回放（任意日期 + 自定义范围）
- [x] K 线缩放/平移 + 买卖提示标注
- [x] AI 解读（用户自带 OpenAI 兼容 API Key）
- [x] PyInstaller 一键打包（~171MB onedir）

### 工程化
- [x] pytest 12/12 通过
- [x] git 版本控制
- [x] 依赖锁定（`requirements.txt`）
- [x] `.gitignore` 清理

---

## 防泄漏纪律（规格书 0.2/0.5/5.1 节）

| 规则 | 实现 |
|---|---|
| Pivot 只用已确认点 | `PivotView.as_of(t)` 过滤 `available_at <= t` |
| ATR 唯一入口 | `wilder_atr_series()` 统一调用，禁止内联重算 |
| 窗口参数命名化 | `min_lookback`/`confirm_window` 等字段取代魔法数字 |
| 异常不静默 | 形态 `scan` 失败打印 `[warn]` 而非 `continue` |
| 成本后收益 | 每笔往返扣 0.70%（佣金+印花税+滑点） |
| 参数哈希落盘 | `params_hash` 写入事件记录 |

---

## 路线图

### 已完成
- [x] V0.1：数据层 + Pivot 引擎 + W 底/平台突破 + 去重 + 统计
- [x] V0.5：参数化筛选器 + 8 风格模板 + 组合回测
- [x] V1.0：Web UI + 持仓管理 + 回放 + K 线标注 + AI 解读
- [x] V1.1：PyInstaller 桌面打包 + 用户文档

### 待做
- [ ] 形态引擎：补齐 VCP、头肩底等剩余形态（研究向）
- [ ] 分层统计：bootstrap 置信区间 / 聚类去相关（方法论）
- [ ] 测试：`CUP_HANDLE`/`POCKET_PIVOT` 合成 K 线正例/反例
- [ ] 基本面漏斗：PE/ROE/营收增速作为第一关过滤

---

## 依赖

```bash
pip install -r requirements.txt
```

核心包：`akshare`（行情）、`pandas`、`flask`、`pyarrow`（parquet）、`pyinstaller`（打包）

---

## 免责声明

仅供个人研究学习，不构成投资建议。任何历史回测胜率都不代表未来表现。
