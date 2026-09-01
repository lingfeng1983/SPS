# -*- coding: utf-8 -*-
"""用户自带 LLM API 的 AI 解读模块。

配置存本地 data/meta/ai_config.json（API Key 永不上传，仅本机使用）。
兼容所有 OpenAI 兼容接口（OpenAI / DeepSeek / 通义 / Kimi / GLM / 本地 Ollama 等）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from sps.data import DATA_DIR

CFG_FILE = DATA_DIR / "meta" / "ai_config.json"

DEFAULT_CFG = {
    "base_url": "https://api.openai.com/v1",
    "api_key": "",
    "model": "gpt-4o-mini",
    "temperature": 0.4,
    "max_tokens": 1500,
}

PROMPT_SYSTEM = (
    "你是一名严谨的A股技术面分析助手。用户会给你一组基于技术指标筛选出的股票"
    "（含满足的条件、买点、止损价、历史回测胜率）。"
    "请从以下角度输出简明解读（总长控制在400字内）：\n"
    "1. 整体观察：这批标的的共同特征（集中在什么形态/行业，说明市场当下什么风格占优）\n"
    "2. 重点提示：哪几只值得优先关注、为什么（结合买点与止损的距离）\n"
    "3. 风险提醒：这组条件的适用环境、当前可能失效的情形\n"
    "要求：只基于给出的数据，不要编造未提供的财务或消息面信息；语言通俗，不堆术语。"
)


def load_config() -> dict:
    if CFG_FILE.exists():
        try:
            cfg = json.loads(CFG_FILE.read_text(encoding="utf-8"))
            return {**DEFAULT_CFG, **cfg}
        except Exception:
            pass
    return dict(DEFAULT_CFG)


def save_config(cfg: dict) -> None:
    CFG_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged = {**load_config(), **{k: v for k, v in cfg.items() if k in DEFAULT_CFG}}
    CFG_FILE.write_text(json.dumps(merged, ensure_ascii=False, indent=2),
                        encoding="utf-8")


def masked_config() -> dict:
    """返回给前端的配置（key 打码）。"""
    cfg = load_config()
    key = cfg.get("api_key") or ""
    cfg["api_key_set"] = bool(key)
    cfg["api_key_masked"] = (key[:6] + "…" + key[-4:]) if len(key) > 12 else ("已设置" if key else "")
    cfg.pop("api_key", None)
    return cfg


def _chat(messages: list[dict], cfg: dict, timeout: int = 90) -> str:
    """调用 OpenAI 兼容 chat/completions。"""
    import urllib.request

    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": cfg["model"],
        "messages": messages,
        "temperature": float(cfg.get("temperature", 0.4)),
        "max_tokens": int(cfg.get("max_tokens", 1500)),
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {cfg['api_key']}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def screen_interpretation(screen_result: dict) -> dict:
    """把筛选结果交给 LLM 解读。返回 {ok, text, model, elapsed} 或 {ok:False, error}。"""
    cfg = load_config()
    if not cfg.get("api_key"):
        return {"ok": False, "error": "尚未配置 API Key，请先点击右上角 ⚙ 设置"}

    trig = screen_result.get("triggered") or []
    if not trig:
        return {"ok": False, "error": "当前没有已触发标的，无需解读"}

    # 摘要：最多20只，保留关键字段
    items = []
    for r in trig[:20]:
        items.append({
            "代码": r.get("symbol"), "名称": r.get("_name", ""),
            "满足条件": r.get("met", []), "买点": r.get("entry_price"),
            "止损": r.get("stop_price"),
        })
    summary = json.dumps({
        "扫描股票数": screen_result.get("scanned"),
        "触发数量": len(trig),
        "行业分布": (screen_result.get("industry_summary") or [])[:8],
        "标的": items,
    }, ensure_ascii=False)

    t0 = time.time()
    try:
        text = _chat(
            [{"role": "system", "content": PROMPT_SYSTEM},
             {"role": "user", "content": f"筛选结果如下：\n{summary}"}],
            cfg)
        return {"ok": True, "text": text, "model": cfg["model"],
                "elapsed": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"AI 调用失败：{e}"}


def test_connection(cfg: dict | None = None) -> dict:
    """测试连通性。"""
    cfg = cfg or load_config()
    if not cfg.get("api_key"):
        return {"ok": False, "error": "API Key 为空"}
    t0 = time.time()
    try:
        text = _chat([{"role": "user", "content": "回复：OK"}], cfg, timeout=30)
        return {"ok": True, "reply": text[:50], "elapsed": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}
