"""HTML 报告生成：候选标的清单优先 + 统计附后。"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent.parent / "data" / "runs"


def build_report(out: Path | None = None) -> Path:
    ev_path = RUN_DIR / "events.jsonl"
    events = [json.loads(l) for l in open(ev_path, encoding="utf-8")] \
        if ev_path.exists() else []

    cand_path = RUN_DIR / "candidates.json"
    cands = json.loads(cand_path.read_text(encoding="utf-8")) \
        if cand_path.exists() else []

    # 候选清单卡片
    cards = ""
    for e in cands[:60]:
        en = e.get("entry", {})
        stops = en.get("stops", {})
        kl = e.get("key_levels", {})
        ft = e.get("features", {})
        fit_cls = {"good": "g", "neutral": "n", "bad": "b"}.get(
            e.get("env_fit"), "n")
        feat = " · ".join(f"{k}={v}" for k, v in ft.items()) or "-"
        lvl = " · ".join(f"{k}={v}" for k, v in kl.items()) or "-"
        cards += f"""
<div class="card">
  <div class="card-h"><b>{e['symbol']}</b> {e['pattern']}
    <span class="tag {fit_cls}">{e.get('env_fit')}</span>
    <span class="score">{e.get('score',0)}</span></div>
  <div class="card-b">
   信号日 {e.get('signal_date')} · RPS50={e.get('rps50')} · 环境={e.get('regime')}<br>
   进场 {en.get('price')} @ {en.get('date')} · 止损 -7%={stops.get('-0.07')}<br>
   关键位：{lvl}<br>特征：{feat}
  </div>
</div>"""

    # 统计表（来自 stratified）
    stats_html = ""
    for f in sorted(RUN_DIR.glob("stats_*.csv")):
        import csv
        rows = list(csv.reader(open(f, encoding="utf-8")))
        if len(rows) < 2:
            continue
        pat = f.stem.replace("stats_", "")
        body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                       for r in rows[1:])
        head = "".join(f"<th>{c}</th>" for c in rows[0])
        stats_html += f"<h3>{pat}</h3><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>SPS 候选标的报告</title>
<style>
 body{{font-family:'Segoe UI',system-ui,sans-serif;max-width:1100px;margin:24px auto;
      padding:0 16px;color:#1a1a2e;background:#fafafa}}
 h1{{border-bottom:3px solid #4361ee;padding-bottom:8px}}
 h2{{color:#3a0ca3;margin-top:24px}} h3{{color:#555;margin:14px 0 6px}}
 table{{border-collapse:collapse;width:100%;font-size:13px;background:#fff}}
 th,td{{border:1px solid #ddd;padding:5px 9px;text-align:right}}
 th{{background:#eef2ff}}
 .card{{background:#fff;border:1px solid #e5e5ea;border-left:4px solid #4361ee;
       border-radius:6px;margin:8px 0;padding:9px 13px;box-shadow:0 1px 3px rgba(0,0,0,.05)}}
 .card-h{{font-size:15px}} .card-b{{font-size:13px;color:#555;line-height:1.7}}
 .tag{{font-size:11px;padding:2px 8px;border-radius:10px;color:#fff;margin-left:8px}}
 .g{{background:#2a9d8f}} .n{{background:#e9a23b}} .b{{background:#d62828}}
 .score{{float:right;font-weight:700;font-size:20px;color:#4361ee}}
 .meta{{color:#777;font-size:13px}}
</style></head><body>
<h1>SPS 牛股形态候选清单</h1>
<p class="meta">生成 {date.today()} · 规则v1.1 · 进场=信号次日开盘 ·
评分=环境适配(good100/neutral50/bad0)+RPS强度(≤30) · 仅列确认且未重复的标的</p>

<h2>候选标的（按评分降序，前 {len(cands[:60])} 个）</h2>
{cards or '<p class="meta">无候选</p>'}

<h2>历史统计（可执行口径）</h2>
{stats_html or '<p class="meta">暂无</p>'}

<p class="meta">⚠️ 自动生成，仅供研究，不构成投资建议。评分仅反映形态×环境×强度匹配度，非收益承诺。</p>
</body></html>"""
    out = out or RUN_DIR / "report.html"
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build_report())
