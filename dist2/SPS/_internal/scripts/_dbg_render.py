"""检查 indForm 渲染逻辑：直接模拟 loadIndicators 的关键路径。"""
import json
import urllib.request

# 1) API 返回结构
d = json.load(urllib.request.urlopen("http://127.0.0.1:5000/api/indicators"))
inds = d["indicators"]
print("API 指标数:", len(inds))

ps = json.load(urllib.request.urlopen("http://127.0.0.1:5000/api/param_stats"))
print("param_stats ready:", ps.get("ready"), "| stats keys:", len(ps.get("stats", {})))

# 2) 复现 JS 里的 byCat 分组 + indRowHtml 字符串构造（Python 版）
byCat = {}
for i in inds:
    byCat.setdefault(i["category"], []).append(i)

parts = []
for cat, lst in byCat.items():
    parts.append(f"<div>{cat}</div>")
    for ind in lst:
        arr = isinstance(ind["default"], list)
        d0 = ind["default"][0] if arr else ind["default"]
        # btBadge 逻辑：找最近档位
        badge = ""
        g = (ps.get("stats") or {}).get(ind["name"])
        if g:
            keys = sorted(g.keys(), key=lambda k: abs(float(k) - float(d0)) if not arr else 0)
            rec = g[keys[0]]
            badge = f"badge win20={rec['win20']}"
        row = f"ROW[{ind['name']}] label={ind['label']} default={d0} {badge}"
        parts.append(row)

out = "".join(parts)
print("\n渲染行数:", sum(1 for p in parts if p.startswith('ROW')))
print("\n".join(parts[:8]))
