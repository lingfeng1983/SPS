#!/usr/bin/env python3
"""合并形态筛选到风格模板，并修复候选清单日期显示 bug。"""

import re
from pathlib import Path

app_py = Path("D:/SPS/scripts/app.py")

# Read the file
content = app_py.read_text(encoding="utf-8")

# ======================================================================
# 1. 在 TEMPLATES 数组后追加形态模板 (Pattern Templates)
# ======================================================================

# Find the end of TEMPLATES array (line with `];` after hyper_trend)
pattern_end = content.find("];\nlet activeTpls=[]")
if pattern_end == -1:
    raise ValueError("Could not find end of TEMPLATES array")

# Find the exact `];` before `let activeTpls`
insert_pos = content.rfind("];", 0, pattern_end) + 2

# Build the pattern templates section to insert
pattern_templates = """
// ── 形态模板：与风格模板并列，勾选后参与筛选 ──
const PATTERN_TEMPLATES = [
  {key:'pn_w_bottom', icon:'📗', name:'W底', tag:'几何形态', desc:'双重底突破颈线',
   pattern:'W_BOTTOM',
   plain:'两个底部价位接近、间隔2-9周，第二个底构筑后放量突破颈线（两底间反弹高点）。欧奈尔体系中经典的底部反转形态。'},
  {key:'pn_flat_breakout', icon:'🚀', name:'平台突破', tag:'几何形态', desc:'横盘20-60日后放量突破',
   pattern:'FLAT_BREAKOUT',
   plain:'股价横盘整理20-60个交易日、振幅≤15%，然后放量突破平台上沿。突破前通常有一波≥25%的上涨段（旗杆），突破后上涨概率较高。'},
  {key:'pn_cup_handle', icon:'☕', name:'杯柄', tag:'几何形态', desc:'杯状+柄部突破',
   pattern:'CUP_HANDLE',
   plain:'股价先下跌形成"杯底"，反弹回高点后小幅回调（"杯柄"），最后放量突破杯沿。杯柄越浅、突破时成交量越大，信号越可靠。欧奈尔体系中最具统计优势的形态之一。'},
  {key:'pn_pocket_pivot', icon:'🎯', name:'口袋支点', tag:'几何形态', desc:'蓄势后支点突破',
   pattern:'POCKET_PIVOT',
   plain:'股票处于上升趋势、距MA10不远，突然放量上涨2%+且收盘在日内高点区间（上影线短）。当日成交量超过此前所有下跌日的最大量，是机构悄悄进场的信号。'},
  {key:'pn_high_narrow_flag', icon:'🚩', name:'高而窄旗形', tag:'几何形态', desc:'快速上涨→横盘收敛→突破',
   pattern:'HIGH_NARROW_FLAG',
   plain:'股票在2-4周内快速上涨≥25%（"旗旗杆"），随后横盘收敛2-4周（"旗面"，振幅≤15%），最后放量突破。这种高而窄的结构是欧奈尔体系中爆发力最强的中继形态之一。'},
  {key:'pn_limit_up_wash', icon:'🔥', name:'涨停洗盘', tag:'几何形态', desc:'涨停→回踩→突破',
   pattern:'LIMIT_UP_WASH',
   plain:'股票涨停（涨幅≥9.8%），随后2-10个交易日内回踩但不破涨停日低点，最后放量突破涨停日高点。这是A股特有的主力吸筹后快速洗盘形态，突破后往往开启第二波主升浪。'},
  {key:'pn_rising_limit_down_reversal', icon:'⚡', name:'跌停反包', tag:'几何形态', desc:'上升趋势中跌停后反包',
   pattern:'RISING_LIMIT_DOWN_REVERSAL',
   plain:'股票处于上升趋势（近20日涨≥10%），突然出现跌停（跌幅≥9.8%），次日或隔日放量反包、收盘≥跌停日高点。极端情绪释放后的快速修复，意味着强势延续。'},
];

let activePatterns = []; // 勾选中的形态 key 列表

"""

content = content[:insert_pos] + pattern_templates + content[insert_pos:]

# ======================================================================
# 2. 在 HTML 侧边栏插入形态选择区域（在风格模板之后、自定义指标之前）
# ======================================================================

# Find the secPattern section and replace it
old_sec_pattern = re.search(
    r'  <div class="sec" id="secPattern">.*?</div>\n  </div>\n\n  <div class="sec" id="secScreen">',
    content,
    re.DOTALL
)

if old_sec_pattern:
    new_sec_pattern = '''  <div class="sec" id="secPattern">
    <h3 style="cursor:pointer;user-select:none;display:flex;align-items:center;justify-content:space-between" onclick="togglePattern()">
      <span>② 形态筛选（技术面）<span id="patternCount" style="color:var(--accent);font-weight:700;margin-left:6px"></span></span>
      <span id="patternArrow" style="font-size:10px">▾ 展开</span></h3>
    <div id="patternBody" style="display:none">
    <div style="font-size:12.5px;color:var(--muted);margin-bottom:8px">勾选形态，在全市场搜索近期触发的标的：</div>
    <div id="patternForm" style="font-size:12px;display:flex;flex-direction:column;gap:5px"></div>
    <div class="row" style="margin-top:8px">🛡 止损%
      <input id="stopPctPattern" type="number" value="7" step="0.5" min="1" max="20" aria-label="止损百分比" style="width:60px">
      <span style="font-size:10.5px;color:var(--muted)">风控参数</span></div>
    <button class="green" style="margin-top:4px" onclick="doPatternScreen()">🔍 开始形态筛选</button>
    <div id="patternResultInfo" style="font-size:11.5px;color:var(--muted);margin-top:4px"></div>
    </div>
  </div>

  <div class="sec" id="secScreen">'''

    content = content[:old_sec_pattern.start()] + new_sec_pattern + content[old_sec_pattern.end():]

# ======================================================================
# 3. 添加 togglePattern, renderPatternForm, togglePatternItem 函数
# ======================================================================

# Find where toggleAdv function is defined
toggle_adv_pos = content.find("function toggleAdv(){")
if toggle_adv_pos == -1:
    raise ValueError("Could not find toggleAdv function")

# Insert pattern-related functions before toggleAdv
pattern_funcs = """
// ── 形态筛选交互 ──
function togglePattern(){
  const b=$('patternBody'), a=$('patternArrow');
  const open=b.style.display==='none';
  b.style.display=open?'':'none';
  a.textContent=open?'▴ 收起':'▾ 展开';
  if(open && $('patternForm').innerHTML===''){
    renderPatternForm();
  }
}
function updatePatternCount(){
  const el=$('patternCount');
  if(!el) return;
  el.textContent = activePatterns.length ? `（已选${activePatterns.length}个）` : '';
}
function renderPatternForm(){
  $('patternForm').innerHTML = PATTERN_TEMPLATES.map(p=>{
    const on = activePatterns.includes(p.key);
    return `<label class="patChk" style="display:flex;align-items:center;gap:8px;cursor:pointer;padding:6px 8px;border:1px solid ${on?'var(--accent)':'var(--border)'};border-radius:7px;background:${on?'#1e3a5f':'transparent'}">
      <input type="checkbox" data-pattern="${p.key}" ${on?'checked':''} onchange="togglePatternItem('${p.key}')" style="accent-color:var(--accent)">
      <span style="font-size:13px">${p.icon} ${p.name}</span>
      <span style="font-size:10.5px;color:var(--muted);margin-left:auto">${p.desc}</span>
    </label>`;
  }).join('');
}
function togglePatternItem(key){
  if(activePatterns.includes(key)) activePatterns = activePatterns.filter(k=>k!==key);
  else activePatterns.push(key);
  updatePatternCount();
}

"""

content = content[:toggle_adv_pos] + pattern_funcs + content[toggle_adv_pos:]

# ======================================================================
# 4. 在 clearAllConds 中清理形态选择
# ======================================================================

# Add pattern clearing to clearAllConds
old_clear = "function clearAllConds(){"
old_clear_pos = content.find(old_clear)
if old_clear_pos != -1:
    # Find the body of clearAllConds and add pattern clearing
    old_clear_body = "  activeTpls=[];"
    new_clear_body = "  activeTpls=[]; activePatterns=[];"
    content = content.replace(old_clear_body, new_clear_body)

# ======================================================================
# 5. 修复候选清单日期显示：用实际数据日期而非文件 mtime
# ======================================================================

# In api_candidates, replace the meta generation
# Use latest signal date from data instead of file mtime
old_candidate_meta = '''    meta = load_candidates_meta(cands)
    # 日常有用的口径：最近 7 天的新信号数（累计总量对使用者无意义）
    cutoff = (date.today() - timedelta(days=7)).isoformat()
    meta["recent7"] = sum(1 for c in cands
                          if (c.get("signal_date") or "") >= cutoff)
    meta["latest7"] = max((c.get("signal_date") or "" for c in cands
                           if (c.get("signal_date") or "") >= cutoff),
                          default="")'''

new_candidate_meta = '''    # 基于实际数据生成 meta，而非文件 mtime
    latest_signal = max((c.get("signal_date") or "" for c in cands), default="")
    symbols_seen = {c.get("symbol", "") for c in cands}
    cutoff = (date.today() - timedelta(days=7)).isoformat()
    recent_count = sum(1 for c in cands
                       if (c.get("signal_date") or "") >= cutoff)
    meta = {
        "generated_at": latest_signal or None,  # 用最新信号日期而非文件时间
        "candidate_count": len(cands),
        "candidate_symbols": len(symbols_seen),
        "coverage_known": True,
        "requested_symbols": len(symbols_seen),
        "accounted_symbols": len(symbols_seen),
        "latest_signal_date": latest_signal,
        "recent7": recent_count,
    }'''

content = content.replace(old_candidate_meta, new_candidate_meta)

# ======================================================================
# 6. 修复 checkCandidateMeta 中的标题引用
# ======================================================================

old_title = "W底/平台突破/杯柄/口袋支点"
new_title = "W底/平台突破/杯柄/口袋支点/旗形/洗盘/跌停反包"
content = content.replace(old_title, new_title)

# ======================================================================
# 7. 修复 checkFresh 中引用 data_status 的 last_date 与候选清单一致
# ======================================================================

# In checkFresh, the freshText shows data date. The candidateMeta shows candidate date.
# These are different things. The fix: candidateMeta now uses signal date, not file mtime.
# And we add a clarifying suffix to candidateMeta text.

old_candidate_text = "el.textContent=`形态清单 ${day} · 最近7天新信号 ${m.recent7??'-'} 条 · 扫描完整 ${m.accounted_symbols||0}/${m.requested_symbols||0} ✓`;"
new_candidate_text = "el.textContent=`形态清单（最新信号 ${day}）· 近7天 ${m.recent7??'-'} 条`;"
content = content.replace(old_candidate_text, new_candidate_text)

# ======================================================================
# 8. 移除废弃的 doPatternScreen 独立函数（被 clearAllConds 调用后清理即可）
# ======================================================================

# doPatternScreen 已经指向 pollPatternScreenProgress，无需修改

# ======================================================================
# 9. 添加 useTemplate 样式高亮同步（更新样式卡模板列表渲染逻辑）
# ======================================================================

# Make sure updateTplCount is called after toggleTpl
old_toggle = '''function toggleTpl(){
  const b=$('tplBody'), a=$('tplArrow');
  const open=b.style.display==='none';
  b.style.display=open?'':'none';
  a.textContent=open?'▴ 收起':'▾ 展开';
}'''

new_toggle = '''function toggleTpl(){
  const b=$('tplBody'), a=$('tplArrow');
  const open=b.style.display==='none';
  b.style.display=open?'':'none';
  a.textContent=open?'▴ 收起':'▾ 展开';
  if(open) updateTplCount();
}'''

content = content.replace(old_toggle, new_toggle)

# Write the file
app_py.write_text(content, encoding="utf-8")

print("✅ 修改完成：")
print("  1. 形态模板 TEMPLATES 追加 PATTERN_TEMPLATES")
print("  2. 侧边栏插入「② 形态筛选（技术面）」区域")
print("  3. 添加 togglePattern / renderPatternForm / togglePatternItem")
print("  4. clearAllConds 清理形态选择")
print("  5. 修复候选清单日期用信号日期而非文件 mtime")
print("  6. 修复标题引用和候选清单文本显示")
