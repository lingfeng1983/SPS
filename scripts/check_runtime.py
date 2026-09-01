"""用 jsdom 思路的极简 DOM 桩，在 node 里真实执行页面 JS，抓运行时错误。"""
import subprocess
import urllib.request
import os
import tempfile

html = urllib.request.urlopen("http://127.0.0.1:5000/").read().decode("utf-8")
import re
scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
page_js = "\n".join(scripts)

runner = """
const { execSync } = require('child_process');
const fs = require('fs');
const store = {};   // id -> innerHTML
const els = {};
function mkEl(id){
  return {
    id,
    set innerHTML(v){ store[id]=v; },
    get innerHTML(){ return store[id]||''; },
    style: {},
    dataset: {},
    classList: { toggle(){}, add(){}, remove(){} },
    addEventListener(){},
    querySelectorAll(){ return []; },
    querySelector(){ return null; },
    disabled: false,
    scrollTop: 0, scrollHeight: 0,
    value: ''
  };
}
global.document = {
  getElementById: id => els[id] || (els[id] = mkEl(id)),
  querySelectorAll(){ return []; }
};
global.window = { addEventListener(){} };
global.alert = m => console.log('[alert]', m);
global.prompt = () => null;
global.confirm = () => false;
global.Plotly = { newPlot(){} };
global.fetch = async (url) => {
  const out = execSync(`curl -s "http://127.0.0.1:5000${url}"`, {maxBuffer: 10*1024*1024}).toString();
  return { json: async () => JSON.parse(out), ok: true };
};

(async () => {
  try {
    eval(fs.readFileSync(process.argv[2], 'utf8'));
    // 等异步任务落地
    await new Promise(r => setTimeout(r, 3000));
    const h = store['indForm'] || '';
    console.log('=== indForm length:', h.length);
    console.log(h.slice(0, 400));
    if (h.length < 100) {
      // 打印其他容器内容帮助定位
      for (const [k,v] of Object.entries(store)) {
        if (v && v.length) console.log('container', k, 'len', v.length);
      }
    }
  } catch(e) {
    console.log('RUNTIME ERROR:', e.message);
    console.log(e.stack.split('\\n').slice(0,4).join('\\n'));
  }
})();
"""

tmp_page = os.path.join(tempfile.gettempdir(), "sps_page.js")
tmp_run = os.path.join(tempfile.gettempdir(), "sps_runner.cjs")
open(tmp_page, "w", encoding="utf-8").write(page_js)
open(tmp_run, "w", encoding="utf-8").write(runner)

r = subprocess.run(["node", tmp_run, tmp_page], capture_output=True, text=True,
                   timeout=60, cwd=tempfile.gettempdir())
print("STDOUT:", r.stdout[:1200])
if r.stderr:
    print("STDERR:", r.stderr[:400])
