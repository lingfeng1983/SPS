"""用 node --check 检查页面 JS 语法（从运行中的服务器抓取）。"""
import re
import subprocess
import urllib.request
import os
import tempfile

html = urllib.request.urlopen("http://127.0.0.1:5000/").read().decode("utf-8")
scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
tmp = os.path.join(tempfile.gettempdir(), "sps_check.js")
open(tmp, "w", encoding="utf-8").write("\n".join(scripts))
print("script blocks:", len(scripts))

r = subprocess.run(["node", "--check", tmp], capture_output=True, text=True)
if r.returncode == 0:
    print("JS SYNTAX OK")
else:
    print("JS ERROR:")
    print(r.stderr[:1500])
