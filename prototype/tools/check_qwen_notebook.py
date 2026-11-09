"""Checks the generated Kaggle notebook locally: runs its module-writing cell (Kaggle path swapped for a temp dir),
confirms the embedded module is byte-identical to prototype/layerproto/qwen_layered.py, and compiles every cell."""
import hashlib
import json
import os
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
nb = json.load(open(os.path.join(PROTO, "kaggle", "upload_qwen_test", "qwen-layered-feasibility.ipynb"), encoding="utf-8"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
tmp = tempfile.mkdtemp().replace(os.sep, "/")
exec(cells[1].replace('WORK = "/kaggle/working"', f'WORK = "{tmp}"'), {})
a = open(os.path.join(tmp, "layerproto", "qwen_layered.py"), "rb").read()
b = open(os.path.join(PROTO, "layerproto", "qwen_layered.py"), "rb").read()
print("embedded module identical:", hashlib.md5(a.replace(b"\r\n", b"\n")).hexdigest() == hashlib.md5(b.replace(b"\r\n", b"\n")).hexdigest())
for i, c in enumerate(cells):
    compile("\n".join(l for l in c.splitlines() if not l.lstrip().startswith("!")), f"cell{i}", "exec")
print("all", len(cells), "code cells compile")
