# -*- coding: utf-8 -*-
"""Deploy do app em uma sessao Colab explicita (via -s <session>)."""
import sys, os, json, base64, subprocess, time
sys.stdout.reconfigure(encoding='utf-8')

SESSION = sys.argv[1] if len(sys.argv) > 1 else "157vvflz3r2au"

def run_wsl_colab(cmd_args, input_str=None, timeout=120):
    cmd = ["wsl", "bash", "-c", f"export PATH=\"$HOME/.local/bin:$PATH\"; colab {' '.join(cmd_args)}"]
    try:
        res = subprocess.run(cmd, input=input_str, text=True, capture_output=True, timeout=timeout, encoding='utf-8', errors='replace')
        return res.returncode, res.stdout, res.stderr
    except Exception as e:
        return -1, "", str(e)

def exec_code(code, timeout=120):
    rc, out, err = run_wsl_colab(["exec", "-s", SESSION], input_str=code, timeout=timeout)
    return out + "\n" + err

print("📦 Lendo Notebook_Definitivo_CivitAI.ipynb...")
nb = json.load(open("Notebook_Definitivo_CivitAI.ipynb", encoding="utf-8"))
code = "".join(nb["cells"][1]["source"])
b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")

print("🚀 [2/4] Enviando run_all.py para /content (sessao " + SESSION + ")...")
script_upload = f"""
import base64
with open("/content/run_all.py", "w", encoding="utf-8") as f:
    f.write(base64.b64decode("{b64}").decode("utf-8"))
print("OK run_all.py atualizado:", len(base64.b64decode("{b64}")), "bytes")
"""
print(exec_code(script_upload, timeout=120))

print("⚡ [3/4] Executando inicializacao (app sobe em background)...")
script_run = """
import os, subprocess, sys
os.system("touch /content/.stop_supervisor")
os.system("pkill -9 -f universal_app.py")
if os.path.exists("/content/.stop_supervisor"):
    try: os.remove("/content/.stop_supervisor")
    except: pass
res = subprocess.run([sys.executable, "/content/run_all.py"], capture_output=True, text=True, timeout=280)
print("STDOUT_TAIL:", res.stdout[-1500:])
print("STDERR_TAIL:", res.stderr[-800:])
"""
print(exec_code(script_run, timeout=300))

print("🔍 [4/4] Verificando URL publica do Gradio...")
script_check = """
import os, time, re, requests
log_file = "/content/universal_app.log"
gradio_url = None
for _ in range(60):
    if os.path.exists(log_file):
        with open(log_file, encoding="utf-8", errors="replace") as f:
            content = f.read()
        m = re.findall(r"https://[a-z0-9\\-]+\\.gradio\\.live", content)
        if m:
            gradio_url = m[-1]
            break
    time.sleep(3)
print("=== RESULTADO ===")
print("Gradio Public URL:", gradio_url)
try:
    h = requests.get("http://127.0.0.1:7861/api/health", timeout=5).json()
    print("API Health:", h)
except Exception as e:
    print("API Health Exception:", e)
"""
print(exec_code(script_check, timeout=200))
