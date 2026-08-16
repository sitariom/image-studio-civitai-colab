# -*- coding: utf-8 -*-
import json
import base64
import sys
import time
import official_colab_cli

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

def main():
    print("📦 [1/4] Lendo Notebook_Definitivo_CivitAI.ipynb...")
    with open("Notebook_Definitivo_CivitAI.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)
    code = "".join(nb["cells"][1]["source"])
    b64_code = base64.b64encode(code.encode("utf-8")).decode("ascii")

    print("🚀 [2/4] Enviando código atualizado (/content/run_all.py) para o Colab...")
    script_upload = f"""
import base64
with open("/content/run_all.py", "w", encoding="utf-8") as f:
    f.write(base64.b64decode("{b64_code}").decode("utf-8"))
print("✅ run_all.py atualizado no Colab.")
"""
    out1 = official_colab_cli.colab_exec_code(script_upload)
    print(out1)

    print("⚡ [3/4] Executando inicialização do app no Colab...")
    script_run = """
import os, subprocess, sys
# Interrompe instâncias anteriores se houver
os.system("touch /content/.stop_supervisor")
os.system("pkill -9 -f universal_app.py")
if os.path.exists("/content/.stop_supervisor"):
    try: os.remove("/content/.stop_supervisor")
    except: pass

res = subprocess.run([sys.executable, "/content/run_all.py"], capture_output=True, text=True, timeout=300)
print("STDOUT:", res.stdout[-2000:])
print("STDERR:", res.stderr[-1000:])
"""
    out2 = official_colab_cli.colab_exec_code(script_run)
    print(out2)

    print("🔍 [4/4] Verificando logs e obtendo URL pública do Gradio...")
    script_check = """
import os, time, re, requests
log_file = "/content/universal_app.log"
gradio_url = None
if os.path.exists(log_file):
    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
        m = re.findall(r"https://[a-z0-9\-]+\.gradio\.live", content)
        if m:
            gradio_url = m[-1]

print("=== RESULTADO ===")
print("Gradio Public URL:", gradio_url)
try:
    h = requests.get("http://127.0.0.1:7861/api/health", timeout=3).json()
    print("API Health:", h)
except Exception as e:
    print("API Health Exception:", e)
"""
    out3 = official_colab_cli.colab_exec_code(script_check)
    print(out3)

if __name__ == "__main__":
    main()
