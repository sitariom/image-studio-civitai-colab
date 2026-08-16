# -*- coding: utf-8 -*-
"""Deploy da versao REGENERADA (2026-08-15) em uma sessao Colab explicita.

Fluxo (validado nas sessoes anteriores):
1) le a celula unica do Notebook_Definitivo_CivitAI.ipynb (runner + APP_SRC embutido)
2) envia via base64 -> /content/run_all.py (evita problemas de encoding do heredoc)
3) executa run_all.py com timeout longo (pip setup ~60s + app start + health ~180s)
4) extrai URL do Gradio + resultado do /api/health (app_ver) do output
"""
import sys, os, json, base64, subprocess, time, re

SESSION = sys.argv[1] if len(sys.argv) > 1 else "image_studio"
NB = "Notebook_Definitivo_CivitAI.ipynb"

def run_wsl_colab(cmd_args, input_str=None, timeout=180):
    cmd = ["wsl", "bash", "-c", f"export PATH=\"$HOME/.local/bin:$PATH\"; colab {' '.join(cmd_args)}"]
    try:
        res = subprocess.run(cmd, input=input_str, text=True, capture_output=True,
                             timeout=timeout, encoding='utf-8', errors='replace')
        return res.returncode, res.stdout, res.stderr
    except Exception as e:
        return -1, "", str(e)

def exec_code(code, timeout=180):
    code += "\nprint('__EXEC_DONE__')"
    rc, out, err = run_wsl_colab(["exec", "-s", SESSION, "--timeout", str(timeout)], input_str=code, timeout=timeout + 30)
    return out + "\n" + err

print("=" * 64)
print("DEPLOY image_studio | sessao:", SESSION, "| notebook:", NB)

# [1/4] Le a celula unica (cell index 1) do notebook REGENERADO
cell = "".join(json.load(open(NB, encoding="utf-8"))["cells"][1]["source"])
print(f"[1/4] Celula unica capturada: {len(cell)} chars | APP_SRC presente: {'APP_SRC = r' in cell}")

# [2/4] Envia run_all.py para /content (base64)
b64 = base64.b64encode(cell.encode("utf-8")).decode("ascii")
upload = f"""
import base64
data = base64.b64decode("{b64}").decode("utf-8")
open("/content/run_all.py", "w", encoding="utf-8").write(data)
print("OK run_all.py:", len(data), "chars")
"""
out = exec_code(upload, timeout=90)
print("[2/4] upload:", out.strip()[-200:])
if "__EXEC_DONE__" not in out or "OK run_all.py" not in out:
    print("FALHA no upload"); sys.exit(1)

# [3/4] Executa o setup + app + health (pode levar 2-6 min na 1a vez)
print("[3/4] Executando setup+app+health (ate ~6 min)...")
runner = """
import subprocess, sys, time, re
t0 = time.time()
proc = subprocess.Popen([sys.executable, "-u", "/content/run_all.py"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
out = []
done = False
start = time.time()
while time.time() - start < 300:
    line = proc.stdout.readline()
    if not line:
        if proc.poll() is not None:
            break
        time.sleep(0.5)
        continue
    out.append(line)
    sys.stdout.write(line)
    sys.stdout.flush()
    if "CONCLUIDO" in line or "ERRO" in line or "SystemExit" in line:
        break
if proc.poll() is None:
    proc.terminate()
print(f"\\n__RUN_DONE__ rc={proc.poll()} tempo={int(time.time()-t0)}s")
"""
out = exec_code(runner, timeout=360)
print("----- output da celula -----")
print(out[-4500:])