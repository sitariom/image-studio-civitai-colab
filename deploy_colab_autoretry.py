# -*- coding: utf-8 -*-
"""Deploy automático com retry — provisiona VM T4 + sobe o app multi-modelo.

Quando o Colab responde 503 (Service Unavailable — pico/instabilidade), este script
tenta de novo com backoff e ao obter a sessão executa o deploy completo do notebook
regenerado (célula única v3, v2.1.20260815): upload run_all.py -> setup+pip ->
app detached + supervisor -> health (app_ver) -> repair_vae.

Uso:
    python deploy_colab_autoretry.py          # provisiona e deploya (até N tentativas)
    python deploy_colab_autoretry.py --check  # só reporta estado (sessões/token)
"""
import sys, os, json, base64, subprocess, time, re, argparse

SESSION = "image_studio"
NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Notebook_Definitivo_CivitAI.ipynb")
MAX_TRIES = 30          # tentativas de colab new (backoff 30s-180s)
PROVISION_TIMEOUT = 120

def wsl(cmd, input_str=None, timeout=120):
    full = f'export PATH="$HOME/.local/bin:$PATH"; {cmd}'
    try:
        r = subprocess.run(["wsl", "bash", "-c", full], input=input_str, text=True,
                           capture_output=True, timeout=timeout, encoding="utf-8", errors="replace")
        return r.returncode, r.stdout, r.stderr
    except Exception as e:
        return -1, "", str(e)

def session_exists():
    rc, out, _ = wsl("colab ls 2>&1")
    return SESSION in out or "image_studio" in out

def create_session():
    rc, out, err = wsl(f"colab new -s {SESSION} --gpu T4", timeout=PROVISION_TIMEOUT)
    ok = rc == 0 and "Created session" in out or "created" in out.lower() and rc == 0
    return ok, out.strip()[-400:] + err.strip()[-200:]

def deploy():
    """Espelha deploy_new_session.py: céluula única -> /content/run_all.py -> exec."""
    cell = "".join(json.load(open(NB, encoding="utf-8"))["cells"][1]["source"])
    print(f"[1/4] Célula única: {len(cell)} chars | APP_SRC: {'APP_SRC = r' in cell}")

    def exec_code(code, timeout=180):
        code += "\nprint('__EXEC_DONE__')"
        rc, out, err = wsl(f"colab exec -s {SESSION} --timeout {timeout}", input_str=code, timeout=timeout + 30)
        return out + "\n" + err

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
        print("FALHA no upload"); return False

    runner = """
import subprocess, sys, time
t0 = time.time()
proc = subprocess.Popen([sys.executable, "-u", "/content/run_all.py"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                        encoding="utf-8", errors="replace")
out = []
start = time.time()
while time.time() - start < 600:
    line = proc.stdout.readline()
    if not line:
        if proc.poll() is not None:
            break
        time.sleep(0.5)
        continue
    out.append(line)
    sys.stdout.write(line); sys.stdout.flush()
    if "CONCLUIDO" in line or "ERRO" in line or "SystemExit" in line:
        break
if proc.poll() is None:
    proc.terminate()
print(f"\\n__RUN_DONE__ rc={proc.poll()} tempo={int(time.time()-t0)}s")
"""
    out = exec_code(runner, timeout=660)
    tail = out.strip()[-4500:]
    print("[3/4] run_all (tail):")
    print(tail)
    print("[4/4] FIM")
    return ("app_ver" in tail or "CONCLUIDO" in tail) and "__RUN_DONE__" in out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tries", type=int, default=MAX_TRIES)
    args = ap.parse_args()

    if args.check:
        rc, out, err = wsl("colab ls 2>&1 | head -5")
        print("estado:", out.strip()[:300])
        return

    # 1) Provisiona com retry
    got = False
    for i in range(1, args.tries + 1):
        print(f"\n=== [{time.strftime('%H:%M:%S')}] Tentativa {i}/{args.tries} de provisionar T4 ===")
        ok, msg = create_session()
        if ok or session_exists():
            got = True
            print("Sessão pronta:", msg)
            break
        delay = min(20 + i * 10, 300)
        print(f"falhou ({msg[-120:]}) | retry em {delay}s")
        time.sleep(delay)
    if not got:
        print("\nNÃO foi possível provisionar após", args.tries, "tentativas. Rodar de novo depois.")
        return 1

    # 2) Deploy
    ok = deploy()
    print("\nDEPLOY:", "OK" if ok else "FALHOU (ver tail acima)")
    return 0 if ok else 2

if __name__ == "__main__":
    sys.exit(main())