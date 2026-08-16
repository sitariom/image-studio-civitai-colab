# -*- coding: utf-8 -*-
"""
Script de Autenticação PKCE Persistente para o Google Colab CLI.
Mantém o mesmo processo OAuth aberto para garantir que o code_verifier do PKCE corresponda exatamente ao código retornado pelo Google.
"""
import os
import sys
import subprocess
import time
import re

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

def start_and_wait():
    if os.path.exists("auth_code.txt"):
        try: os.remove("auth_code.txt")
        except Exception: pass

    cmd = ["wsl", "bash", "-c", "export PATH=\"$HOME/.local/bin:$PATH\"; colab sessions"]
    p = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1
    )
    
    url = None
    t0 = time.time()
    out_lines = []
    
    while time.time() - t0 < 10:
        line = p.stderr.readline()
        if not line:
            line = p.stdout.readline()
        if line:
            out_lines.append(line)
            m = re.search(r"https://accounts\.google\.com/o/oauth2/auth\?[^\s]+", line)
            if m:
                url = m.group(0)
                break
        time.sleep(0.1)

    if url:
        print("URL_ENCONTRADA:")
        print(url, flush=True)
        open("latest_oauth_url.txt", "w", encoding="utf-8").write(url)
        print("Aguardando código ser escrito em auth_code.txt...", flush=True)
        code = None
        for i in range(300): # espera até 5 minutos pelo código em auth_code.txt
            if os.path.exists("auth_code.txt"):
                code = open("auth_code.txt", encoding="utf-8").read().strip()
                if code:
                    os.remove("auth_code.txt")
                    break
            time.sleep(1)
        
        if code:
            print(f"Enviando código {code[:10]}... para o processo ativo do Colab CLI...", flush=True)
            stdout, stderr = p.communicate(input=code + "\n", timeout=30)
            print("STDOUT:", stdout)
            if stderr:
                print("STDERR:", stderr)
            if p.returncode == 0 or "No active sessions found" in stdout or "gpu" in stdout.lower():
                print("SUCCESS: Autenticado com sucesso!")
            else:
                print("FAIL: Falha ao autenticar.")
        else:
            print("TIMEOUT: Nenhum código fornecido em 5 minutos.")
            p.kill()
    else:
        print("Nenhuma URL gerada ou já autenticado.")
        p.kill()

if __name__ == "__main__":
    start_and_wait()
