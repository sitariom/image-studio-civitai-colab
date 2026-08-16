# -*- coding: utf-8 -*-
"""
Interface de Automação para o Google Colab CLI Oficial (googlecolab/google-colab-cli v0.6.0).
Executa gerenciamento de sessões, execução remota de scripts e rotação de contas via WSL.
"""

import os
import sys
import subprocess
import json
import re

APP_DIR = os.path.expanduser("~/.pi/agent/colab")
os.makedirs(APP_DIR, exist_ok=True)

def run_wsl_colab(cmd_args, input_str=None):
    """Executa comandos do oficial google-colab-cli dentro do WSL."""
    cmd = ["wsl", "bash", "-c", f"export PATH=\"$HOME/.local/bin:$PATH\"; colab {' '.join(cmd_args)}"]
    try:
        res = subprocess.run(cmd, input=input_str, text=True, capture_output=True, timeout=120, encoding='utf-8', errors='replace')
        return res.returncode, res.stdout, res.stderr
    except Exception as e:
        return -1, "", str(e)

def get_auth_url():
    """Obtém a URL oficial de autorização do Google Colab CLI."""
    rc, out, err = run_wsl_colab(["sessions"])
    full_output = out + "\n" + err
    m = re.search(r"https://accounts\.google\.com/o/oauth2/auth\?[^\s]+", full_output)
    if m:
        return m.group(0)
    return None

def submit_auth_code(code_str):
    """Envia o código de autorização do Google gerado pelo navegador."""
    code = code_str.strip()
    rc, out, err = run_wsl_colab(["sessions"], input_str=code + "\n")
    if rc == 0:
        return True, "✅ Autenticação oficial do Google Colab CLI concluída com sucesso!"
    return False, f"Falha na autenticação: {out}\n{err}"

def colab_new_session(hardware="T4"):
    """Cria uma nova sessão no Colab com GPU (T4/L4/A100)."""
    if hardware.lower() == "cpu":
        rc, out, err = run_wsl_colab(["new"])
    else:
        rc, out, err = run_wsl_colab(["new", "--gpu", hardware])
    return out + err

def colab_list_sessions():
    """Lista as sessões ativas do Colab."""
    rc, out, err = run_wsl_colab(["sessions"])
    return out + err

def colab_exec_code(code_snippet):
    """Executa um trecho de código na sessão remota."""
    rc, out, err = run_wsl_colab(["exec"], input_str=code_snippet)
    return out + err

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    args = sys.argv[1:]
    if not args or args[0] == "url":
        url = get_auth_url()
        if url:
            print(f"🔗 URL Oficial de Autorização do Google Colab CLI:\n{url}")
        else:
            print("Já autenticado ou status atual:")
            print(colab_list_sessions())
    elif args[0] == "auth" and len(args) >= 2:
        ok, msg = submit_auth_code(args[1])
        print(msg)
    elif args[0] == "sessions":
        print(colab_list_sessions())
    elif args[0] == "new":
        hw = args[1] if len(args) > 1 else "gpu"
        print(colab_new_session(hw))
    else:
        print("Uso: official_colab_cli.py [url | auth <code> | sessions | new [gpu|cpu|tpu]]")
