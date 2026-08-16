# -*- coding: utf-8 -*-
"""
Script Interativo de Autenticação Oficial do Google Colab CLI (v0.6.0).
Garante correspondência exata do desafio PKCE OAuth2 (S256).
"""

import os
import sys
import subprocess
import time
import re

def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    print("🚀 Iniciando fluxo oficial do Google Colab CLI no WSL...\n")
    
    cmd = ["wsl", "bash", "-c", "export PATH=\"$HOME/.local/bin:$PATH\"; colab sessions"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    url = None
    t0 = time.time()
    out_lines = []
    
    while time.time() - t0 < 15:
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

    if not url:
        # Pega qualquer URL gerada no output acumulado
        accumulated = "".join(out_lines)
        m = re.search(r"https://accounts\.google\.com/o/oauth2/auth\?[^\s]+", accumulated)
        if m:
            url = m.group(0)

    if not url:
        print("Já autenticado ou resposta do Colab CLI:")
        print("".join(out_lines))
        p.kill()
        return

    print("🔗 URL Oficial de Autorização do Google Colab CLI (desafio PKCE atual):")
    print(f"\n{url}\n")
    print("👉 Acesse a URL acima no seu navegador, clique em 'Permitir' e passe o código de 4/... gerado!")

    # Se chamado com parametro de codigo
    if len(sys.argv) > 1:
        auth_code = sys.argv[1].strip()
        print(f"\nEnviando código de autorização: {auth_code[:10]}...")
        stdout_data, stderr_data = p.communicate(input=auth_code + "\n", timeout=30)
        print("Saída:", stdout_data)
        if stderr_data:
            print("Status:", stderr_data)
    else:
        p.kill()

if __name__ == "__main__":
    main()
