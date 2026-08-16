# -*- coding: utf-8 -*-
import sys
import subprocess
import os

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    if len(sys.argv) < 2:
        print("Informe o código de autorização: python submit_code_wsl.py <codigo>")
        sys.exit(1)

    code = sys.argv[1].strip()
    print(f"Enviando código para o Google Colab CLI no WSL: {code[:10]}...")
    
    cmd = "export PATH=\"$HOME/.local/bin:$PATH\"; colab sessions"
    p = subprocess.Popen(["wsl", "bash", "-c", cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    
    out, err = p.communicate(input=code + "\n", timeout=30)
    print("\n--- RESPOSTA DO COLAB CLI ---")
    print(out)
    if err:
        print("ERR:", err)
