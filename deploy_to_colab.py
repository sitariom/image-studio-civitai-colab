# -*- coding: utf-8 -*-
import base64
import sys
import os
import official_colab_cli

def deploy():
    print("📦 [1/4] Lendo e codificando universal_app.py...")
    with open("universal_app.py", "rb") as f:
        content = f.read()
    
    b64_data = base64.b64encode(content).decode("ascii")
    
    print(f" [2/4] Enviando universal_app.py ({len(content)} bytes) para a sessão ativa do Colab...")
    
    # Send in chunks if needed, or single chunk
    remote_script = f"""
import base64, os
b64 = "{b64_data}"
with open("universal_app.py", "wb") as f:
    f.write(base64.b64decode(b64))
print("✅ Arquivo universal_app.py gravado com sucesso no Colab. Tamanho:", os.path.getsize("universal_app.py"))
"""
    out = official_colab_cli.colab_exec_code(remote_script)
    print(out)

    print("⚡ [3/4] Verificando e instalando dependências essenciais no Colab...")
    install_script = """
import sys, subprocess
reqs = ["gradio", "nest_asyncio", "requests", "pillow", "numpy", "psutil"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + reqs)
print("✅ Dependências Python instaladas.")
"""
    out_deps = official_colab_cli.colab_exec_code(install_script)
    print(out_deps)

    print("🚀 [4/4] Iniciando a aplicação universal_app.py no Colab...")
    run_script = """
import subprocess, os, time
cmd = "nohup python universal_app.py > app.log 2>&1 &"
os.system(cmd)
time.sleep(5)
if os.path.exists("app.log"):
    with open("app.log") as f:
        print("--- LOG DE INICIALIZAÇÃO ---")
        print(f.read()[:2000])
"""
    out_run = official_colab_cli.colab_exec_code(run_script)
    print(out_run)

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    deploy()
