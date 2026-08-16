# -*- coding: utf-8 -*-
import json
import base64
import sys
import official_colab_cli

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

def run():
    print("📦 [1/3] Extraindo a célula única do Notebook_Definitivo_CivitAI.ipynb...")
    with open("Notebook_Definitivo_CivitAI.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    code = "".join(nb["cells"][1]["source"])
    print(f"   Tamanho do código: {len(code)} caracteres.")
    
    b64_code = base64.b64encode(code.encode("utf-8")).decode("ascii")
    
    upload_script = f"""
import base64
b64 = "{b64_code}"
with open("/content/run_all.py", "w", encoding="utf-8") as f:
    f.write(base64.b64decode(b64).decode("utf-8"))
print("✅ /content/run_all.py gravado no Colab com sucesso.")
"""
    print("🚀 [2/3] Enviando /content/run_all.py para a sessão do Colab...")
    out = official_colab_cli.colab_exec_code(upload_script)
    print(out)
    
    print("⚡ [3/3] Executando /content/run_all.py no Colab (aguarde a conclusão do setup e inicialização)...")
    exec_script = """
import subprocess, sys
res = subprocess.run([sys.executable, "/content/run_all.py"], capture_output=False, text=True)
"""
    out_exec = official_colab_cli.colab_exec_code(exec_script)
    print(out_exec)

if __name__ == "__main__":
    run()
