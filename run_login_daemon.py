# -*- coding: utf-8 -*-
import subprocess
import time
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

if os.path.exists("auth_code.txt"):
    try: os.remove("auth_code.txt")
    except: pass
if os.path.exists("auth_url.txt"):
    try: os.remove("auth_url.txt")
    except: pass

cmd = ["wsl", "bash", "-c", "cd /mnt/c/Users/simoe/Downloads/image_gerador_colab && /home/simoesfsa/.local/share/uv/tools/google-colab-cli/bin/python login_daemon.py"]
p = subprocess.Popen(cmd)

url = None
for _ in range(30):
    if os.path.exists("auth_url.txt"):
        url = open("auth_url.txt", encoding="utf-8").read().strip()
        if url: break
    time.sleep(0.5)

if url:
    print("URL_ENCONTRADA_SUCESSO:")
    print(url)
else:
    print("Nenhuma URL foi escrita.")
