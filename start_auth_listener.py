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
    except Exception: pass
if os.path.exists("latest_oauth_url.txt"):
    try: os.remove("latest_oauth_url.txt")
    except Exception: pass

p = subprocess.Popen([sys.executable, "authenticate_pkce_persistent.py"], cwd=os.getcwd())

url = None
for _ in range(30):
    if os.path.exists("latest_oauth_url.txt"):
        url = open("latest_oauth_url.txt", encoding="utf-8").read().strip()
        if url:
            break
    time.sleep(0.5)

if url:
    print("URL_OK:", url)
else:
    print("FAIL_NO_URL")
