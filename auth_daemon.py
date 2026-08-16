# -*- coding: utf-8 -*-
import subprocess, sys, time, re, os

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

if os.path.exists("code_input.txt"):
    os.remove("code_input.txt")
if os.path.exists("latest_url.txt"):
    os.remove("latest_url.txt")

cmd = 'export PATH="/home/simoesfsa/.local/bin:$PATH"; colab sessions'
p = subprocess.Popen(['wsl', 'bash', '-c', cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

url = None
t0 = time.time()
while time.time() - t0 < 10:
    line = p.stderr.readline() or p.stdout.readline()
    if line:
        m = re.search(r'https://accounts\.google\.com/o/oauth2/auth\?[^\s]+', line)
        if m:
            url = m.group(0)
            break

if not url:
    print("ERR: No URL found.")
    sys.exit(1)

with open("latest_url.txt", "w", encoding="utf-8") as f:
    f.write(url)

print("URL_READY:" + url, flush=True)

# Loop indefinitely until code_input.txt is written
while True:
    if os.path.exists("code_input.txt"):
        c = open("code_input.txt", encoding="utf-8").read().strip()
        if c:
            print(f"Submitting code ({len(c)} chars)...", flush=True)
            out, err = p.communicate(input=c + "\n", timeout=30)
            print("STDOUT:", out)
            print("STDERR:", err)
            if "No active sessions" in out or "m-s-" in out or p.returncode == 0:
                print("AUTH_SUCCESS")
            else:
                print("AUTH_RESULT:", out)
            os.remove("code_input.txt")
            break
    time.sleep(0.5)
