# -*- coding: utf-8 -*-
import subprocess, sys, time, re, os

wsl_clean = '''
if [ -f ~/.config/colab-cli/token.json ]; then
    rm ~/.config/colab-cli/token.json
fi
'''
subprocess.run(['wsl', 'bash', '-c', wsl_clean])

if os.path.exists('code_input.txt'): os.remove('code_input.txt')

cmd = '/home/simoesfsa/.local/bin/colab sessions'
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

print("URL_AUTENTICACAO_EMPARELHADA:")
print(url, flush=True)

open("current_url.txt", "w", encoding="utf-8").write(url)

# Wait up to 180 seconds for code_input.txt
code = None
for i in range(180):
    if os.path.exists("code_input.txt"):
        code = open("code_input.txt", encoding="utf-8").read().strip()
        if code:
            os.remove("code_input.txt")
            break
    time.sleep(1)

if code:
    print(f"Enviando código ao processo ativo...", flush=True)
    out, err = p.communicate(input=code + "\n", timeout=30)
    print("STDOUT:", out)
    if err:
        print("STDERR:", err)
    if p.returncode == 0 or "No active sessions" in out:
        print("AUTENTICADO_COM_SUCESSO!")
    else:
        print("FALHA_NA_AUTENTICACAO")
else:
    print("TIMEOUT_SEM_CODIGO")
    p.kill()
