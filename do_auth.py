# -*- coding: utf-8 -*-
import subprocess, sys, time, re, os

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

print("Iniciando fluxo de autenticacao do Google Colab CLI...", flush=True)

# Limpa pendências anteriores apenas se for string vazia
if os.path.exists("code_input.txt"):
    if not open("code_input.txt", encoding="utf-8").read().strip():
        os.remove("code_input.txt")

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
    print("❌ Falha ao obter a URL de autorização.")
    sys.exit(1)

print("\n" + "="*80)
print(" NOVA URL DE AUTORIZAÇÃO GERADA:")
print(url)
print("="*80 + "\n", flush=True)

print("Aguardando o envio do código via code_input.txt...", flush=True)

code = None
for i in range(120):
    if os.path.exists("code_input.txt"):
        code = open("code_input.txt", encoding="utf-8").read().strip()
        if code:
            os.remove("code_input.txt")
            break
    time.sleep(1)

if code:
    print(f"\nCodigo recebido. Enviando para o Colab CLI...", flush=True)
    out, err = p.communicate(input=code + "\n", timeout=30)
    print("STDOUT:", out)
    if err:
        print("STDERR:", err)
    if p.returncode == 0 or "No active sessions" in out or "m-s-" in out:
        print("AUTENTICACAO CONCLUIDA COM SUCESSO!")
    else:
        print("STATUS DA AUTENTICAÇÃO:", out)
else:
    print("TIMEOUT: Nenhum codigo fornecido dentro de 120 segundos.")
    p.kill()
