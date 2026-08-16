# -*- coding: utf-8 -*-
"""Release automatizado — mantem o ciclo de versionamento completo.

Uso:
    python _release.py "feat: descricao" [nova_versao]
    python _release.py "fix: corrige X"              # mantem APP_VER atual
    python _release.py "feat: Y" v2.6.20260817       # bump APP_VER explicito

Fluxo:
1. (opcional) bump APP_VER no universal_app.py
2. python _gerar_notebook.py (regenera o notebook)
3. valida roundtrip embedded == disk + AST
4. git add -A && git commit -m "<msg>" && git push

Regras:
- Nunca editar o notebook a mao (sempre regenerar).
- Nao commitar tokens (gitignore cobre _local_secrets_backup/ e backups/).
- Conventional Commits: feat:, fix:, docs:, chore:.
"""
import re, subprocess, sys, os

APP = "universal_app.py"

def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print("FALHA:", cmd, "\n", r.stderr[-800:])
        sys.exit(1)
    return r.stdout

def bump_version(version):
    src = open(APP, encoding="utf-8").read()
    m = re.search(r'APP_VER = "([^"]+)"', src)
    if not m:
        print("APP_VER nao encontrado"); sys.exit(1)
    old = m.group(1)
    src = src.replace('APP_VER = "' + old + '"', 'APP_VER = "' + version + '"')
    open(APP, "w", encoding="utf-8").write(src)
    print(f"[1/5] APP_VER: {old} -> {version}")

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    msg = sys.argv[1]
    version = sys.argv[2] if len(sys.argv) > 2 else None

    if version:
        bump_version(version)

    print("[2/5] Regenerando notebook...")
    run([sys.executable, "_gerar_notebook.py"])

    print("[3/5] Validando roundtrip + AST...")
    check = '''
import json, re, ast, sys
nb = json.load(open("Notebook_Definitivo_CivitAI.ipynb", encoding="utf-8"))
c = "".join(nb["cells"][1]["source"])
m = re.search(r"APP_SRC = r\\'\\'\\'(.*?)\\'\\'\\'", c, re.S)
disk = open("universal_app.py", encoding="utf-8").read()
ast.parse(c); ast.parse(disk)
assert m and m.group(1) == disk, "ROUNDTRIP FALHOU: notebook != app"
assert chr(39)*3 not in disk, "app contem aspas triplas simples (embed quebra)"
assert not re.search(r"f[\\"\\']{3}", c), "f-string tripla no notebook"
print("roundtrip OK | app:", len(disk), "chars")
'''
    run([sys.executable, "-c", check])

    print("[4/5] Commit + push...")
    subprocess.run(["git", "add", "-A"], check=True)
    r = subprocess.run(["git", "commit", "-m", msg], capture_output=True, text=True)
    print(r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:])
    subprocess.run(["git", "push"], check=True)

    print("[5/5] Release concluido:", msg)
    print("     Ver em https://github.com/sitariom/image-studio-civitai-colab")

if __name__ == "__main__":
    main()
