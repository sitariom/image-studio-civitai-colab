# -*- coding: utf-8 -*-
"""Validacao completa do projeto (usada pelo GitHub Actions CI e localmente).

Falha se:
- universal_app.py / worker embutido / notebook tem erro de sintaxe
- roundtrip notebook != universal_app.py (regra de ouro)
- f-string tripla ou aspas triplas simples no embed
- tokens de API vazados nos arquivos versionados
- pyflakes aponta 'undefined name'

Uso: python _validate.py
"""
import json, re, ast, subprocess, sys

TOKENS = ["31e722c6" + "1d7d631c534b23ba0f8bb57d", "hf_NWQbI" + "EaZgaARRMOxatlQPKCFGRdxMHVmUN"]
TRACKED = ["universal_app.py", "_gerar_notebook.py", "Notebook_Definitivo_CivitAI.ipynb", "krea2_worker.py"]


def main():
    errors = []
    warnings = []

    # 1. AST universal_app.py
    src = open("universal_app.py", encoding="utf-8").read()
    try:
        ast.parse(src)
    except SyntaxError as e:
        errors.append("SyntaxError universal_app.py: %s" % e)

    # 2. AST + regras do worker embutido
    m = re.search(r'KREA2_WORKER_SRC = r"""(.*?)"""', src, re.S)
    if not m:
        errors.append("KREA2_WORKER_SRC nao encontrado em universal_app.py")
    else:
        w = m.group(1)
        try:
            ast.parse(w)
        except SyntaxError as e:
            errors.append("SyntaxError worker embutido: %s" % e)
        if chr(39) * 3 in w or chr(34) * 3 in w:
            errors.append("worker embutido contem aspas triplas (quebra o embed)")

    # 3. Regras do app (embed no notebook)
    if chr(39) * 3 in src:
        errors.append("universal_app.py contem aspas triplas simples (quebra o APP_SRC)")
    if re.search(r'f["\']{3}', src):
        errors.append("f-string tripla em universal_app.py")

    # 4. Roundtrip notebook == app
    nb = json.load(open("Notebook_Definitivo_CivitAI.ipynb", encoding="utf-8"))
    c = "".join(nb["cells"][1]["source"])
    m2 = re.search(r"APP_SRC = r'''(.*?)'''", c, re.S)
    if not m2:
        errors.append("APP_SRC nao encontrado no notebook")
    elif m2.group(1) != src:
        errors.append("ROUNDTRIP FALHOU: notebook != universal_app.py (rode _gerar_notebook.py)")
    if re.search(r'f["\']{3}', c):
        errors.append("f-string tripla no notebook")

    # 5. Tokens vazados
    for tok in TOKENS:
        for f in TRACKED:
            try:
                if tok in open(f, encoding="utf-8", errors="replace").read():
                    errors.append("TOKEN VAZADO em %s! Sanitizar antes de commitar." % f)
            except FileNotFoundError:
                pass

    # 6. pyflakes (undefined name = erro; unused = aviso)
    r = subprocess.run([sys.executable, "-m", "pyflakes", "universal_app.py", "krea2_worker.py"],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "undefined name" in line:
            errors.append("pyflakes: " + line.strip())
        elif "imported but unused" in line or "assigned to but never used" in line:
            warnings.append("pyflakes: " + line.strip())

    print("=" * 56)
    if warnings:
        print("AVISOS (%d):" % len(warnings))
        for wl in warnings[:6]:
            print("  !", wl)
    if errors:
        print("ERROS (%d):" % len(errors))
        for e in errors:
            print("  x", e)
        print("=" * 56)
        print("VALIDACAO FALHOU")
        sys.exit(1)
    print("VALIDACAO OK | app: %d chars | worker: %d chars | notebook: %d celulas"
          % (len(src), len(m.group(1)) if m else 0, len(nb["cells"])))
    print("=" * 56)


if __name__ == "__main__":
    main()
