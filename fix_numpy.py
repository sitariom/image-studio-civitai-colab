# -*- coding: utf-8 -*-
"""Fix numpy: 2.3.5 remove _slice (scipy quebra). Usar 2.2.6 (tem _slice e _blas_supports_fpe)."""
import re

# ===========================================================================
# 1. Worker
# ===========================================================================
wp = "krea2_worker.py"
w = open(wp, encoding="utf-8").read()

old_w = '''    # 3) pins criticos: numpy 2.3.5 (scipy/_blas) + quanto 0.2.4 (API)
    try:
        import numpy as np2
        if np2.__version__ != "2.3.5":
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==2.3.5"], timeout=300)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==2.3.5"], timeout=300)'''
new_w = '''    # 3) pins: numpy 2.2.6 (2.3.x remove _slice do scipy; 2.4+ remove _blas_supports_fpe)
    #    + optimum-quanto 0.2.4 (API antiga)
    try:
        import numpy as np2
        _nv = [int(x) for x in str(np2.__version__).split('.')[:2]]
        if _nv[0] > 2 or (_nv[0] == 2 and _nv[1] >= 3):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==2.2.6"], timeout=300)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==2.2.6"], timeout=300)'''
assert old_w in w, "worker numpy block"
w = w.replace(old_w, new_w)
open(wp, "w", encoding="utf-8").write(w)
print("[OK] worker numpy -> 2.2.6 (condicional)")

# ===========================================================================
# 2. App (2 blocos: ensure_requirements duplicado + _ensure_wan2gp_deps)
# ===========================================================================
APP = "universal_app.py"
src = open(APP, encoding="utf-8").read()

# re-embutir worker
worker = open(wp, encoding="utf-8").read()
assert chr(39) * 3 not in worker and chr(34) * 3 not in worker
m = re.search(r'KREA2_WORKER_SRC = r"""(.*?)"""', src, re.S)
assert m
src = src.replace(m.group(0), 'KREA2_WORKER_SRC = r"""' + worker + '"""')

count = src.count('"numpy==2.3.5"')
assert count >= 2, "ocorrencias numpy 2.3.5 no app: %d" % count
# bloco completo (comentario + try/except) — substitui nos 2 lugares
old_a = '''    # 3) pins criticos: numpy 2.3.5 (scipy/_blas) + quanto 0.2.4 (API)
    try:
        import numpy as np2
        if np2.__version__ != "2.3.5":
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==2.3.5"], timeout=300)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==2.3.5"], timeout=300)'''
new_a = '''    # 3) pins: numpy 2.2.6 (2.3.x remove _slice do scipy; 2.4+ remove _blas_supports_fpe)
    #    + optimum-quanto 0.2.4 (API antiga)
    try:
        import numpy as np2
        _nv = [int(x) for x in str(np2.__version__).split('.')[:2]]
        if _nv[0] > 2 or (_nv[0] == 2 and _nv[1] >= 3):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==2.2.6"], timeout=300)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==2.2.6"], timeout=300)'''
n = src.count(old_a)
assert n == 2, "blocos numpy antigos no app: %d" % n
src = src.replace(old_a, new_a)
open(APP, "w", encoding="utf-8").write(src)
print("[OK] app: 2 blocos numpy -> 2.2.6 (condicional)")

# ===========================================================================
# 3. _gerar_notebook.py (cell_run: 2 linhas do pin)
# ===========================================================================
G = "_gerar_notebook.py"
g = open(G, encoding="utf-8").read()
old_g1 = '"\\"_pip(\'numpy==2.3.5\')  # trava versao: scipy 1.16 exige _blas_supports_fpe (removido no numpy>=2.4)\\n\\""'
old_g1 = '"_pip(\'numpy==2.3.5\')  # trava versao: scipy 1.16 exige _blas_supports_fpe (removido no numpy>=2.4)\\n"'
new_g1 = '"_pip(\'numpy==2.2.6\')  # pin seguro: 2.3.x remove _slice (scipy), 2.4+ remove _blas_supports_fpe\\n"'
assert old_g1 in g, "gerador linha1"
g = g.replace(old_g1, new_g1)
old_g2 = '"_pip(\'numpy==2.3.5\')  # reforco pos-instalacao (compat scipy/Wan2GP)\\n"'
new_g2 = '"# (numpy ja fixado acima; nao reinstalar)\\n"'
assert old_g2 in g, "gerador linha2"
g = g.replace(old_g2, new_g2)
open(G, "w", encoding="utf-8").write(g)
print("[OK] _gerar_notebook.py: numpy -> 2.2.6")
