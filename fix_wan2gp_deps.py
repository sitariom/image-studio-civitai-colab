# -*- coding: utf-8 -*-
"""Fix: instalar requirements.txt do Wan2GP (smplfitter, mmgp, etc.) — fiel aos notebooks."""
import re

# ===========================================================================
# 1. Worker: ensure_requirements completo (requirements.txt + mmgp/gradio + pins)
# ===========================================================================
wp = "krea2_worker.py"
w = open(wp, encoding="utf-8").read()

old_ensure = '''def ensure_requirements():
    # pins criticos (notebooks funcionam com eles): numpy 2.3.5 (scipy/_blas) + quanto 0.2.4 (API)
    try:
        import numpy as np2
        if np2.__version__ != "2.3.5":
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==2.3.5"], timeout=300)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==2.3.5"], timeout=300)
    try:
        import optimum.quanto
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "optimum-quanto==0.2.4"], timeout=300)
    try:
        from mmgp import offload
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "mmgp"], timeout=300)'''

new_ensure = '''def ensure_requirements():
    # 1) requirements.txt COMPLETO do Wan2GP (smplfitter, mmgp, ...) — fiel aos notebooks
    reqs_txt = os.path.join(WAN2GP_DIR, "requirements.txt")
    if os.path.exists(reqs_txt):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                                   "--timeout", "120", "--retries", "5", "-r", reqs_txt], timeout=900)
            log("requirements.txt do Wan2GP instalado")
        except Exception as e:
            log("pip requirements warn: " + str(e)[:150])
    # 2) mmgp + gradio (fiel ao notebook)
    try:
        import mmgp  # noqa: F401
        import gradio  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "mmgp", "gradio"], timeout=600)
    # 3) pins criticos: numpy 2.3.5 (scipy/_blas) + quanto 0.2.4 (API)
    try:
        import numpy as np2
        if np2.__version__ != "2.3.5":
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==2.3.5"], timeout=300)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==2.3.5"], timeout=300)
    try:
        import optimum.quanto
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "optimum-quanto==0.2.4"], timeout=300)'''

assert old_ensure in w, "worker ensure nao encontrado"
w = w.replace(old_ensure, new_ensure)
open(wp, "w", encoding="utf-8").write(w)
print("[OK] worker ensure_requirements completo (requirements.txt do Wan2GP)")

# ===========================================================================
# 2. App: _ensure_wan2gp_deps chamado em _spawn_krea_worker (apos clone)
# ===========================================================================
APP = "universal_app.py"
src = open(APP, encoding="utf-8").read()

# re-embutir worker corrigido
worker = open(wp, encoding="utf-8").read()
assert chr(39) * 3 not in worker and chr(34) * 3 not in worker, "worker tem triplas"
m = re.search(r'KREA2_WORKER_SRC = r"""(.*?)"""', src, re.S)
assert m, "KREA2_WORKER_SRC"
src = src.replace(m.group(0), 'KREA2_WORKER_SRC = r"""' + worker + '"""')
print("[OK] worker re-embutido (%d chars)" % len(worker))

# inserir _ensure_wan2gp_deps antes de _spawn_krea_worker
anchor = src.find("def _spawn_krea_worker(ckpt_path, model_name, progress_cb=None):")
assert anchor != -1, "spawn anchor"
deps_fn = '''def _ensure_wan2gp_deps(progress_cb=None):
    """Instala requirements.txt do Wan2GP + mmgp/gradio + pins (fiel aos notebooks).
    Roda ANTES do spawn do worker — o worker sobe rapido e sem falta de modulo."""
    if not os.path.exists(WAN2GP_DIR):
        raise RuntimeError("Wan2GP nao clonado.")
    reqs_txt = os.path.join(WAN2GP_DIR, "requirements.txt")
    if os.path.exists(reqs_txt):
        try:
            if progress_cb:
                progress_cb(0.05, 1.0, "Instalando requirements do Wan2GP (smplfitter, mmgp...)...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                                   "--timeout", "120", "--retries", "5", "-r", reqs_txt], timeout=1200)
        except Exception as e:
            print("  WARN pip requirements:", str(e)[:150])
    try:
        import mmgp  # noqa: F401
        import gradio  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "mmgp", "gradio"], timeout=600)
    try:
        import numpy as np2
        if np2.__version__ != "2.3.5":
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==2.3.5"], timeout=300)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==2.3.5"], timeout=300)
    try:
        import optimum.quanto
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "optimum-quanto==0.2.4"], timeout=300)


'''
src = src[:anchor] + deps_fn + src[anchor:]

# chamar _ensure_wan2gp_deps dentro do _spawn_krea_worker, depois do clone
old_clone = '''    if not os.path.exists(WAN2GP_DIR):
        if progress_cb:
            progress_cb(0.10, 1.0, "Clonando Wan2GP...")
        subprocess.check_call(["git", "clone", "-q", "https://github.com/DeepBeepMeep/Wan2GP.git", WAN2GP_DIR], timeout=900)
    _write_krea_worker_file()'''
new_clone = '''    if not os.path.exists(WAN2GP_DIR):
        if progress_cb:
            progress_cb(0.10, 1.0, "Clonando Wan2GP...")
        subprocess.check_call(["git", "clone", "-q", "https://github.com/DeepBeepMeep/Wan2GP.git", WAN2GP_DIR], timeout=900)
    _ensure_wan2gp_deps(progress_cb=progress_cb)
    _write_krea_worker_file()'''
assert old_clone in src, "clone block no spawn"
src = src.replace(old_clone, new_clone)
print("[OK] _spawn_krea_worker chama _ensure_wan2gp_deps")

open(APP, "w", encoding="utf-8").write(src)
print("universal_app.py atualizado — deps completas do Wan2GP")
print("tamanho:", len(src), "chars")
