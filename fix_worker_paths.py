# -*- coding: utf-8 -*-
"""Corrige os bugs do worker Krea-2: path do Wan2GP (--wan2gp) e TE path (WAN2GP_DIR/models)."""
import re

APP = "universal_app.py"
src = open(APP, encoding="utf-8").read()

# 1. Re-embutir o worker corrigido (krea2_worker.py -> KREA2_WORKER_SRC)
worker = open("krea2_worker.py", encoding="utf-8").read()
assert chr(39) * 3 not in worker and chr(34) * 3 not in worker, "worker tem triplas!"
m = re.search(r'KREA2_WORKER_SRC = r"""(.*?)"""', src, re.S)
assert m, "KREA2_WORKER_SRC nao encontrado"
src = src.replace(m.group(0), 'KREA2_WORKER_SRC = r"""' + worker + '"""')
print("[OK] worker re-embutido (%d chars)" % len(worker))

# 2. _spawn_krea_worker: TE path correto (WAN2GP_DIR/models) + env WAN2GP_DIR + --wan2gp no cmd
old_te = '    te = os.path.join(MODELS_DIR, "Qwen3-VL-4B-Instruct", "Qwen3-VL-4B-Instruct_quanto_bf16_int8.safetensors")'
new_te = '    te = os.path.join(WAN2GP_DIR, "models", "Qwen3-VL-4B-Instruct", "Qwen3-VL-4B-Instruct_quanto_bf16_int8.safetensors")'
assert old_te in src, "te path"
src = src.replace(old_te, new_te)

old_env = '    env = dict(os.environ)\n    env["KREA2_WORKER_PROGRESS"] = KREA2_WORKER_PROGRESS\n    env["KREA2_WORKER_LOG"] = wlog'
new_env = '    env = dict(os.environ)\n    env["KREA2_WORKER_PROGRESS"] = KREA2_WORKER_PROGRESS\n    env["KREA2_WORKER_LOG"] = wlog\n    env["WAN2GP_DIR"] = WAN2GP_DIR'
assert old_env in src, "env block"
src = src.replace(old_env, new_env)

old_cmd = '           "--ckpt", ckpt_path, "--te", te, "--model", str(model_name), "--port", str(port)]'
new_cmd = '           "--ckpt", ckpt_path, "--te", te, "--model", str(model_name), "--port", str(port),\n           "--wan2gp", WAN2GP_DIR]'
assert old_cmd in src, "cmd block"
src = src.replace(old_cmd, new_cmd)
print("[OK] _spawn_krea_worker: TE path + env + --wan2gp")

# 3. _ensure_krea_te_vae (TE+VAE sem transformer) — inserido antes de download_krea_official
anchor = src.find("def download_krea_official(progress_cb=None):")
assert anchor != -1, "download_krea_official"
ensure_fn = '''def _ensure_krea_te_vae(progress_cb=None):
    """Baixa APENAS Text Encoder Qwen3-VL-4B + VAE Qwen (usado por custom e oficial)."""
    from huggingface_hub import hf_hub_download
    repo = "DeepBeepMeep/krea-2"
    qwen_repo = "DeepBeepMeep/Qwen_image"
    if not os.path.exists(WAN2GP_DIR):
        raise RuntimeError("Wan2GP nao clonado.")
    model_dir = os.path.join(WAN2GP_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    te_dir = os.path.join(model_dir, "Qwen3-VL-4B-Instruct")
    os.makedirs(te_dir, exist_ok=True)
    te_files = ["Qwen3-VL-4B-Instruct_quanto_bf16_int8.safetensors", "config.json",
                "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
    for f in te_files:
        fp = os.path.join(te_dir, f)
        if not os.path.exists(fp):
            if progress_cb:
                progress_cb(0.30, 1.0, "Baixando Text Encoder Qwen3-VL-4B (" + f + ")...")
            hf_hub_download(repo_id=repo, filename="Qwen3-VL-4B-Instruct/" + f,
                            local_dir=model_dir, local_dir_use_symlinks=False)
    for f in ["qwen_vae.safetensors", "qwen_vae_config.json"]:
        fp = os.path.join(model_dir, f)
        if not os.path.exists(fp):
            if progress_cb:
                progress_cb(0.80, 1.0, "Baixando VAE Qwen (" + f + ")...")
            hf_hub_download(repo_id=qwen_repo, filename=f,
                            local_dir=model_dir, local_dir_use_symlinks=False)
    import shutil as _sh
    for d in [os.path.join(model_dir, ".cache"), te_dir + os.sep + ".cache"]:
        if os.path.exists(d):
            try:
                _sh.rmtree(d)
            except Exception:
                pass


'''
src = src[:anchor] + ensure_fn + src[anchor:]
print("[OK] _ensure_krea_te_vae inserida")

# 4. download_krea_official usa _ensure_krea_te_vae (remove duplicacao TE/VAE)
old_dko_head = '''    model_dir = os.path.join(WAN2GP_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    tf_path = os.path.join(model_dir, "Krea2Turbo_quanto_bf16_int8.safetensors")
    if not os.path.exists(tf_path):
        if progress_cb:
            progress_cb(0.15, 1.0, "Baixando Krea2-Turbo Transformer INT8 (12.5GB)...")
        hf_hub_download(repo_id=repo, filename="Krea2Turbo_quanto_bf16_int8.safetensors",
                        local_dir=model_dir, local_dir_use_symlinks=False)
    te_dir = os.path.join(model_dir, "Qwen3-VL-4B-Instruct")
    os.makedirs(te_dir, exist_ok=True)
    te_files = ["Qwen3-VL-4B-Instruct_quanto_bf16_int8.safetensors", "config.json",
                "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
    for f in te_files:
        fp = os.path.join(te_dir, f)
        if not os.path.exists(fp):
            if progress_cb:
                progress_cb(0.45, 1.0, "Baixando Text Encoder Qwen3-VL-4B (" + f + ")...")
            hf_hub_download(repo_id=repo, filename="Qwen3-VL-4B-Instruct/" + f,
                            local_dir=model_dir, local_dir_use_symlinks=False)
    for f in ["qwen_vae.safetensors", "qwen_vae_config.json"]:
        fp = os.path.join(model_dir, f)
        if not os.path.exists(fp):
            if progress_cb:
                progress_cb(0.80, 1.0, "Baixando VAE Qwen (" + f + ")...")
            hf_hub_download(repo_id=qwen_repo, filename=f,
                            local_dir=model_dir, local_dir_use_symlinks=False)
    import shutil as _sh
    for d in [os.path.join(model_dir, ".cache"), os.path.join(model_dir, "Qwen3-VL-4B-Instruct", ".cache")]:
        if os.path.exists(d):
            try:
                _sh.rmtree(d)
            except Exception:
                pass
    return tf_path'''
new_dko_head = '''    model_dir = os.path.join(WAN2GP_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    _ensure_krea_te_vae(progress_cb=progress_cb)
    tf_path = os.path.join(model_dir, "Krea2Turbo_quanto_bf16_int8.safetensors")
    if not os.path.exists(tf_path):
        if progress_cb:
            progress_cb(0.15, 1.0, "Baixando Krea2-Turbo Transformer INT8 (12.5GB)...")
        hf_hub_download(repo_id=repo, filename="Krea2Turbo_quanto_bf16_int8.safetensors",
                        local_dir=model_dir, local_dir_use_symlinks=False)
    return tf_path'''
assert old_dko_head in src, "download_krea_official body"
src = src.replace(old_dko_head, new_dko_head)
print("[OK] download_krea_official reusa _ensure_krea_te_vae")

# 5. load_krea_custom: garantir TE/VAE antes do spawn
old_custom = '''        base_model = version.get("baseModel") or "Krea 2"
        trained_words = version.get("trainedWords") or []
        write_model_meta(local_path, base_model, model_name, "krea2", version.get("name"), trained_words, target)
        w = _spawn_krea_worker(local_path, model_name, progress_cb=progress_cb)'''
new_custom = '''        base_model = version.get("baseModel") or "Krea 2"
        trained_words = version.get("trainedWords") or []
        write_model_meta(local_path, base_model, model_name, "krea2", version.get("name"), trained_words, target)
        if not os.path.exists(WAN2GP_DIR):
            if progress_cb:
                progress_cb(0.08, 1.0, "Clonando Wan2GP (1-2 min)...")
            subprocess.check_call(["git", "clone", "-q", "https://github.com/DeepBeepMeep/Wan2GP.git", WAN2GP_DIR], timeout=900)
        _ensure_krea_te_vae(progress_cb=progress_cb)
        w = _spawn_krea_worker(local_path, model_name, progress_cb=progress_cb)'''
assert old_custom in src, "load_krea_custom spawn"
src = src.replace(old_custom, new_custom)
print("[OK] load_krea_custom garante TE/VAE")

# 6. load_krea (oficial) tambem usa _ensure_krea_te_vae (via download_krea_official ja refatorado)
open(APP, "w", encoding="utf-8").write(src)
print("universal_app.py corrigido — bugs do worker resolvidos")
print("tamanho:", len(src), "chars")
