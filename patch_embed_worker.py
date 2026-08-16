# -*- coding: utf-8 -*-
"""Reintegra o Krea-2-Turbo (worker isolado fiel aos notebooks validados) no universal_app.py.
Le o krea2_worker.py do disco e o embute como KREA2_WORKER_SRC (r\"\"\"...\"\"\" sem triplas internas)."""
import sys, os

APP = "universal_app.py"
src = open(APP, encoding="utf-8").read()

# Worker lido do disco (sem triplas internas — validado)
WORKER_SRC = open("krea2_worker.py", encoding="utf-8").read()
assert chr(39) * 3 not in WORKER_SRC, "worker contem aspas triplas simples!"
assert chr(34) * 3 not in WORKER_SRC, "worker contem aspas triplas duplas!"
print("worker embed chars:", len(WORKER_SRC))

# ===========================================================================
# 1. APP_VER bump
# ===========================================================================
src = src.replace('APP_VER = "v2.3.20260817"', 'APP_VER = "v2.4.20260817"')

# ===========================================================================
# 2. WAN2GP_DIR
# ===========================================================================
src = src.replace('COMFY_DIR = "/content/ComfyUI"\nAPI_PORT', 'COMFY_DIR = "/content/ComfyUI"\nWAN2GP_DIR = "/content/Wan2GP"\nAPI_PORT')

# ===========================================================================
# 3. STATE init
# ===========================================================================
src = src.replace('"loaded": False,', '"krea_model": None, "krea_worker": None, "loaded": False,')

# ===========================================================================
# 4. Bloco helpers + constante (antes da secao 8 COMFYUI)
# ===========================================================================
helpers = '\nKREA2_WORKER_PORT = 7862\n'
helpers += 'KREA2_WORKER_PROGRESS = os.path.join(APP_DIR, "krea2_worker_progress.json")\n'
helpers += 'KREA2_WORKER_SRC = r"""' + WORKER_SRC + '"""\n\n\n'
helpers += '''def _write_krea_worker_file():
    path = os.path.join(APP_DIR, "krea2_worker.py")
    cur = ""
    if os.path.exists(path):
        try:
            cur = open(path, encoding="utf-8").read()
        except Exception:
            cur = ""
    if cur != KREA2_WORKER_SRC:
        with open(path, "w", encoding="utf-8") as f:
            f.write(KREA2_WORKER_SRC)
    return path


def _kill_krea_worker():
    w = STATE.get("krea_worker")
    if w and w.get("proc"):
        try:
            w["proc"].terminate()
        except Exception:
            pass
        try:
            w["proc"].wait(timeout=10)
        except Exception:
            try:
                w["proc"].kill()
            except Exception:
                pass
    subprocess.run(["pkill", "-9", "-f", "krea2_worker.py"], capture_output=True)
    STATE["krea_worker"] = None


def _krea_worker_health(port, timeout=3):
    try:
        r = requests.get("http://127.0.0.1:" + str(port) + "/health", timeout=timeout)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


def _spawn_krea_worker(ckpt_path, model_name, progress_cb=None):
    """Sobe o worker Krea-2-Turbo em processo limpo (padrao dos notebooks validados).
    Se o OOM memcg 12Gi matar o worker, SO ele morre — kernel/sessao/app continuam vivos."""
    if not os.path.exists(ckpt_path):
        raise RuntimeError("Checkpoint Krea2 nao encontrado: " + ckpt_path)
    te = os.path.join(MODELS_DIR, "Qwen3-VL-4B-Instruct", "Qwen3-VL-4B-Instruct_quanto_bf16_int8.safetensors")
    if not os.path.exists(WAN2GP_DIR):
        if progress_cb:
            progress_cb(0.10, 1.0, "Clonando Wan2GP...")
        subprocess.check_call(["git", "clone", "-q", "https://github.com/DeepBeepMeep/Wan2GP.git", WAN2GP_DIR], timeout=900)
    _write_krea_worker_file()
    _kill_krea_worker()
    port = KREA2_WORKER_PORT
    while _krea_worker_health(port):
        time.sleep(1)
    wlog = os.path.join(APP_DIR, "krea2_worker.log")
    try:
        if os.path.exists(wlog):
            os.remove(wlog)
    except Exception:
        pass
    env = dict(os.environ)
    env["KREA2_WORKER_PROGRESS"] = KREA2_WORKER_PROGRESS
    env["KREA2_WORKER_LOG"] = wlog
    logf = open(wlog, "w", encoding="utf-8")
    cmd = [sys.executable, "-u", os.path.join(APP_DIR, "krea2_worker.py"),
           "--ckpt", ckpt_path, "--te", te, "--model", str(model_name), "--port", str(port)]
    proc = subprocess.Popen(cmd, cwd=APP_DIR, env=env, stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)
    t0 = time.time()
    ok = False
    while time.time() - t0 < 600:
        if _krea_worker_health(port):
            ok = True
            break
        if proc.poll() is not None:
            break
        if progress_cb:
            tail = _read_log_tail(wlog, 1).strip()
            progress_cb(min((time.time() - t0) / 700, 0.85), 1.0, "Worker Krea2 subindo | " + tail)
        time.sleep(3)
    if not ok:
        try:
            proc.terminate()
        except Exception:
            pass
        tail = _read_log_tail(wlog, 30)
        raise RuntimeError("Worker Krea2 nao subiu:\\n" + tail)
    if progress_cb:
        progress_cb(0.9, 1.0, "Worker Krea2 pronto (isolado)")
    try:
        logf.close()
    except Exception:
        pass
    return {"proc": proc, "port": port, "log": wlog, "ts": time.time()}


def _proxy_worker_generate(prompt, negative, steps, width, height, cfg, seed, progress_cb=None):
    """Proxy de geracao para o worker Krea-2 (thread + poll de progresso)."""
    w = STATE.get("krea_worker")
    if not w:
        raise RuntimeError("Worker Krea2 inativo.")
    port = w.get("port") or KREA2_WORKER_PORT
    url = "http://127.0.0.1:" + str(port) + "/generate"
    payload = {
        "prompt": prompt, "negative": negative or "", "steps": int(steps),
        "width": int(width), "height": int(height), "cfg": float(cfg or 0),
        "seed": int(seed),
    }
    try:
        if os.path.exists(KREA2_WORKER_PROGRESS):
            os.remove(KREA2_WORKER_PROGRESS)
    except Exception:
        pass
    result = {}

    def _do():
        try:
            result["resp"] = requests.post(url, json=payload, timeout=3600)
        except Exception as e:
            result["err"] = e

    t = threading.Thread(target=_do, daemon=True)
    t.start()
    last_pct = -1.0
    while t.is_alive():
        try:
            pj = json.load(open(KREA2_WORKER_PROGRESS, encoding="utf-8"))
            done = float(pj.get("done", 0))
            total = float(pj.get("total", max(1, int(steps))))
            pct = done / max(1.0, total)
            if progress_cb and pct != last_pct:
                progress_cb(min(pct, 1.0), 1.0, str(pj.get("desc", "Krea2 ...")))
                last_pct = pct
        except Exception:
            pass
        time.sleep(1.0)
    t.join(timeout=5)
    if result.get("err"):
        raise RuntimeError("Worker Krea2 falhou: " + str(result["err"])[:300])
    r = result["resp"]
    data = r.json()
    if data.get("status") != "ok":
        raise RuntimeError("Worker Krea2 erro: " + str(data.get("error", r.text[:400])))
    img_b64 = data.get("image")
    if not img_b64:
        raise RuntimeError("Worker Krea2 nao retornou imagem.")
    import io as _bio
    from PIL import Image as _PIL
    try:
        img = _PIL.open(_bio.BytesIO(base64.b64decode(img_b64))).convert("RGB")
    except Exception as e:
        raise RuntimeError("Falha ao decodificar PNG do worker: " + str(e)[:200])
    if progress_cb:
        progress_cb(1.0, 1.0, "Krea2 concluido (worker)")
    return img

'''
anchor = src.find("# ============================================================================\n# 8. MOTOR COMFYUI")
assert anchor != -1, "anchor secao 8 nao encontrada"
src = src[:anchor] + helpers + "\n" + src[anchor:]

# ===========================================================================
# 5. load_krea -> fluxo completo (download HF + spawn worker)
# ===========================================================================
old_load_krea = '''def load_krea(progress_cb=None):
    return "O suporte ao modelo Krea-2 foi DESCONTINUADO permanentemente (causava desconexao por OOM no Colab). Use SDXL, Pony, Illustrious, NoobAI, Anima ou FLUX.", None'''
new_load_krea = '''def download_krea_official(progress_cb=None):
    """Baixa o Krea-2-Turbo oficial INT8 (DeepBeepMeep/krea-2) — hf_hub_download com resume."""
    from huggingface_hub import hf_hub_download
    repo = "DeepBeepMeep/krea-2"
    qwen_repo = "DeepBeepMeep/Qwen_image"
    if not os.path.exists(WAN2GP_DIR):
        raise RuntimeError("Wan2GP nao clonado (chame load_krea para setup completo).")
    model_dir = os.path.join(WAN2GP_DIR, "models")
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
    return tf_path


def load_krea(progress_cb=None):
    """Carrega o Krea-2-Turbo Oficial INT8 em WORKER ISOLADO (padrao notebooks validados)."""
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU indisponivel — use Runtime > T4 GPU.")
        unload_current_model()
        if progress_cb:
            progress_cb(0.05, 1.0, "Preparando Wan2GP...")
        if not os.path.exists(WAN2GP_DIR):
            if progress_cb:
                progress_cb(0.08, 1.0, "Clonando Wan2GP (1-2 min)...")
            subprocess.check_call(["git", "clone", "-q", "https://github.com/DeepBeepMeep/Wan2GP.git", WAN2GP_DIR], timeout=900)
        ckpt = download_krea_official(progress_cb=progress_cb)
        w = _spawn_krea_worker(ckpt, "Krea-2-Turbo (INT8)", progress_cb=progress_cb)
        STATE["krea_worker"] = w
        STATE["backend"] = "wan2gp"
        STATE["family"] = "krea2"
        STATE["loaded"] = True
        STATE["krea_model"] = None
        STATE["pipe"] = None
        STATE["config"] = {"label": "Krea-2-Turbo (Official INT8)", "backend": "wan2gp", "family": "krea2"}
        if progress_cb:
            progress_cb(1.0, 1.0, "Krea-2-Turbo Oficial pronto (worker isolado)!")
        return "Krea-2-Turbo Oficial (INT8) carregado com sucesso!", "krea2"
    except Exception as e:
        traceback.print_exc()
        return "Erro ao carregar Krea-2-Turbo: " + str(e), None'''
assert old_load_krea in src, "old_load_krea not found"
src = src.replace(old_load_krea, new_load_krea)

# ===========================================================================
# 6. _gen_wan2gp -> proxy
# ===========================================================================
old_gen_pos = src.find('def _mk_cb(cb):')
assert old_gen_pos != -1, "mk_cb not found"
wan_gen = '''def _gen_wan2gp(prompt, negative, steps, width, height, cfg, seed, progress_cb=None):
    # Worker isolado: geracao via proxy HTTP (OOM mata o worker, nao a sessao)
    if STATE.get("krea_worker"):
        return _proxy_worker_generate(prompt, negative, steps, width, height, cfg, seed, progress_cb)
    raise RuntimeError("Krea-2 nao carregado (use o botao Krea-2-Turbo na aba Modelo).")

'''
src = src[:old_gen_pos] + wan_gen + src[old_gen_pos:]

# ===========================================================================
# 7. Branch wan2gp na geracao
# ===========================================================================
old_branch = '    if backend == "comfy":'
new_branch = '''    if backend == "wan2gp":
        if init_image is not None:
            raise RuntimeError("Krea-2-Turbo nao suporta img2img diretamente.")
        return _gen_wan2gp(prompt, negative, steps, width, height, cfg, seed, progress_cb)
    if backend == "comfy":'''
assert old_branch in src, "branch not found"
src = src.replace(old_branch, new_branch, 1)

# ===========================================================================
# 8. unload_current_model -> matar worker
# ===========================================================================
old_unload = '''def unload_current_model():
    for key in ["pipe"]:
        if STATE.get(key) is not None:
            try:
                del STATE[key]
            except Exception:
                pass
    STATE["pipe"] = None'''
new_unload = '''def unload_current_model():
    w = STATE.get("krea_worker")
    if w:
        try:
            requests.post("http://127.0.0.1:" + str(w.get("port") or KREA2_WORKER_PORT) + "/unload", timeout=5)
        except Exception:
            pass
        _kill_krea_worker()
    for key in ["pipe", "krea_model", "krea_worker"]:
        if STATE.get(key) is not None:
            try:
                del STATE[key]
            except Exception:
                pass
    STATE["pipe"] = None
    STATE["krea_model"] = None
    STATE["krea_worker"] = None'''
assert old_unload in src, "unload not found"
src = src.replace(old_unload, new_unload)

# ===========================================================================
# 9. Guard civitai -> orienta para o botao oficial
# ===========================================================================
old_guard = 'if any(k in str(base_model).lower() for k in ["krea 2", "krea2", "krea-2"]):\n            return "O suporte ao modelo Krea-2 foi DESCONTINUADO permanentemente (causava desconexao por OOM no Colab). Use SDXL, Pony, Illustrious, NoobAI, Anima ou FLUX.", None'
new_guard = 'if any(k in str(base_model).lower() for k in ["krea 2", "krea2", "krea-2"]):\n            return "Checkpoints Krea-2 customizados do Civitai exigem +30GB RAM. Para rodar Krea-2 na T4, use o botao \\"Krea-2-Turbo\\" (modelo oficial INT8) na aba Modelo.", None'
assert old_guard in src, "guard not found"
src = src.replace(old_guard, new_guard)

# ===========================================================================
# 10. Health API krea_worker
# ===========================================================================
old_health = '                                          "app_ver": APP_VER,\n'
new_health = '                                          "app_ver": APP_VER,\n                                          "krea_worker": bool(STATE.get("krea_worker")),\n'
assert old_health in src, "health not found"
src = src.replace(old_health, new_health, 1)

# ===========================================================================
# 11. Subtitle + exemplo
# ===========================================================================
src = src.replace('Diffusers + ComfyUI', 'Diffusers + ComfyUI + Wan2GP')
old_ex = '    ("Arthemy Western Art", "https://civitai.com/models/2241572"),'
new_ex = '    ("Arthemy Western Art", "https://civitai.com/models/2241572"),\n    ("Krea-2-Turbo Oficial (INT8, via botao)", "DeepBeepMeep/krea-2"),'
if old_ex in src:
    src = src.replace(old_ex, new_ex)

# ===========================================================================
# 12. Preset krea2
# ===========================================================================
preset_krea2 = '''    "krea2": {
        "label": "Krea-2-Turbo (Wan2GP INT8)", "diffusers_cls": None, "single_file": False,
        "backend": "wan2gp", "steps": {"min": 1, "max": 12, "default": 8}, "cfg": 0.0,
        "scheduler": "krea", "sampler": "krea",
        "resolutions": ["1024px (Standard)", "1536px (High)"],
        "default_res": "1024px (Standard)",
        "aspects": ["16:9 Landscape", "9:16 Portrait", "1:1 Square", "4:3 Standard", "3:4 Portrait"],
        "default_aspect": "16:9 Landscape",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "none", "controlnet": None, "cfg_rescale": 0.0,
        "notes": "Krea-2-Turbo via Wan2GP INT8 oficial (DeepBeepMeep/krea-2).",
    },
'''
anchor_preset = src.find('    "flux2_klein": {')
assert anchor_preset != -1, "preset anchor not found"
src = src[:anchor_preset] + preset_krea2 + src[anchor_preset:]

open(APP, "w", encoding="utf-8").write(src)
print("universal_app.py atualizado: Krea-2 worker reintegrado (v2.4.20260817)")
print("tamanho:", len(src), "chars")
