# -*- coding: utf-8 -*-
import sys, os, re

APP = "universal_app.py"
src = open(APP, encoding="utf-8").read()

# 1. Re-adicionar WAN2GP_DIR
src = src.replace('COMFY_DIR = "/content/ComfyUI"\nAPI_PORT', 'COMFY_DIR = "/content/ComfyUI"\nWAN2GP_DIR = "/content/Wan2GP"\nAPI_PORT')

# 2. Re-adicionar "Krea 2" no BASE_MODEL_MAP
src = src.replace('"FLUX.2 Klein 4B-base": "flux2_klein",', '"FLUX.2 Klein 4B-base": "flux2_klein", "Krea 2": "krea2",')

# 3. Re-adicionar preset krea2 em FAMILY_PRESETS
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
    },\n'''
insert_preset_pos = src.find('    "flux2_klein": {')
assert insert_preset_pos != -1, "insert_preset_pos not found"
src = src[:insert_preset_pos] + preset_krea2 + src[insert_preset_pos:]

# 4. APP_VER bump para v2.2.20260817
src = src.replace('APP_VER = "v2.1.20260817"', 'APP_VER = "v2.2.20260817"')

# 5. STATE init — re-adicionar krea_model
src = src.replace('"loaded": False,', '"krea_model": None, "loaded": False,')

# 6. Bloco da Seção 7 MOTOR WAN2GP (versão oficial ultra-estável baseada nos notebooks do usuário)
sec7_code = '''# ============================================================================
# 7. MOTOR WAN2GP (Krea-2-Turbo Oficial INT8 — 100% Estavel na T4)
# ============================================================================
def _patch_krea2_main():
    """Aplica os patches de compatibilidade float16/T4 no krea2_main.py do Wan2GP."""
    try:
        _kp = os.path.join(WAN2GP_DIR, "models", "krea2", "krea2_main.py")
        if not os.path.exists(_kp):
            return
        subprocess.run(["git", "-C", WAN2GP_DIR, "checkout", "models/krea2/krea2_main.py"],
                       check=False, capture_output=True)
        with open(_kp, "r", encoding="utf-8") as _f:
            _src = _f.read().replace("\\r\\n", "\\n")
        modified = False

        if "        dtype = torch.bfloat16" in _src:
            _src = _src.replace("        dtype = torch.bfloat16",
                                "        # dtype = torch.bfloat16  # patched: float16 for T4")
            modified = True

        _old_tf = "    offload.load_model_data(transformer, model_filename, writable_tensors=False, preprocess_sd=preprocess_sd, default_dtype=dtype)"
        _new_tf = (
            "    def tf_preprocess(sd):\\n"
            "        sd = preprocess_sd(sd)\\n"
            "        return {k: v.to(dtype) if (isinstance(v, torch.Tensor) and v.is_floating_point()) else v for k, v in sd.items()}\\n"
            "    offload.load_model_data(transformer, model_filename, writable_tensors=False, preprocess_sd=tf_preprocess, default_dtype=dtype)"
        )
        if _old_tf in _src:
            _src = _src.replace(_old_tf, _new_tf)
            modified = True

        _old_te = (
            "    offload.load_model_data(\\n"
            "        text_encoder.language_model,\\n"
            "        text_encoder_filename,\\n"
            "        writable_tensors=False,\\n"
            "        default_dtype=dtype,\\n"
            "        preprocess_sd=_build_krea2_text_encoder_preprocessor(config.text_config),\\n"
            "    )"
        )
        _new_te = (
            "    _orig_te_preprocess = _build_krea2_text_encoder_preprocessor(config.text_config)\\n"
            "    def te_preprocess(sd):\\n"
            "        sd = _orig_te_preprocess(sd)\\n"
            "        return {k: v.to(dtype) if (isinstance(v, torch.Tensor) and v.is_floating_point()) else v for k, v in sd.items()}\\n"
            "    offload.load_model_data(\\n"
            "        text_encoder.language_model,\\n"
            "        text_encoder_filename,\\n"
            "        writable_tensors=False,\\n"
            "        default_dtype=dtype,\\n"
            "        preprocess_sd=te_preprocess,\\n"
            "    )"
        )
        if _old_te in _src:
            _src = _src.replace(_old_te, _new_te)
            modified = True

        _old_vae = "    offload.load_model_data(vae, filename, writable_tensors=False, default_dtype=None, preprocess_sd=preprocess_sd)"
        _new_vae = (
            "    _orig_vae_pp = preprocess_sd\\n"
            "    def vae_preprocess(sd):\\n"
            "        sd = _orig_vae_pp(sd) if _orig_vae_pp else sd\\n"
            "        return {k: v.to(dtype) if (isinstance(v, torch.Tensor) and v.is_floating_point()) else v for k, v in sd.items()}\\n"
            "    offload.load_model_data(vae, filename, writable_tensors=False, default_dtype=None, preprocess_sd=vae_preprocess)"
        )
        if _old_vae in _src:
            _src = _src.replace(_old_vae, _new_vae)
            modified = True

        if "def __call__(self, text=None" not in _src:
            call_override = (
                "    def __call__(self, text=None, images=None, videos=None, **kwargs):\\n"
                "        if images is None and videos is None and text is not None:\\n"
                "            return self.tokenizer(text, **kwargs)\\n"
                "        return super().__call__(text=text, images=images, videos=videos, **kwargs)\\n"
            )
            target = 'ProcessorMixin.__init__(self, image_processor, tokenizer, chat_template=getattr(tokenizer, "chat_template", None))'
            if target in _src:
                _src = _src.replace(target, target + "\\n" + call_override)
                modified = True

        if modified:
            with open(_kp, "w", encoding="utf-8") as _f:
                _f.write(_src)
            print("  krea2_main.py patched (float16 + preprocess_sd + processor fix).")
        for mod_key in list(sys.modules.keys()):
            if "krea2" in mod_key:
                sys.modules.pop(mod_key, None)
    except Exception as e:
        print("  Warn patch krea2_main:", e)

def _patch_transformers_docstring():
    try:
        import transformers.utils as _tu
        _orig = _tu.auto_docstring
        if getattr(_orig, "_krea_docstring_patched", False):
            return
        import inspect
        try:
            _accepted = set(inspect.signature(_orig).parameters)
        except Exception:
            _accepted = {"obj", "custom_intro", "custom_args", "checkpoint"}

        def _safe(*args, **kwargs):
            call_kwargs = {k: v for k, v in kwargs.items() if k in _accepted}
            try:
                if args:
                    return _orig(*args, **call_kwargs)
                return _orig(**call_kwargs)
            except (ValueError, TypeError):
                if args:
                    return args[0]
                def _identity(f):
                    return f
                return _identity

        _safe._krea_docstring_patched = True
        _tu.auto_docstring = _safe
    except Exception:
        pass

def _ensure_wan2gp_requirements(progress_cb=None):
    reqs = ["mmgp", "gradio"]
    missing = []
    for pkg in reqs:
        try:
            __import__(pkg)
        except Exception:
            missing.append(pkg)
    if missing:
        if progress_cb:
            progress_cb(0.21, 1.0, "Instalando dependencias do Wan2GP...")
        align_pkgs = ["numpy==2.3.5", "optimum-quanto==0.2.4"]
        install_list = list(set(missing + align_pkgs))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "-q"] + install_list)

def download_krea_models(progress_cb=None):
    """Baixa o modelo Krea-2-Turbo oficial INT8 da Hugging Face (DeepBeepMeep/krea-2)."""
    from huggingface_hub import hf_hub_download
    repo = "DeepBeepMeep/krea-2"
    qwen_repo = "DeepBeepMeep/Qwen_image"
    model_dir = os.path.join(WAN2GP_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Transformer (~12.5 GB)
    tf_path = os.path.join(model_dir, "Krea2Turbo_quanto_bf16_int8.safetensors")
    if not os.path.exists(tf_path):
        if progress_cb:
            progress_cb(0.25, 1.0, "Baixando Krea2-Turbo Transformer INT8 (12.5GB)...")
        hf_hub_download(repo_id=repo, filename="Krea2Turbo_quanto_bf16_int8.safetensors",
                        local_dir=model_dir, local_dir_use_symlinks=False)

    # 2. Text Encoder (~4 GB)
    te_dir = os.path.join(model_dir, "Qwen3-VL-4B-Instruct")
    os.makedirs(te_dir, exist_ok=True)
    te_files = ["Qwen3-VL-4B-Instruct_quanto_bf16_int8.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
    for f in te_files:
        fp = os.path.join(te_dir, f)
        if not os.path.exists(fp):
            if progress_cb:
                progress_cb(0.55, 1.0, f"Baixando TE Qwen3-VL-4B ({f})...")
            hf_hub_download(repo_id=repo, filename=f"Qwen3-VL-4B-Instruct/{f}",
                            local_dir=model_dir, local_dir_use_symlinks=False)

    # 3. VAE (~1.5 GB)
    vae_files = ["qwen_vae.safetensors", "qwen_vae_config.json"]
    for f in vae_files:
        fp = os.path.join(model_dir, f)
        if not os.path.exists(fp):
            if progress_cb:
                progress_cb(0.85, 1.0, f"Baixando VAE Qwen ({f})...")
            hf_hub_download(repo_id=qwen_repo, filename=f,
                            local_dir=model_dir, local_dir_use_symlinks=False)

def load_wan2gp_krea(cfg, progress_cb=None):
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch sem suporte CUDA / GPU indisponível nesta sessão. Alterne o Runtime para GPU (T4/L4/A100) no menu do Colab.")
    
    # Previne que o TensorFlow consuma RAM
    for mod_name in list(sys.modules):
        if 'tensorflow' in mod_name or 'tf' == mod_name:
            try: del sys.modules[mod_name]
            except Exception: pass
    gc.collect()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.6"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    if progress_cb:
        progress_cb(0.10, 1.0, "Verificando repositório Wan2GP...")
    if not os.path.exists(WAN2GP_DIR):
        if progress_cb:
            progress_cb(0.15, 1.0, "Clonando Wan2GP...")
        subprocess.check_call(["git", "clone", "-q", "https://github.com/DeepBeepMeep/Wan2GP.git", WAN2GP_DIR], timeout=900)

    _patch_krea2_main()
    _ensure_wan2gp_requirements(progress_cb=progress_cb)
    _patch_transformers_docstring()

    download_krea_models(progress_cb=progress_cb)

    if WAN2GP_DIR not in sys.path:
        sys.path.insert(0, WAN2GP_DIR)

    old_cwd = os.getcwd()
    os.chdir(WAN2GP_DIR)
    try:
        if progress_cb:
            progress_cb(0.60, 1.0, "Importando módulos Wan2GP...")
        from mmgp import offload
        from shared.utils import files_locator as fl
        from models.krea2.krea2_handler import family_handler
        fl.set_checkpoints_paths(["models", "ckpts", "."])

        transformer_path = os.path.join("models", "Krea2Turbo_quanto_bf16_int8.safetensors")
        text_encoder_path = os.path.join("models", "Qwen3-VL-4B-Instruct", "Qwen3-VL-4B-Instruct_quanto_bf16_int8.safetensors")

        dtype = torch.float16
        base_model_type = "krea2_turbo"
        if progress_cb:
            progress_cb(0.70, 1.0, "Carregando Krea-2-Turbo INT8 na GPU T4...")
        model_def = family_handler.query_model_def(base_model_type, {})
        krea_model, pipe = family_handler.load_model(
            model_filename=transformer_path,
            model_type=base_model_type,
            base_model_type=base_model_type,
            model_def=model_def,
            quantizeTransformer=False,
            dtype=dtype, VAE_dtype=dtype,
            text_encoder_filename=text_encoder_path,
        )

        if progress_cb:
            progress_cb(0.90, 1.0, "Otimizando VRAM com MMGP Offload (Colab T4 optimized)...")
        offload.profile(
            pipe, profile_no=2, quantizeTransformer=False,
            convertWeightsFloatTo=dtype,
            pinnedMemory=False,
            asyncTransfers=False,
            budgets={"transformer": 10000, "text_encoder": 4000, "vae": 1500, "*": 500},
        )
        offload.shared_state["_attention"] = "sdpa"

        STATE["krea_model"] = krea_model
        STATE["pipe"] = pipe
        STATE["backend"] = "wan2gp"
        STATE["family"] = "krea2"
        STATE["loaded"] = True
        STATE["config"] = {"label": "Krea-2-Turbo (Official INT8)", "backend": "wan2gp", "family": "krea2"}
        if progress_cb:
            progress_cb(1.0, 1.0, "Krea-2-Turbo Oficial pronto!")
    finally:
        os.chdir(old_cwd)

'''

sec8_pos = src.find("# ============================================================================\n# 8. MOTOR COMFYUI")
assert sec8_pos != -1, "sec8_pos not found"
src = src[:sec8_pos] + sec7_code + "\n" + src[sec8_pos:]

# 7. Função _gen_wan2gp
wan_code = '''def _gen_wan2gp(prompt, negative, steps, width, height, cfg, seed, progress_cb=None):
    krea_model = STATE["krea_model"]
    if hasattr(krea_model, "_interrupt"):
        krea_model._interrupt = False
    if hasattr(krea_model, "pipeline") and hasattr(krea_model.pipeline, "_interrupt"):
        krea_model.pipeline._interrupt = False

    guide = float(cfg) if (cfg is not None and float(cfg) > 0) else 0.0
    n_prompt = negative.strip() if (negative and negative.strip()) else None

    def cb(step_idx, latent, is_start, override_num_inference_steps=None, **kw):
        if progress_cb:
            progress_cb(step_idx + 1, int(steps), f"Krea2 passo {step_idx + 1}/{steps}")

    with torch.inference_mode():
        result = krea_model.generate(
            seed=int(seed), input_prompt=prompt, n_prompt=n_prompt,
            sampling_steps=int(steps), width=int(width), height=int(height),
            guide_scale=guide, batch_size=1, callback=cb, loras_slists={"phase1": []}
        )
    if result is None:
        raise RuntimeError("Geracao Krea2 retornou None.")
    if torch.is_tensor(result):
        result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
    img = Image.fromarray(result[:, 0].permute(1, 2, 0).numpy())
    del result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return img

'''
mk_cb_pos = src.find("def _mk_cb(cb):")
assert mk_cb_pos != -1, "mk_cb_pos not found"
src = src[:mk_cb_pos] + wan_code + src[mk_cb_pos:]

# 8. Dispatcher da geração em run_generation
gen_diff_pos = src.find('    if backend == "comfy":')
assert gen_diff_pos != -1, "gen_diff_pos not found"
wan_dispatch = '''    if backend == "wan2gp":
        if init_image is not None:
            raise RuntimeError("Krea-2-Turbo nao suporta img2img diretamente.")
        return _gen_wan2gp(prompt, negative, steps, width, height, cfg, seed, progress_cb)
'''
src = src[:gen_diff_pos] + wan_dispatch + src[gen_diff_pos:]

# 9. Unload_current_model: adicionar "krea_model"
src = src.replace('for key in ["pipe"]:', 'for key in ["pipe", "krea_model"]:')
src = src.replace('STATE["pipe"] = None', 'STATE["pipe"] = None\n    STATE["krea_model"] = None')

# 10. Guard de Krea2 custom no CivitAI
guard_old = 'if "krea" in (str(base_model) + " " + str(family)).lower():\n            return "O suporte a modelos Krea-2 foi removido (causava desconexão por OOM no Colab). Use SDXL, Pony, Illustrious, NoobAI, Anima ou FLUX.", None'
guard_new = 'if any(k in str(base_model).lower() for k in ["krea 2", "krea2", "krea-2"]):\n            return "Checkpoints Krea-2 customizados do Civitai (24GB FP8) exigem +30GB RAM e nao rodam na T4 (12GB RAM). Para usar o Krea-2 na GPU T4 do Colab, utilize o modelo oficial Krea-2-Turbo INT8 (disponivel no botao abaixo na aba Modelo).", None'
assert guard_old in src, "guard_old not found"
src = src.replace(guard_old, guard_new)

# 11. load_krea (botão/função oficial do Krea-2-Turbo)
load_krea_old = '''def load_krea(progress_cb=None):
    return "O suporte a modelos Krea-2 foi removido (causava desconexao por OOM no Colab). Use SDXL, Pony, Illustrious, NoobAI, Anima ou FLUX.", None'''

load_krea_new = '''def load_krea(progress_cb=None):
    try:
        unload_current_model()
        load_wan2gp_krea({}, progress_cb=progress_cb)
        return "Krea-2-Turbo Oficial (INT8) carregado com sucesso!", "krea2"
    except Exception as e:
        traceback.print_exc()
        return "Erro ao carregar Krea-2-Turbo: " + str(e), None'''
assert load_krea_old in src, "load_krea_old not found"
src = src.replace(load_krea_old, load_krea_new)

# 12. CIVITAI_EXAMPLES Krea2 (adicionar de volta como referência informativa)
ex_old = '    ("Arthemy Western Art", "https://civitai.com/models/2241572"),'
ex_new = '    ("Arthemy Western Art", "https://civitai.com/models/2241572"),\n    ("Arthemy Comics Krea2 (Aviso: +30GB RAM)", "https://civitai.com/models/2759057"),'
assert ex_old in src, "ex_old not found"
src = src.replace(ex_old, ex_new)

# 13. Subtitle Wan2GP
src = src.replace('Diffusers + ComfyUI', 'Diffusers + ComfyUI + Wan2GP')

with open(APP, "w", encoding="utf-8") as f:
    f.write(src)

print("universal_app.py atualizado com o Krea-2-Turbo Oficial INT8! Tamanho:", len(src), "chars")
