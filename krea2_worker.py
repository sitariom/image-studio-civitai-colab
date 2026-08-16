# -*- coding: utf-8 -*-
# Krea-2-Turbo Wan2GP worker — processo limpo e isolado (anti-OOM Colab).
# Baseado fielmente nos notebooks que funcionam na T4:
#   krea_2_turbo_colab_implementado.ipynb (Colab T4 12GB RAM)
#   krea-2-turbo-fast-text-to-image-generator_implementado.ipynb (Kaggle T4 x2)
# Uso: python -u krea2_worker.py --ckpt <path> [--te <path>] [--model <nome>] [--port 7862]
# API HTTP local:
#   GET  /health    -> {status, loaded, model, backend}
#   POST /generate  -> {prompt, negative, steps, width, height, cfg, seed} -> {status, image: base64 PNG}
#   POST /unload    -> encerra o processo (exit 0)

import os, sys, gc, io, json, time, base64, threading, subprocess, traceback, importlib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.5"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"

# Bloqueia TensorFlow (economiza ~1-2GB RAM; Colab pre-carrega TF no kernel)
class _TFBlocker:
    def find_module(self, name, path=None):
        if name == "tensorflow" or name.startswith("tensorflow."):
            return self
    def load_module(self, name):
        raise ImportError("TensorFlow blocked to save RAM")
sys.meta_path.insert(0, _TFBlocker())

import torch
from PIL import Image
import numpy as np

LOG_PATH = os.environ.get("KREA2_WORKER_LOG", "/content/krea2_worker.log")
PROG_PATH = os.environ.get("KREA2_WORKER_PROGRESS", "/content/krea2_worker_progress.json")
WAN2GP_DIR = os.environ.get("WAN2GP_DIR") or os.path.abspath("Wan2GP")


def log(msg):
    line = "[" + time.strftime("%H:%M:%S") + "] " + str(msg) + "\n"
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
    sys.stdout.write(line)
    sys.stdout.flush()


def progress(done, total, desc):
    try:
        with open(PROG_PATH, "w", encoding="utf-8") as f:
            json.dump({"done": done, "total": total, "desc": desc}, f)
    except Exception:
        pass


def patch_krea2_main():
    # Patches 100% fieis aos notebooks: float16 p/ T4 + preprocess_sd + processor fix.
    # Cobre 2 padroes do upstream (single-line antigo com modelPrefix e multiline novo).
    try:
        kp = os.path.join(WAN2GP_DIR, "models", "krea2", "krea2_main.py")
        if not os.path.exists(kp):
            log("krea2_main.py nao encontrado")
            return
        subprocess.run(["git", "-C", WAN2GP_DIR, "checkout", "models/krea2/krea2_main.py"],
                       check=False, capture_output=True)
        src = open(kp, encoding="utf-8").read().replace("\r\n", "\n")
        mod = False

        # 1) dtype bfloat16 comentado (T4 nao suporta bf16 nativo)
        if "        dtype = torch.bfloat16" in src:
            src = src.replace("        dtype = torch.bfloat16",
                              "        # dtype = torch.bfloat16  # patched: float16 for T4")
            mod = True

        # 2) Transformer preprocess (single-line, mesmo dos notebooks)
        old_tf = "    offload.load_model_data(transformer, model_filename, writable_tensors=False, preprocess_sd=preprocess_sd, default_dtype=dtype)"
        if old_tf in src:
            new_tf = (
                "    def tf_preprocess(*a):\n"
                "        sd = a[0] if a else {}\n"
                "        if isinstance(sd, tuple):\n"
                "            sd = sd[0] if sd else {}\n"
                "        if callable(preprocess_sd):\n"
                "            sd = preprocess_sd(sd)\n"
                "            if isinstance(sd, tuple):\n"
                "                sd = sd[0] if sd else {}\n"
                "        return {k: v.to(dtype) if (isinstance(v, torch.Tensor) and v.is_floating_point()) else v for k, v in sd.items()}\n"
                "    offload.load_model_data(transformer, model_filename, writable_tensors=False, preprocess_sd=tf_preprocess, default_dtype=dtype)"
            )
            src = src.replace(old_tf, new_tf)
            mod = True

        # 3) Text encoder preprocess (padrao notebook: modelPrefix="language_model")
        old_te = '    offload.load_model_data(text_encoder.language_model, text_encoder_filename, modelPrefix="language_model", writable_tensors=False, default_dtype=dtype)'
        if old_te in src:
            new_te = (
                "    def te_preprocess(*a):\n"
                "        sd = a[0] if a else {}\n"
                "        if isinstance(sd, tuple):\n"
                "            sd = sd[0] if sd else {}\n"
                "        return {k: v.to(dtype) if (isinstance(v, torch.Tensor) and v.is_floating_point()) else v for k, v in sd.items()}\n"
                '    offload.load_model_data(text_encoder.language_model, text_encoder_filename, modelPrefix="language_model", writable_tensors=False, preprocess_sd=te_preprocess, default_dtype=dtype)'
            )
            src = src.replace(old_te, new_te)
            mod = True
        # 3b) TE preprocess (padrao multiline do upstream atual)
        old_te2 = (
            "    offload.load_model_data(\n"
            "        text_encoder.language_model,\n"
            "        text_encoder_filename,\n"
            "        writable_tensors=False,\n"
            "        default_dtype=dtype,\n"
            "        preprocess_sd=_build_krea2_text_encoder_preprocessor(config.text_config),\n"
            "    )"
        )
        if old_te2 in src:
            new_te2 = (
                "    def te_preprocess(*a):\n"
                "        sd = a[0] if a else {}\n"
                "        if isinstance(sd, tuple):\n"
                "            sd = sd[0] if sd else {}\n"
                "        return {k: v.to(dtype) if (isinstance(v, torch.Tensor) and v.is_floating_point()) else v for k, v in sd.items()}\n"
                "    offload.load_model_data(\n"
                "        text_encoder.language_model,\n"
                "        text_encoder_filename,\n"
                "        modelPrefix=\"language_model\",\n"
                "        writable_tensors=False,\n"
                "        default_dtype=dtype,\n"
                "        preprocess_sd=te_preprocess,\n"
                "    )"
            )
            src = src.replace(old_te2, new_te2)
            mod = True

        # 4) VAE preprocess
        old_vae = "    offload.load_model_data(vae, filename, writable_tensors=False, default_dtype=None, preprocess_sd=preprocess_sd)"
        if old_vae in src:
            new_vae = (
                "    def vae_preprocess(*a):\n"
                "        sd = a[0] if a else {}\n"
                "        if isinstance(sd, tuple):\n"
                "            sd = sd[0] if sd else {}\n"
                "        out = {}\n"
                "        for _k, _v in sd.items():\n"
                "            _nk = _k\n"
                "            if _k.startswith('vae.'):\n"
                "                _nk = _k[len('vae.'):]\n"
                "            elif _k.startswith('model.'):\n"
                "                _nk = _k[len('model.'):]\n"
                "            out[_nk] = _v.to(dtype) if (isinstance(_v, torch.Tensor) and _v.is_floating_point()) else _v\n"
                "        return out\n"
                "    offload.load_model_data(vae, filename, writable_tensors=False, default_dtype=None, preprocess_sd=vae_preprocess)"
            )
            src = src.replace(old_vae, new_vae)
            mod = True

        # 5) Processor __call__ override (transformers >= 4.45)
        if "def __call__(self, text=None" not in src:
            call_override = (
                "    def __call__(self, text=None, images=None, videos=None, **kwargs):\n"
                "        if images is None and videos is None and text is not None:\n"
                "            return self.tokenizer(text, **kwargs)\n"
                "        return super().__call__(text=text, images=images, videos=videos, **kwargs)\n"
            )
            target = 'ProcessorMixin.__init__(self, image_processor, tokenizer, chat_template=getattr(tokenizer, "chat_template", None))'
            if target in src:
                src = src.replace(target, target + "\n" + call_override)
                mod = True

        if mod:
            open(kp, "w", encoding="utf-8").write(src)
            log("krea2_main.py patched (float16 T4 + preprocess + processor)")
    except Exception as e:
        log("patch krea2_main warn: " + str(e))


def patch_transformers_docstring():
    # auto_docstring identity (transformers 5.x) - nao bloqueia definicao de classe
    try:
        import transformers.utils as tu
        orig = tu.auto_docstring
        if getattr(orig, "_krea_worker_patched", False):
            return
        import inspect
        try:
            accepted = set(inspect.signature(orig).parameters)
        except Exception:
            accepted = {"obj", "custom_intro", "custom_args", "checkpoint"}
        def safe(*args, **kwargs):
            ck = {k: v for k, v in kwargs.items() if k in accepted}
            try:
                if args:
                    return orig(*args, **ck)
                return orig(**ck)
            except (ValueError, TypeError):
                if args:
                    return args[0]
                def identity(f):
                    return f
                return identity
        safe._krea_worker_patched = True
        tu.auto_docstring = safe
    except Exception:
        pass


def ensure_requirements():
    # 1) requirements.txt COMPLETO do Wan2GP (smplfitter, mmgp, ...) — fiel aos notebooks
    reqs_txt = os.path.join(WAN2GP_DIR, "requirements.txt")
    if os.path.exists(reqs_txt):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                                   "--timeout", "120", "--retries", "5", "-r", reqs_txt], timeout=900)
            log("requirements.txt do Wan2GP instalado")
        except Exception as e:
            log("pip requirements warn: " + str(e)[:150])
    # 2) FORCA os pins EXATOS do notebook (rebaixa/atualiza mesmo se outra versao ja
    #    instalada — o try/except de import nao pega versao errada ja presente)
    for _pin in ("mmgp==3.7.12", "gradio==5.29.0", "optimum-quanto==0.2.7",
                 "smplfitter==0.2.10", "torchao>=0.16.0"):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", _pin], timeout=600)
        except Exception as _e:
            log("pip " + _pin + " warn: " + str(_e)[:100])


def load_model(ckpt_path, te_path, model_name):
    # Carregamento 100% fiel ao notebook Colab: quantizeTransformer=False (modelo ja INT8)
    from mmgp import offload  # noqa: F401
    from shared.utils import files_locator as fl
    from models.krea2.krea2_handler import family_handler
    fl.set_checkpoints_paths(["models", "ckpts", "."])
    log("carregando: " + os.path.basename(ckpt_path))
    dtype = torch.float16
    base_model_type = "krea2_turbo"
    model_def = family_handler.query_model_def(base_model_type, {})
    krea_model, pipe = family_handler.load_model(
        model_filename=ckpt_path,
        model_type=base_model_type,
        base_model_type=base_model_type,
        model_def=model_def,
        quantizeTransformer=False,
        dtype=dtype,
        VAE_dtype=dtype,
        text_encoder_filename=te_path,
    )
    log("modelo carregado (" + base_model_type + ")")
    return krea_model, pipe


def profile_model(pipe):
    # offload.profile Colab-T4-optimized (12GB RAM): pinnedMemory=False evita OOM kill
    from mmgp import offload
    offload.profile(
        pipe,
        profile_no=2,
        quantizeTransformer=False,
        convertWeightsFloatTo=torch.float16,
        pinnedMemory=False,
        asyncTransfers=False,
        budgets={"transformer": 10000, "text_encoder": 4000, "vae": 1500, "*": 500},
    )
    offload.shared_state["_attention"] = "sdpa"
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _result_to_image(result):
    # Contrato real do Wan2GP: tensor (3, 1, H, W) -> result[:, 0].permute(1,2,0) -> (H, W, 3)
    # Fallback robusto para outros shapes/escalas.
    try:
        if torch.is_tensor(result):
            result = result.detach().float().cpu().numpy()
    except Exception:
        pass
    a = np.asarray(result)
    if a.ndim == 4:
        # (3,1,H,W) do wan2gp
        if a.shape[1] == 1 and a.shape[0] in (1, 3, 4):
            a = np.transpose(a[:, 0], (1, 2, 0))
        # (B,C,H,W) padrao
        elif a.shape[1] in (1, 3, 4) and a.shape[3] not in (1, 3, 4):
            a = np.transpose(a[0], (1, 2, 0))
        else:
            a = a[0]
    elif a.ndim == 3:
        if a.shape[0] in (1, 3, 4) and a.shape[2] not in (1, 3, 4):
            a = np.transpose(a, (1, 2, 0))
    if a.dtype.kind == "f":
        amin, amax = float(a.min()), float(a.max())
        if amax - amin > 1e-9:
            a = (a - amin) / (amax - amin)
        a = a * 255.0
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    a = np.asarray(a, dtype=np.uint8)
    return Image.fromarray(a).convert("RGB")


def do_generate(krea_model, prompt, negative, steps, width, height, cfg, seed):
    if hasattr(krea_model, "_interrupt"):
        krea_model._interrupt = False
    if hasattr(krea_model, "pipeline") and hasattr(krea_model.pipeline, "_interrupt"):
        krea_model.pipeline._interrupt = False
    guide = float(cfg) if (cfg is not None and float(cfg) > 0) else 0.0
    n_prompt = negative.strip() if (negative and negative.strip()) else None

    def cb(step_idx, latent, is_start, override_num_inference_steps=None, **kw):
        progress(step_idx + 1, int(steps), "Krea2 passo " + str(step_idx + 1) + "/" + str(steps))

    with torch.inference_mode():
        result = krea_model.generate(
            seed=int(seed),
            input_prompt=prompt,
            n_prompt=n_prompt,
            sampling_steps=int(steps),
            width=int(width),
            height=int(height),
            guide_scale=guide,
            batch_size=1,
            callback=cb,
            loras_slists={"phase1": []},
        )
    if result is None:
        raise RuntimeError("geracao retornou None")
    img = _result_to_image(result)
    del result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return base64.b64encode(bio.getvalue()).decode("ascii")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--te", default="")
    ap.add_argument("--model", default="Krea-2-Turbo")
    ap.add_argument("--port", type=int, default=7862)
    ap.add_argument("--wan2gp", default="")
    args = ap.parse_args()

    global WAN2GP_DIR
    WAN2GP_DIR = args.wan2gp or os.environ.get("WAN2GP_DIR") or os.path.abspath("Wan2GP")
    log("WAN2GP_DIR: " + WAN2GP_DIR)
    sys.path.insert(0, WAN2GP_DIR)
    os.chdir(WAN2GP_DIR)

    patch_krea2_main()
    patch_transformers_docstring()
    ensure_requirements()

    if args.te and not os.path.exists(args.te):
        raise RuntimeError("Text encoder nao encontrado: " + args.te)
    if not args.ckpt or not os.path.exists(args.ckpt):
        raise RuntimeError("Checkpoint Krea2 nao encontrado: " + str(args.ckpt))

    krea_model = None
    try:
        krea_model, pipe = load_model(args.ckpt, args.te, args.model)
        profile_model(pipe)
        log("MODELO PRONTO: " + args.model)
    except Exception:
        log("FALHA AO CARREGAR: " + traceback.format_exc())
        raise

    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    _lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def _send(self, code, obj):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            if self.path.startswith("/health"):
                self._send(200, {"status": "ok", "loaded": True, "model": args.model, "backend": "wan2gp"})
            else:
                self._send(404, {"status": "error", "error": "not found"})
        def do_POST(self):
            if self.path.startswith("/unload"):
                self._send(200, {"status": "ok", "unloaded": True})
                threading.Thread(target=os._exit, args=(0,), daemon=True).start()
                return
            if self.path.startswith("/generate"):
                try:
                    n = int(self.headers.get("Content-Length", "0"))
                    data = json.loads(self.rfile.read(n).decode("utf-8"))
                    with _lock:
                        img_b64 = do_generate(
                            krea_model,
                            data.get("prompt", ""),
                            data.get("negative", ""),
                            int(data.get("steps", 8)),
                            int(data.get("width", 1024)),
                            int(data.get("height", 1024)),
                            float(data.get("cfg", 0.0)),
                            int(data.get("seed", 0)),
                        )
                    self._send(200, {"status": "ok", "image": img_b64, "seed": int(data.get("seed", 0))})
                except Exception as e:
                    traceback.print_exc()
                    self._send(500, {"status": "error", "error": str(e)[:500]})
                return
            self._send(404, {"status": "error", "error": "not found"})

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    log("worker online na porta " + str(args.port))
    srv.serve_forever()


if __name__ == "__main__":
    main()
