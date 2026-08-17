# -*- coding: utf-8 -*-
"""
universal_app.py — ADVANCED MULTI-MODEL IMAGE STUDIO (DEFINITIVO v2)
============================================================================
Motor duplo: DIFFUSERS (nativo) + COMFYUI (fallback universal)
Fontes: Civitai (qualquer base model) | Hugging Face | arquivo local

v2 — todas as melhorias P0/P1/P2/P3:
  P0: trigger words automaticas, dropdown de versoes, FLUX sem gate (mirrors),
      dead code removido, limpeza de temp files
  P1: Hires fix 2-passos, CFG rescale (SDXL), negativas por familia, inpainting,
      ControlNet (ComfyUI), prompt matrix, PNGInfo
  P2: auto-budget de VRAM (pynvml), quantizacao opcional (torchao/bnb), speed-up
      (TF32/compile), download paralelo (aria2c), verificacao pos-download,
      cache+backoff da API Civitai, gestao de disco, supervisor
  P3: card do modelo, biblioteca local, comparacao, TextualInversion,
      API HTTP externa (JSON) p/ outras aplicacoes
  Extra: LoRA baixa e ativa automaticamente; sempre versao mais recente
         (recomendada = maior arquivo SafeTensor do tipo Model).

Regra de seguranca: NUNCA use f-string multilinha (causa SyntaxError).
============================================================================
"""
import os
import sys
import json
import gc
import time
import random
import traceback
import re
import shutil
import subprocess
import tempfile
import threading
import struct
import base64
import io
import hashlib
import urllib.parse
from pathlib import Path
# numpy e usado em _parse_image_input (np.ndarray de componentes gr.Image do Gradio).
# Importe explicito: nao depende de re-export de torch/diffusers.
try:
    import numpy as np
except Exception:
    np = None  # ndarray nao sera reconhecido; PIL/str continuam funcionando

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ---- Bloqueia TensorFlow para poupar RAM ---------------------------------
import importlib

class _TFBlocker:
    def find_module(self, name, path=None):
        if name == "tensorflow" or name.startswith("tensorflow."):
            return self
    def load_module(self, name):
        raise ImportError("TensorFlow blocked to save RAM")

if not any(isinstance(m, _TFBlocker) for m in sys.meta_path):
    sys.meta_path.insert(0, _TFBlocker())

# ---- Caminhos --------------------------------------------------------------
BASE_DIR = "/content"
APP_DIR = os.path.join(BASE_DIR, "studio")
MODELS_DIR = os.path.join(APP_DIR, "models")
CIVITAI_DIR = os.path.join(MODELS_DIR, "civitai")
LORA_DIR = os.path.join(MODELS_DIR, "loras")
VAE_DIR = os.path.join(MODELS_DIR, "vaes")
TI_DIR = os.path.join(MODELS_DIR, "embeddings")
HF_CACHE = os.path.join(APP_DIR, "hf")
API_CACHE = os.path.join(APP_DIR, "api_cache")
OUTPUTS_DIR = os.path.join(APP_DIR, "outputs")
COMFY_DIR = "/content/ComfyUI"
WAN2GP_DIR = "/content/Wan2GP"
API_PORT = int(os.environ.get("STUDIO_API_PORT", "7861"))

for d in [APP_DIR, MODELS_DIR, CIVITAI_DIR, LORA_DIR, VAE_DIR, TI_DIR, HF_CACHE, API_CACHE, OUTPUTS_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.5"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE, "hub")

# ---- API keys: auto-registro como variaveis de ambiente (funciona via notebook OU CLI) ----
# Garante que CIVITAI_TOKEN / HF_TOKEN estejam SEMPRE definidos no ambiente,
# independentemente de como o app foi lancado (celula do notebook ou exec remoto no Colab CLI).
DEFAULT_CIVITAI_TOKEN = os.environ.get("CIVITAI_TOKEN", "")
DEFAULT_HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.environ.setdefault("CIVITAI_TOKEN", DEFAULT_CIVITAI_TOKEN)
os.environ.setdefault("HF_TOKEN", DEFAULT_HF_TOKEN)
# Persiste em tokens.json (idempotente) para o caminho de arquivo do app.
# Respeita valores ja existentes: so preenche se ausente.
try:
    import json as _json
    _tj = os.path.join(APP_DIR, "tokens.json")
    _tok = {}
    if os.path.exists(_tj):
        try:
            _tok = _json.load(open(_tj, encoding="utf-8"))
        except Exception:
            _tok = {}
    _tok.setdefault("civitai", DEFAULT_CIVITAI_TOKEN)
    _tok.setdefault("hf", DEFAULT_HF_TOKEN)
    with open(_tj, "w", encoding="utf-8") as _f:
        _json.dump(_tok, _f)
except Exception:
    pass

import torch
from PIL import Image
import requests
import gradio as gr

# Speed-up generico (P2-15)
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

# Locks de concorrencia (UI + API compartilham o estado)
GEN_LOCK = threading.Lock()
LOAD_LOCK = threading.Lock()
DOWNLOAD_LOCK = threading.RLock()  # serializa download+rename do mesmo arquivo (anti-race .part); RLock p/ aux aninhado

# ============================================================================
# 1. REGISTRY DE FAMILIAS (definitivo — cobre TODOS os base models do Civitai)
# ============================================================================
BASE_MODEL_MAP = {
    "SD 1.4": "sd15", "SD 1.5": "sd15", "SD 1.5 Hyper": "sd15_hyper", "SD 1.5 LCM": "sd15_lcm",
    "SD 2.0": "sd2", "SD 2.1": "sd2", "SD 2.1 768": "sd2_768",
    "SDXL 0.9": "sdxl", "SDXL 1.0": "sdxl",
    "SDXL 1.0 LCM": "sdxl_lcm", "SDXL Hyper": "sdxl_hyper", "SDXL Lightning": "sdxl_lightning",
    "Pony": "pony", "Illustrious": "illustrious", "NoobAI": "noobai", "Animagine XL": "animagine",
    "FLUX.1 D": "flux_dev", "FLUX.1 S": "flux_schnell", "FLUX.1 K": "flux_dev", "FLUX.1 Krea": "flux_dev",
    "FLUX.2 Klein 4B-base": "flux2_klein", "Krea 2": "krea2",
    "SD 3": "sd3", "SD 3.5": "sd35", "SD 3.5 Large": "sd35",
    "Anima": "anima", "Chroma": "chroma", "AuraFlow": "auraflow",
    "Hunyuan DiT": "hunyuan", "HiDream": "hidream", "Lumina": "lumina",
    "ERNIE": "ernie", "Grok": "grok",
    "ZImageBase": "zimage", "ZImageTurbo": "zimage_turbo",
    "Qwen": "qwen",
    "Wan Video 1.3B t2v": "wan_video", "Wan Video 14B t2v": "wan_video",
    "Upscaler": "upscaler", "Other": "other",
}


def _family_from_base(base_model):
    """Civitai usa labels INCONSISTENTES de baseModel ('Flux.1 D' vs mapa 'FLUX.1 D';
    em Brazil o case muda entre versions). Lookup case-insensitive + aliases comuns,
    evitando que um checkpoint FLUX caia no branch generico (CheckpointLoaderSimple) e
    falhe com 'clip input is invalid: None' (flux nao tem CLIP dentro do safetensors)."""
    if not base_model:
        return "other"
    norm = str(base_model).strip().lower()
    aliases = {
        "flux": "flux_dev", "flux.1": "flux_dev", "flux.1 d": "flux_dev",
        "flux.1 dev": "flux_dev", "flux.1 s": "flux_schnell",
        "flux.1 schnell": "flux_schnell", "flux.1 krea": "flux_dev",
        "flux.1 krea [fp8]": "flux_dev", "flux.2 klein": "flux2_klein",
        "sdxl": "sdxl", "sdxl 1.0": "sdxl", "sd xl": "sdxl", "xl": "sdxl",
        "sd 1.5": "sd15", "sd1.5": "sd15", "stable diffusion 1.5": "sd15",
        "pony": "pony", "illustrious": "illustrious", "noobai": "noobai",
        "animagine xl": "animagine", "qwen image": "qwen",
        "wan video 1.3b t2v": "wan_video", "wan video 14b t2v": "wan_video",
        "other": "other", "unknown": "other", "": "other", "none": "other",
    }
    if norm in aliases:
        return aliases[norm]
    for k, v in BASE_MODEL_MAP.items():
        if str(k).strip().lower() == norm:
            return v
    return "other"


def _sniff_family_from_file(path, family="other"):
    """REDE DE SEGURANCA: se o metadata do Civitai for 'Other'/desconhecido, detecta a
    familia pelo CONTEUDO do safetensors (header JSON completo) — mesma ideia do
    detect_te_model do Anima. Cobre os DOIS formatos de FLUX:
      ComfyUI  : model.diffusion_model.double_blocks.* + single_blocks.* (Kestral/FP8)
      diffusers: transformer_blocks.* + guidance_embed/additive_encodings/double_blocks
    Alem de SD3 (joint_transformer_blocks), SD1.5 (input_blocks) e SDXL (conditioner)."""
    if family != "other" or not path or not os.path.exists(path):
        return family
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n <= 0 or n > 256 * 1024 * 1024:
                return family
            head = json.loads(f.read(n).decode("utf-8", "replace"))
        # header COMPLETO (nao cortar em 120 keys — a ordem das chaves varia por criador;
        # o Kestral comeca por time_in/vector_in e double_blocks pode vir depois)
        j = "\n".join(head.keys())
        # FLUX (ComfyUI unet): double_blocks + single_blocks
        if ("model.diffusion_model.double_blocks." in j and
                "model.diffusion_model.single_blocks." in j):
            return "flux_dev"
        # FLUX (diffusers transformer) ou variantes quantizadas (keys podem vir sem prefixo model.diffusion_model)
        if "double_blocks." in j and "single_blocks." in j:
            return "flux_dev"
        if ("transformer_blocks." in j and
                ("guidance_embed" in j or "additive_encodings" in j or "double_blocks" in j)):
            return "flux_dev"
        if "model.diffusion_model.joint_transformer_blocks." in j:
            return "sd3"
        if "model.diffusion_model.input_blocks." in j:
            return "sd15"
        if "conditioner.embedders." in j:
            return "sdxl"
    except Exception:
        pass
    return family


FAMILY_PRESETS = {
    "sd15": {
        "label": "SD 1.5", "diffusers_cls": "StableDiffusionPipeline",
        "single_file": True, "inject": ["text_encoder", "tokenizer", "vae"],
        "vae": "stabilityai/sd-vae-ft-mse", "te_repo": "runwayml/stable-diffusion-v1-5",
        "steps": {"min": 10, "max": 50, "default": 25}, "cfg": 7.0,
        "scheduler": "DPM++ 2M Karras", "sampler": "dpmpp_2m",
        "resolutions": ["512px (Fast)", "640px (Balanced)", "768px (High)"],
        "default_res": "512px (Fast)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait", "4:3 Standard", "3:4 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "",
        "neg_prefix": "",
        "comfy_template": "checkpoint", "controlnet": "control_v11p_sd15_canny.pth",
        "cfg_rescale": 0.0,
        "notes": "Classe universal do Civitai. Injeta TE/VAE se faltar.",
    },
    "sd15_hyper": {
        "label": "SD 1.5 Hyper", "diffusers_cls": "StableDiffusionPipeline",
        "single_file": True, "inject": ["text_encoder", "tokenizer", "vae"],
        "vae": "stabilityai/sd-vae-ft-mse", "te_repo": "runwayml/stable-diffusion-v1-5",
        "steps": {"min": 1, "max": 8, "default": 4}, "cfg": 1.0,
        "scheduler": "Euler", "sampler": "euler",
        "resolutions": ["512px (Fast)", "640px (Balanced)", "768px (High)"],
        "default_res": "512px (Fast)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "checkpoint", "controlnet": "control_v11p_sd15_canny.pth",
        "cfg_rescale": 0.0, "notes": "Hyper destilado: CFG ~1 e poucos passos.",
    },
    "sd15_lcm": {
        "label": "SD 1.5 LCM", "diffusers_cls": "StableDiffusionPipeline",
        "single_file": True, "inject": ["text_encoder", "tokenizer", "vae"],
        "vae": "stabilityai/sd-vae-ft-mse", "te_repo": "runwayml/stable-diffusion-v1-5",
        "steps": {"min": 1, "max": 8, "default": 4}, "cfg": 1.0,
        "scheduler": "LCM", "sampler": "lcm",
        "resolutions": ["512px (Fast)", "640px (Balanced)", "768px (High)"],
        "default_res": "512px (Fast)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "checkpoint", "controlnet": "control_v11p_sd15_canny.pth",
        "cfg_rescale": 0.0, "notes": "LCM: CFG ~1 e 4 passos.",
    },
    "sd2": {
        "label": "SD 2.1", "diffusers_cls": "StableDiffusionPipeline",
        "single_file": True, "inject": ["text_encoder", "tokenizer", "vae"],
        "vae": "stabilityai/sd-vae-ft-mse", "te_repo": "stabilityai/stable-diffusion-2-1",
        "steps": {"min": 10, "max": 50, "default": 25}, "cfg": 8.0,
        "scheduler": "DPM++ 2M Karras", "sampler": "dpmpp_2m",
        "resolutions": ["512px (Fast)", "640px (Balanced)", "768px (High)"],
        "default_res": "512px (Fast)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "checkpoint", "controlnet": "control_v11p_sd15_canny.pth",
        "cfg_rescale": 0.0, "notes": "SD 2.1 usa tokenizador OpenCLIP.",
    },
    "sd2_768": {
        "label": "SD 2.1 768", "diffusers_cls": "StableDiffusionPipeline",
        "single_file": True, "inject": ["text_encoder", "tokenizer", "vae"],
        "vae": "stabilityai/sd-vae-ft-mse", "te_repo": "stabilityai/stable-diffusion-2-1",
        "steps": {"min": 10, "max": 50, "default": 25}, "cfg": 8.0,
        "scheduler": "DPM++ 2M Karras", "sampler": "dpmpp_2m",
        "resolutions": ["640px (Standard)", "768px (High)"],
        "default_res": "768px (High)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "checkpoint", "controlnet": "control_v11p_sd15_canny.pth",
        "cfg_rescale": 0.0, "notes": "SD 2.1 768 nativo.",
    },
    "sdxl": {
        "label": "SDXL 1.0", "diffusers_cls": "StableDiffusionXLPipeline",
        "single_file": True, "inject": ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae"],
        "vae": "madebyollin/sdxl-vae-fp16-fix", "te_repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "steps": {"min": 10, "max": 60, "default": 28}, "cfg": 6.5,
        "scheduler": "DPM++ 2M Karras", "sampler": "dpmpp_2m",
        "resolutions": ["768px (Balanced)", "1024px (Standard)", "1152px (High)"],
        "default_res": "1024px (Standard)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait", "4:3 Standard", "3:4 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "checkpoint", "controlnet": "diffusion_pytorch_model.safetensors",
        "cfg_rescale": 0.7,
        "notes": "Padrao da industria. TE duplo + VAE fp16 + CFG rescale.",
    },
    "sdxl_lightning": {
        "label": "SDXL Lightning", "diffusers_cls": "StableDiffusionXLPipeline",
        "single_file": True, "inject": ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae"],
        "vae": "madebyollin/sdxl-vae-fp16-fix", "te_repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "steps": {"min": 1, "max": 8, "default": 4}, "cfg": 1.0,
        "scheduler": "Euler", "sampler": "euler",
        "resolutions": ["768px (Balanced)", "1024px (Standard)"],
        "default_res": "768px (Balanced)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "checkpoint", "controlnet": "diffusion_pytorch_model.safetensors",
        "cfg_rescale": 0.0, "notes": "Lightning: CFG ~1, 4 passos.",
    },
    "sdxl_lcm": {
        "label": "SDXL LCM", "diffusers_cls": "StableDiffusionXLPipeline",
        "single_file": True, "inject": ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae"],
        "vae": "madebyollin/sdxl-vae-fp16-fix", "te_repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "steps": {"min": 1, "max": 8, "default": 4}, "cfg": 1.0,
        "scheduler": "LCM", "sampler": "lcm",
        "resolutions": ["768px (Balanced)", "1024px (Standard)"],
        "default_res": "768px (Balanced)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "checkpoint", "controlnet": "diffusion_pytorch_model.safetensors",
        "cfg_rescale": 0.0, "notes": "LCM: CFG ~1, 4 passos.",
    },
    "sdxl_hyper": {
        "label": "SDXL Hyper", "diffusers_cls": "StableDiffusionXLPipeline",
        "single_file": True, "inject": ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae"],
        "vae": "madebyollin/sdxl-vae-fp16-fix", "te_repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "steps": {"min": 1, "max": 8, "default": 4}, "cfg": 1.0,
        "scheduler": "DPM++ 2M Karras", "sampler": "dpmpp_2m",
        "resolutions": ["768px (Balanced)", "1024px (Standard)"],
        "default_res": "1024px (Standard)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "checkpoint", "controlnet": "diffusion_pytorch_model.safetensors",
        "cfg_rescale": 0.0, "notes": "Hyper SDXL: CFG ~1, 4 passos.",
    },
    "pony": {
        "label": "Pony Diffusion V6", "diffusers_cls": "StableDiffusionXLPipeline",
        "single_file": True, "inject": ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae"],
        "vae": "madebyollin/sdxl-vae-fp16-fix", "te_repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "steps": {"min": 10, "max": 50, "default": 28}, "cfg": 6.5,
        "scheduler": "DPM++ 2M Karras", "sampler": "dpmpp_2m",
        "resolutions": ["768px (Balanced)", "1024px (Standard)"],
        "default_res": "1024px (Standard)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait", "4:3 Standard", "3:4 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, ",
        "prompt_suffix": "",
        "neg_prefix": "score_1, score_2, score_3, score_4, worst quality, low quality, blurry, bad anatomy, deformed, text, watermark",
        "comfy_template": "checkpoint", "controlnet": "diffusion_pytorch_model.safetensors",
        "cfg_rescale": 0.7,
        "notes": "REQUER tags score_ (auto). Negativa recomendada aplicada automaticamente.",
    },
    "illustrious": {
        "label": "Illustrious XL", "diffusers_cls": "StableDiffusionXLPipeline",
        "single_file": True, "inject": ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae"],
        "vae": "madebyollin/sdxl-vae-fp16-fix", "te_repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "steps": {"min": 10, "max": 50, "default": 26}, "cfg": 5.0,
        "scheduler": "Euler a", "sampler": "euler_ancestral",
        "resolutions": ["768px (Balanced)", "1024px (Standard)", "1152px (High)"],
        "default_res": "1024px (Standard)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait", "4:3 Standard", "3:4 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "masterpiece, best quality, amazing quality, very aesthetic, absurdres, ",
        "prompt_suffix": "",
        "neg_prefix": "worst quality, low quality, bad anatomy, bad hands, missing fingers, extra digits, blurry, jpeg artifacts, signature, watermark, username, text",
        "comfy_template": "checkpoint", "controlnet": "diffusion_pytorch_model.safetensors",
        "cfg_rescale": 0.5,
        "notes": "Tags de qualidade (auto) + negativa recomendada.",
    },
    "noobai": {
        "label": "NoobAI-XL", "diffusers_cls": "StableDiffusionXLPipeline",
        "single_file": True, "inject": ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae"],
        "vae": "madebyollin/sdxl-vae-fp16-fix", "te_repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "steps": {"min": 10, "max": 50, "default": 26}, "cfg": 5.0,
        "scheduler": "Euler a", "sampler": "euler_ancestral",
        "resolutions": ["768px (Balanced)", "1024px (Standard)", "1152px (High)"],
        "default_res": "1024px (Standard)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait", "4:3 Standard", "3:4 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "masterpiece, best quality, amazing quality, very aesthetic, absurdres, rating: general, ",
        "prompt_suffix": "",
        "neg_prefix": "worst quality, low quality, bad anatomy, bad hands, missing fingers, extra digits, blurry, jpeg artifacts, signature, watermark, username, text",
        "comfy_template": "checkpoint", "controlnet": "diffusion_pytorch_model.safetensors",
        "cfg_rescale": 0.5,
        "notes": "Tags + rating (auto) + negativa recomendada.",
    },
    "animagine": {
        "label": "Animagine XL", "diffusers_cls": "StableDiffusionXLPipeline",
        "single_file": True, "inject": ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae"],
        "vae": "madebyollin/sdxl-vae-fp16-fix", "te_repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "steps": {"min": 10, "max": 50, "default": 28}, "cfg": 6.0,
        "scheduler": "DPM++ 2M Karras", "sampler": "dpmpp_2m",
        "resolutions": ["768px (Balanced)", "1024px (Standard)"],
        "default_res": "1024px (Standard)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "masterpiece, best quality, ",
        "prompt_suffix": "",
        "neg_prefix": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name",
        "comfy_template": "checkpoint", "controlnet": "diffusion_pytorch_model.safetensors",
        "cfg_rescale": 0.5,
        "notes": "Animagine: tags + negativa recomendada.",
    },
    "flux_dev": {
        "label": "FLUX.1 (D/K)", "diffusers_cls": "FluxPipeline",
        "single_file": True, "inject": ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae"],
        "vae": None, "te_repo": None,
        "steps": {"min": 10, "max": 50, "default": 28}, "cfg": 3.5,
        "scheduler": "Euler", "sampler": "euler",
        "resolutions": ["512px (Low VRAM)", "768px (Balanced)", "1024px (Standard)"],
        "default_res": "768px (Balanced)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "quantize": "torchao_int8", "comfy_template": "flux", "controlnet": None,
        "cfg_rescale": 0.0,
        "notes": "FLUX usa guidance_embedding; INT8 via torchao p/ rodar na T4. Componentes baixados de mirrors publicos.",
    },
    "flux_schnell": {
        "label": "FLUX.1 schnell", "diffusers_cls": "FluxPipeline",
        "single_file": True, "inject": ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae"],
        "vae": None, "te_repo": None,
        "steps": {"min": 1, "max": 8, "default": 4}, "cfg": 0.0,
        "scheduler": "Euler", "sampler": "euler",
        "resolutions": ["512px (Low VRAM)", "768px (Balanced)", "1024px (Standard)"],
        "default_res": "768px (Balanced)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "quantize": "torchao_int8", "comfy_template": "flux", "controlnet": None,
        "cfg_rescale": 0.0, "notes": "FLUX schnell: CFG 0, 4 passos.",
    },
    "krea2": {
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
    "flux2_klein": {
        "label": "FLUX.2 Klein 4B", "diffusers_cls": "FluxPipeline2",
        "single_file": True, "inject": ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae"],
        "vae": None, "te_repo": None,
        "steps": {"min": 10, "max": 40, "default": 24}, "cfg": 3.5,
        "scheduler": "Euler", "sampler": "euler",
        "resolutions": ["512px (Low VRAM)", "768px (Balanced)", "1024px (Standard)"],
        "default_res": "768px (Balanced)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "quantize": "torchao_int8", "comfy_template": "flux", "controlnet": None,
        "cfg_rescale": 0.0, "notes": "FLUX.2 Klein 4B.",
    },
    "sd3": {
        "label": "SD 3", "diffusers_cls": "StableDiffusion3Pipeline",
        "single_file": True, "inject": ["text_encoder", "text_encoder_2", "text_encoder_3", "tokenizer", "tokenizer_2", "tokenizer_3", "vae"],
        "vae": "stabilityai/sd-vae-ft-mse", "te_repo": "stabilityai/stable-diffusion-3-medium",
        "steps": {"min": 10, "max": 50, "default": 28}, "cfg": 7.0,
        "scheduler": "DPM++ 2M Karras", "sampler": "dpmpp_2m",
        "resolutions": ["512px (Low VRAM)", "768px (Balanced)", "1024px (Standard)"],
        "default_res": "768px (Balanced)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "sd3", "controlnet": None, "cfg_rescale": 0.0,
        "notes": "SD 3 requer T5 + 2 CLIP. Pesado na T4.",
    },
    "sd35": {
        "label": "SD 3.5", "diffusers_cls": "StableDiffusion3Pipeline",
        "single_file": True, "inject": ["text_encoder", "text_encoder_2", "text_encoder_3", "tokenizer", "tokenizer_2", "tokenizer_3", "vae"],
        "vae": "stabilityai/sd-vae-ft-mse", "te_repo": "stabilityai/stable-diffusion-3.5-medium",
        "steps": {"min": 10, "max": 50, "default": 30}, "cfg": 4.5,
        "scheduler": "DPM++ 2M Karras", "sampler": "dpmpp_2m",
        "resolutions": ["512px (Low VRAM)", "768px (Balanced)", "1024px (Standard)"],
        "default_res": "768px (Balanced)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "sd3", "controlnet": None, "cfg_rescale": 0.0,
        "notes": "SD 3.5: CFG 4.5, preferir ComfyUI na T4.",
    },
    "zimage": {
        "label": "Z-Image Base", "diffusers_cls": "ZImagePipeline",
        "single_file": True, "inject": ["transformer"],
        "vae": None, "te_repo": None,
        "steps": {"min": 1, "max": 20, "default": 8}, "cfg": 0.0,
        "scheduler": "Euler", "sampler": "euler",
        "resolutions": ["512px (Low VRAM)", "768px (Balanced)", "1024px (Standard)"],
        "default_res": "768px (Balanced)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "checkpoint", "controlnet": None, "cfg_rescale": 0.0,
        "notes": "Z-Image: transformer single-file + pipeline base.",
    },
    "zimage_turbo": {
        "label": "Z-Image Turbo", "diffusers_cls": "ZImagePipeline",
        "single_file": True, "inject": ["transformer"],
        "vae": None, "te_repo": None,
        "steps": {"min": 1, "max": 8, "default": 4}, "cfg": 0.0,
        "scheduler": "Euler", "sampler": "euler",
        "resolutions": ["512px (Low VRAM)", "768px (Balanced)", "1024px (Standard)"],
        "default_res": "768px (Balanced)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "checkpoint", "controlnet": None, "cfg_rescale": 0.0,
        "notes": "Z-Image Turbo: 4 passos, CFG 0.",
    },
    "qwen": {
        "label": "Qwen-Image", "diffusers_cls": "QwenImagePipeline",
        "single_file": True, "inject": ["text_encoder", "tokenizer", "vae"],
        "vae": "Qwen/Qwen-Image", "te_repo": "Qwen/Qwen-Image",
        "steps": {"min": 10, "max": 50, "default": 28}, "cfg": 4.0,
        "scheduler": "Euler", "sampler": "euler",
        "resolutions": ["512px (Low VRAM)", "768px (Balanced)", "1024px (Standard)"],
        "default_res": "768px (Balanced)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "checkpoint", "controlnet": None, "cfg_rescale": 0.0,
        "notes": "Qwen-Image (novo no Civitai).",
    },
    "anima": {
        "label": "Anima (CircleStone)", "diffusers_cls": "AnimaPipeline",
        "single_file": True, "inject": [],
        "vae": None, "te_repo": None,
        "steps": {"min": 10, "max": 60, "default": 35}, "cfg": 6.0,
        "scheduler": "simple", "sampler": "er_sde",
        "resolutions": ["768px (Fast)", "1024px (Standard)", "1344x1024 (Rec.)"],
        "default_res": "1024px (Standard)",
        "aspects": ["1:1 Square", "4:3 Classic", "16:9 Landscape", "9:16 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "",
        "neg_prefix": "score_1, score_2, score_3, sketch, limited palette, watercolor, photo, 3D, flat color, multiple arms, bad anatomy",
        "comfy_template": "anima", "controlnet": None, "cfg_rescale": 0.0,
        "notes": "Anima (DiT 2B): roda via ComfyUI (UNET + Qwen3-0.6B TE + VAE Wan21). TE e VAE baixados automaticamente do Civitai no 1o load. Recomendado pelo criador: CFG 6.0, 35 steps, ~1024x1344.",
    },
    "chroma": {
        "label": "Chroma", "diffusers_cls": "ChromaPipeline",
        "single_file": True, "inject": ["transformer"],
        "vae": None, "te_repo": None,
        "steps": {"min": 10, "max": 40, "default": 24}, "cfg": 3.5,
        "scheduler": "Euler", "sampler": "euler",
        "resolutions": ["512px (Low VRAM)", "768px (Balanced)"],
        "default_res": "768px (Balanced)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "flux", "controlnet": None, "cfg_rescale": 0.0,
        "notes": "Chroma: FLUX-like 8.9B; preferir ComfyUI na T4.",
    },
    "auraflow": {
        "label": "AuraFlow", "diffusers_cls": "AuraFlowPipeline",
        "single_file": False, "inject": [],
        "vae": None, "te_repo": None,
        "steps": {"min": 10, "max": 50, "default": 28}, "cfg": 7.0,
        "scheduler": "DPM++ 2M Karras", "sampler": "dpmpp_2m",
        "resolutions": ["512px (Low VRAM)", "768px (Balanced)"],
        "default_res": "768px (Balanced)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "checkpoint", "controlnet": None, "cfg_rescale": 0.0,
        "notes": "AuraFlow 6.8B: pesado p/ T4; ComfyUI recomendado.",
    },
    "hunyuan": {
        "label": "Hunyuan DiT", "diffusers_cls": "HunyuanDiTPipeline",
        "single_file": False, "inject": [],
        "vae": None, "te_repo": None,
        "steps": {"min": 10, "max": 50, "default": 30}, "cfg": 5.0,
        "scheduler": "DPM++ 2M Karras", "sampler": "dpmpp_2m",
        "resolutions": ["512px (Low VRAM)", "768px (Balanced)"],
        "default_res": "768px (Balanced)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "checkpoint", "controlnet": None, "cfg_rescale": 0.0,
        "notes": "Hunyuan-DiT: sem single-file no diffusers; use ComfyUI.",
    },
    "hidream": {
        "label": "HiDream", "diffusers_cls": "HiDreamImagePipeline",
        "single_file": False, "inject": [],
        "vae": None, "te_repo": None,
        "steps": {"min": 10, "max": 50, "default": 28}, "cfg": 3.5,
        "scheduler": "DPM++ 2M Karras", "sampler": "dpmpp_2m",
        "resolutions": ["512px (Low VRAM)", "768px (Balanced)"],
        "default_res": "768px (Balanced)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "flux", "controlnet": None, "cfg_rescale": 0.0,
        "notes": "HiDream 17B: acima da T4; so tenta em ComfyUI com quantizacao.",
    },
    "lumina": {
        "label": "Lumina", "diffusers_cls": "Lumina2Pipeline",
        "single_file": False, "inject": [],
        "vae": None, "te_repo": None,
        "steps": {"min": 10, "max": 50, "default": 28}, "cfg": 1.0,
        "scheduler": "Euler", "sampler": "euler",
        "resolutions": ["512px (Low VRAM)", "768px (Balanced)"],
        "default_res": "768px (Balanced)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "checkpoint", "controlnet": None, "cfg_rescale": 0.0,
        "notes": "Lumina: sem single-file; ComfyUI recomendado.",
    },
    "ernie": {
        "label": "Baidu ERNIE-Image", "diffusers_cls": None, "single_file": False,
        "inject": [], "vae": None, "te_repo": None,
        "steps": {"min": 10, "max": 50, "default": 28}, "cfg": 3.5,
        "scheduler": "Euler", "sampler": "euler",
        "resolutions": ["512px (Low VRAM)", "768px (Balanced)"],
        "default_res": "768px (Balanced)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "checkpoint", "controlnet": None, "cfg_rescale": 0.0,
        "notes": "ERNIE-Image: sem loader estavel; tenta ComfyUI.",
    },
    "grok": {
        "label": "Grok (xAI)", "diffusers_cls": None, "single_file": False,
        "inject": [], "vae": None, "te_repo": None,
        "steps": {"min": 1, "max": 4, "default": 4}, "cfg": 0.0,
        "scheduler": "Euler", "sampler": "euler",
        "resolutions": ["768px (Balanced)"],
        "default_res": "768px (Balanced)",
        "aspects": ["1:1 Square"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "none", "controlnet": None, "cfg_rescale": 0.0,
        "notes": "Grok e HOSPEDADO (xAI), sem pesos abertos — impossivel rodar localmente.",
    },
    "wan_video": {
        "label": "Wan Video", "diffusers_cls": None, "single_file": False,
        "inject": [], "vae": None, "te_repo": None,
        "steps": {"min": 10, "max": 50, "default": 30}, "cfg": 5.0,
        "scheduler": "Euler", "sampler": "euler",
        "resolutions": ["832x480 (Video)"],
        "default_res": "832x480 (Video)",
        "aspects": ["16:9 Landscape"],
        "default_aspect": "16:9 Landscape",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "none", "controlnet": None, "cfg_rescale": 0.0,
        "notes": "Wan Video gera VIDEO — fora do escopo de imagens deste app.",
    },
    "upscaler": {
        "label": "Upscaler", "diffusers_cls": None, "single_file": False,
        "inject": [], "vae": None, "te_repo": None,
        "steps": {"min": 1, "max": 20, "default": 10}, "cfg": 0.0,
        "scheduler": "Euler", "sampler": "euler",
        "resolutions": ["2x", "4x"],
        "default_res": "4x",
        "aspects": ["1:1 Square"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "upscale", "controlnet": None, "cfg_rescale": 0.0,
        "notes": "Upscaler (RealESRGAN): aplica em imagem enviada.",
    },
    "other": {
        "label": "Other / Desconhecido", "diffusers_cls": None, "single_file": True,
        "inject": [], "vae": None, "te_repo": None,
        "steps": {"min": 10, "max": 50, "default": 25}, "cfg": 6.5,
        "scheduler": "DPM++ 2M Karras", "sampler": "dpmpp_2m",
        "resolutions": ["512px (Fast)", "768px (Balanced)", "1024px (Standard)"],
        "default_res": "768px (Balanced)",
        "aspects": ["1:1 Square", "16:9 Landscape", "9:16 Portrait"],
        "default_aspect": "1:1 Square",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "checkpoint", "controlnet": None, "cfg_rescale": 0.0,
        "notes": "Familia desconhecida: sniffing + ComfyUI universal.",
    },
}

SCHEDULERS = {
    "DPM++ 2M Karras": ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True}),
    "DPM++ 2M SDE Karras": ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True, "algorithm_type": "sde-dpmsolver++"}),
    "Euler": ("EulerDiscreteScheduler", {}),
    "Euler a": ("EulerAncestralDiscreteScheduler", {}),
    "DDIM": ("DDIMScheduler", {}),
    "UniPC": ("UniPCMultistepScheduler", {}),
    "LCM": ("LCMScheduler", {}),
    "DEIS": ("DEISMultistepScheduler", {}),
}

STYLES = ["None", "Cinematic", "Photographic", "Anime", "Cyberpunk", "Fantasy", "Pixel Art", "Oil Painting"]

# ============================================================================
# 2. ESTADO GLOBAL
# ============================================================================
# Versao do app — visivel em /api/health e no startup. Incremente a cada correcao
# para detectar se a VM ainda roda codigo antigo (Celula 4 troca a URL do Gradio).
APP_VER = "v2.5.20260817"
STATE = {
    "config": None, "backend": None, "pipe": None,
    "krea_model": None, "krea_worker": None, "loaded": False,
    "family": None, "lora_stack": [], "lora_scale": 1.0,
    "model_name": None, "model_path": None,
    "trained_words": [],
    "civitai_model": None, "civitai_versions": [],
    "hf_token": None, "civitai_token": None,
    "quant_method": "torchao_int8", "use_compile": False,
    "api_key": os.environ.get("STUDIO_API_KEY", ""),
    "last_loaded_url": None,
}

def get_torch_dtype(name):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return torch.float16

def _load_env_tokens():
    """Tokens de CIVITAI_TOKEN/HF_TOKEN (env, herdado do kernel) ou /content/studio/tokens.json."""
    for var, key in (("CIVITAI_TOKEN", "civitai_token"), ("HF_TOKEN", "hf_token")):
        val = os.environ.get(var)
        if val and not STATE.get(key):
            STATE[key] = str(val).strip()
    tj = os.path.join(APP_DIR, "tokens.json")
    if os.path.exists(tj):
        try:
            with open(tj, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("civitai") and not STATE.get("civitai_token"):
                STATE["civitai_token"] = str(data["civitai"]).strip()
            if data.get("hf") and not STATE.get("hf_token"):
                STATE["hf_token"] = str(data["hf"]).strip()
        except Exception:
            pass
    if STATE.get("civitai_token"):
        print("  Civitai token: configurado (env/tokens.json)")
    if STATE.get("hf_token"):
        print("  HF token: configurado (env/tokens.json)")

_load_env_tokens()

# ============================================================================
# 3. CLIENTE CIVITAI (API v1) — cache em disco + backoff (P2-18)
# ============================================================================
CIVITAI_API = "https://civitai.com/api/v1"
API_CACHE_TTL = 3600

def civitai_headers(token):
    h = {"User-Agent": "Mozilla/5.0 (Colab) advanced-multi-model-studio/2.0"}
    t = (token or STATE.get("civitai_token") or "").strip()
    if t:
        h["Authorization"] = "Bearer " + t
    return h

def parse_civitai_id(url_or_id):
    url_or_id = (url_or_id or "").strip()
    if not url_or_id:
        return None
    if url_or_id.isdigit():
        return url_or_id
    m = re.search(r"civitai\.com/models/(\d+)", url_or_id)
    if m:
        return m.group(1)
    m = re.search(r"api/v1/models/(\d+)", url_or_id)
    if m:
        return m.group(1)
    return None

def civitai_get_model(model_id, token, force=False):
    """Consulta com cache em disco (TTL 1h) e backoff exponencial em 429."""
    cache_file = os.path.join(API_CACHE, str(model_id) + ".json")
    token_used = (token or STATE.get("civitai_token") or "").strip()
    if not force and os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < API_CACHE_TTL:
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    url = CIVITAI_API + "/models/" + str(model_id)
    last_err = None
    for attempt in range(6):
        try:
            r = requests.get(url, headers=civitai_headers(token_used), timeout=60)
            if r.status_code == 200:
                data = r.json()
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                return data
            if r.status_code == 429:
                retry = int(r.headers.get("Retry-After", "0") or "0")
                wait = retry or min(2 ** attempt, 30)
                print("Civitai 429 (rate limit). Aguardando " + str(wait) + "s...")
                time.sleep(wait)
                continue
            last_err = "HTTP " + str(r.status_code) + ": " + r.text[:200]
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError("Falha ao consultar Civitai: " + str(last_err))

def civitai_sorted_versions(model_data):
    versions = model_data.get("modelVersions", [])
    versions = [v for v in versions if v.get("files")]
    versions.sort(key=lambda v: v.get("publishedAt") or "", reverse=True)
    return versions

def fmt_bytes(n):
    if n is None:
        return "?"
    if n > 1024 ** 3:
        return "{:.2f} GB".format(n / 1024 ** 3)
    if n > 1024 ** 2:
        return "{:.1f} MB".format(n / 1024 ** 2)
    return "{:.0f} KB".format(n / 1024)

MODEL_FILE_TYPES = ("Model", "Diffusion Model", "Pruned Model", "Checkpoint")


def pick_file_from_version(version, wanted_type=None):
    """Escolhe o arquivo RECOMENDADO: tipo certo + SafeTensor + maior tamanho.
    Suporta fallback gracioso caso o Civitai tenha rotulado um LoRA/VAE/TI como 'Model'."""
    files = version.get("files", [])
    if not files:
        raise RuntimeError("Nenhum arquivo nessa versao.")
    cands = []
    if wanted_type and wanted_type != "Model":
        cands = [f for f in files if str(f.get("type") or "").lower() == str(wanted_type).lower()]
        if not cands:
            # Fallback: se o Civitai rotulou os arquivos da página de LoRA/VAE como 'Model' ou 'Diffusion Model'
            cands = list(files)
    else:
        cands = [f for f in files if (f.get("type") or "") in MODEL_FILE_TYPES]
        if not cands:
            cands = [f for f in files
                     if (f.get("type") or "") in MODEL_FILE_TYPES + ("LoRA", "VAE", "TextualInversion", "TextEncoder")]
        if not cands:
            cands = list(files)
    safes = [f for f in cands if str(f.get("name") or "").endswith(".safetensors") or (f.get("metadata") or {}).get("format") == "SafeTensor"]
    pool = safes if safes else cands
    pool.sort(key=lambda x: (x.get("sizeKB") or 0), reverse=True)
    if not pool:
        raise RuntimeError("Nenhum arquivo compativel encontrado nessa versao.")
    return pool[0]

def pick_latest_version(model_data, wanted_type=None):
    """Sempre a versao MAIS RECENTE com arquivo do tipo desejado (P3: download recomendado)."""
    versions = civitai_sorted_versions(model_data)
    for v in versions:
        try:
            pick_file_from_version(v, wanted_type)
            return v
        except Exception:
            continue
    if versions:
        return versions[0]
    raise RuntimeError("Nenhuma versao com arquivos.")

# ============================================================================
# 4. DOWNLOADER — aria2c paralelo + requests fallback + resume + verificacao (P2-16/17)
# ============================================================================
def download_with_aria2(url, local_path, headers):
    aria = shutil.which("aria2c")
    if not aria:
        return None
    d = os.path.dirname(local_path)
    Path(d).mkdir(parents=True, exist_ok=True)
    cmd = [aria, "-x", "16", "-s", "16", "-k", "1M", "--continue=true",
           "--auto-file-renaming=false", "--allow-overwrite=true",
           "--summary-interval=0", "--console-log-level=warn",
           "-d", d, "-o", os.path.basename(local_path), url]
    for k, v in headers.items():
        if k.lower() == "authorization":
            cmd += ["--header=" + k + ": " + v]
    try:
        subprocess.run(cmd, timeout=86400, check=True)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
            return local_path
    except Exception as e:
        print("  aria2c falhou, usando requests: " + str(e)[:120])
    return None

def download_file_stream(url, local_path, headers, desc="Baixando", progress_cb=None, expected_bytes=None):
    """Download com streaming, resume (Range) e verificacao de tamanho."""
    # Tenta aria2c para arquivos grandes
    if expected_bytes and expected_bytes > 1024 ** 3:
        r = download_with_aria2(url, local_path, headers)
        if r:
            if progress_cb:
                progress_cb(1, 1, desc + " (aria2c)")
            return local_path
    tmp = local_path + ".part"
    existing = 0
    if os.path.exists(tmp):
        existing = os.path.getsize(tmp)
    hdrs = dict(headers)
    if existing > 0:
        hdrs["Range"] = "bytes=" + str(existing) + "-"
    with requests.get(url, headers=hdrs, stream=True, timeout=120) as r:
        if r.status_code == 416:
            os.replace(tmp, local_path)
            return local_path
        if r.status_code not in (200, 206):
            raise RuntimeError("Falha no download (HTTP " + str(r.status_code) + ").")
        total = int(r.headers.get("content-length", 0)) + existing
        mode = "ab" if (existing > 0 and r.status_code == 206) else "wb"
        done = existing
        last_report = 0
        t0 = time.time()
        with open(tmp, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last_report > 1.0:
                    last_report = now
                    if total > 0 and progress_cb:
                        progress_cb(done, total, desc)
        if progress_cb:
            progress_cb(done, total, desc)
    # Anti-race: se o .part sumiu (outra thread ja renomeou), usa o arquivo final; senao erro claro
    if os.path.exists(tmp):
        os.replace(tmp, local_path)
    elif os.path.exists(local_path):
        return local_path
    else:
        raise RuntimeError("Download interrompido: temporario ausente em " + tmp)
    # Verificacao de tamanho (P2-17)
    if expected_bytes and os.path.exists(local_path):
        actual = os.path.getsize(local_path)
        if abs(actual - expected_bytes) > max(1024, expected_bytes * 0.01):
            print("  WARN tamanho divergente: esperado " + str(expected_bytes) + " obtido " + str(actual))
    return local_path

def write_model_meta(local_path, base_model, model_name, family, version_name, trained_words, file_info):
    """Sidecar JSON com metadados do checkpoint (usado pela biblioteca local)."""
    meta = {
        "base_model": base_model, "model_name": model_name, "family": family,
        "version": version_name, "trained_words": trained_words or [],
        "size_bytes": os.path.getsize(local_path) if os.path.exists(local_path) else None,
        "file_name": file_info.get("name") if file_info else None,
    }
    with open(local_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

def _pick_te_file(version):
    for f in version.get("files", []):
        if (f.get("type") or "") == "Text Encoder":
            return f
    return None

# VAE oficial do workflow Anima (blueprint ComfyUI): circlestone-labs/Anima split_files/vae/qwen_image_vae.safetensors
# (publico). E um WanVAE 2.1 (16ch, convs 3D) — detectado pelo ComfyUI por conteudo.
ANIMA_VAE_URL = "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors"
ANIMA_VAE_BYTES = 253806246

def _anima_vae_valid(path):
    """True se o arquivo e um WanVAE 2.1 (compativel com Anima) no formato que o ComfyUI detecta.
    Header-check (safetensors): presenca de 'decoder.middle.0.residual.0.gamma' e ausencia da chave Wan 2.2.
    O VAE do Qwen/Qwen-Image (diffusion_pytorch_model.safetensors, formato DIFFUSERS com
    decoder.conv_in.weight 2D/128ch) NAO passa — e exatamente o que quebrava no load_state_dict."""
    try:
        if not os.path.exists(path) or os.path.getsize(path) < 4096:
            return False
        with open(path, "rb") as f:
            head = f.read(8)
            if len(head) != 8:
                return False
            n = struct.unpack("<Q", head)[0]
            if n <= 0 or n > 4096 * 1024:
                return False
            hdr = json.loads(f.read(n).decode("utf-8"))
        return ("decoder.middle.0.residual.0.gamma" in hdr and
                "decoder.upsamples.0.upsamples.0.residual.2.weight" not in hdr)
    except Exception:
        return False

def _download_anima_aux(version, token, progress_cb=None):
    """Anima no ComfyUI precisa de 2 auxiliares (baixados uma vez so):
    1) text encoder Qwen3-0.6B: o proprio arquivo 'Text Encoder' do modelo/versao do Civitai;
    2) VAE (latente Wan21/16ch): 'qwen_image_vae.safetensors' OFICIAL do workflow Anima
       (circlestone-labs/Anima split_files/vae, publico, WanVAE 2.1 detectado pelo ComfyUI),
       com fallback p/ 'HDR VAE (Anima, Krea2, QWEN Image)' do Civitai (2718533)."""
    # 1) TE: best-effort — erro no TE NUNCA bloqueia o VAE (obrigatorio)
    try:
        te = _pick_te_file(version)
        if te and te.get("downloadUrl"):
            url = te["downloadUrl"]
            safe = re.sub(r"[^\w\-.]", "_", te.get("name", "anima_te.safetensors"))
            dest = os.path.join(MODELS_DIR, "text_encoders")
            Path(dest).mkdir(parents=True, exist_ok=True)
            local = os.path.join(dest, safe)
            with DOWNLOAD_LOCK:
                if not (os.path.exists(local) and os.path.getsize(local) > 1024):
                    if progress_cb:
                        progress_cb(0.5, 1, "Baixando text encoder Anima (" + fmt_bytes(int((te.get("sizeKB") or 0) * 1024)) + ")...")
                    download_file_stream(url, local, civitai_headers(token),
                                         desc="Baixando " + safe,
                                         expected_bytes=int((te.get("sizeKB") or 0) * 1024))
            STATE["anima_te_path"] = local
            print("  Anima TE: " + os.path.basename(local))
    except Exception as e:
        print("  WARN TE Anima (nao bloqueia o VAE): " + str(e)[:150])
    # 2) VAE: obrigatorio (WanVAE 2.1 ComfyUI-ready). FALHAS VAZAM p/ o caller (sem swallow).
    vae_local = os.path.join(VAE_DIR, "anima_vae.safetensors")
    Path(VAE_DIR).mkdir(parents=True, exist_ok=True)
    with DOWNLOAD_LOCK:
        # So wanVAE 2.1 (formato ComfyUI) e viavel: o VAE do Qwen/Qwen-Image (diffusers,
        # decoder.conv_in.weight 2D/128ch) quebra no ComfyUI (size mismatch no load_state_dict).
        if not _anima_vae_valid(vae_local):
            if os.path.exists(vae_local):
                print("  VAE Anima invalido/incompativel, baixando o oficial de novo...")
                try:
                    os.remove(vae_local)
                except Exception:
                    pass
            ok_vae = False
            # 1) VAE oficial do workflow Anima (circlestone-labs/Anima, publico, ComfyUI-ready)
            try:
                if progress_cb:
                    progress_cb(0.55, 1, "Baixando VAE Anima oficial (qwen_image_vae)...")
                download_file_stream(ANIMA_VAE_URL, vae_local, {},
                                     desc="Baixando qwen_image_vae",
                                     progress_cb=progress_cb,
                                     expected_bytes=ANIMA_VAE_BYTES)
                ok_vae = _anima_vae_valid(vae_local)
            except Exception as e:
                print("  WARN VAE oficial Anima: " + str(e)[:150])
            # 2) fallback: HDR VAE (Anima, Krea2, QWEN Image) via Civitai
            if not ok_vae:
                try:
                    if progress_cb:
                        progress_cb(0.55, 1, "Baixando VAE Anima (HDR, fallback)...")
                    got = download_from_civitai("2718533", token, "Model", 0, progress_cb)
                    if got:
                        shutil.copy(got[0], vae_local)
                    ok_vae = _anima_vae_valid(vae_local)
                except Exception as e:
                    print("  WARN VAE HDR Anima: " + str(e)[:150])
            if not ok_vae:
                raise RuntimeError("VAE Anima indisponivel (oficial circlestone-labs e fallback HDR falharam).")
    STATE["anima_vae_path"] = vae_local
    print("  Anima VAE: anima_vae.safetensors")

def download_from_civitai(model_id, token, wanted_type=None, version_index=0, progress_cb=None, use_latest=True):
    """Baixa SEMPRE a versao mais recente (padrao) com o arquivo recomendado."""
    data = civitai_get_model(model_id, token)
    model_name = data.get("name", "unknown_model")
    versions = civitai_sorted_versions(data)
    if not versions:
        raise RuntimeError("Nenhuma versao com arquivos.")
    if use_latest:
        version = pick_latest_version(data, wanted_type)
    else:
        idx = min(int(version_index), len(versions) - 1)
        version = versions[idx]
    base_model = version.get("baseModel") or "Other"
    family = _family_from_base(base_model)
    trained_words = version.get("trainedWords") or []
    if family == "krea2" and wanted_type in (None, "Model"):
        target = _pick_krea_file(version)
    else:
        target = pick_file_from_version(version, wanted_type)
    download_url = target.get("downloadUrl")
    if not download_url:
        raise RuntimeError("URL de download ausente.")
    file_name = target.get("name", "model.safetensors")
    safe_name = re.sub(r"[^\w\-.]", "_", file_name)
    if wanted_type == "LoRA":
        dest_dir = LORA_DIR
    elif wanted_type == "VAE":
        dest_dir = VAE_DIR
    elif wanted_type == "TextualInversion":
        dest_dir = TI_DIR
    else:
        dest_dir = CIVITAI_DIR
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    local_path = os.path.join(dest_dir, safe_name)
    expected = int((target.get("sizeKB") or 0) * 1024)
    # Anima: baixa TE (Qwen3-0.6B) + VAE Wan21 do Civitai (guard: nao recursa no proprio 2718533)
    if base_model == "Anima" and wanted_type in (None, "Model") and str(model_id) != "2718533":
        _download_anima_aux(version, token, progress_cb)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
        if progress_cb:
            progress_cb(1, 1, "Ja baixado: " + safe_name)
        if wanted_type in (None, "Model"):
            write_model_meta(local_path, base_model, model_name, family, version.get("name"), trained_words, target)
        return local_path, base_model, model_name, family, trained_words
    if progress_cb:
        progress_cb(0, max(1, expected), "Baixando " + safe_name + " (" + fmt_bytes(expected) + ")...")
    # DOWNLOAD_LOCK: impede 2 threads baixando o MESMO arquivo no mesmo .part (race -> os.replace FileNotFoundError)
    with DOWNLOAD_LOCK:
        if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
            if expected and abs(os.path.getsize(local_path) - expected) > max(1024, expected * 0.02):
                print("  WARN arquivo existente com tamanho divergente, baixando de novo: " + safe_name)
            elif safe_name.endswith(".safetensors") and wanted_type in (None, "Model") and not _safetensors_valid(local_path):
                print("  WARN arquivo existente INVALIDO (" + safe_name + ") — removendo e rebaixando...")
                try:
                    os.remove(local_path)
                except Exception:
                    pass
            else:
                if progress_cb:
                    progress_cb(1, 1, "Ja baixado: " + safe_name)
                if wanted_type in (None, "Model"):
                    write_model_meta(local_path, base_model, model_name, family, version.get("name"), trained_words, target)
                return local_path, base_model, model_name, family, trained_words
        download_file_stream(download_url, local_path, civitai_headers(token),
                             desc="Baixando " + safe_name, progress_cb=progress_cb, expected_bytes=expected)
    if safe_name.endswith(".safetensors") and wanted_type in (None, "Model") and not _safetensors_valid(local_path):
        # Download salvo mas INVALIDO (HTML de erro / corrompido) — remove e da erro acionavel
        sz = os.path.getsize(local_path) if os.path.exists(local_path) else 0
        try:
            os.remove(local_path)
        except Exception:
            pass
        raise RuntimeError(
            "Download INVALIDO: " + safe_name + " (" + str(sz) + " bytes) nao e um safetensors valido — "
            "o servidor provavelmente retornou HTML/erro. Tente novamente; se persistir, preencha o token do Civitai.")
    if wanted_type in (None, "Model"):
        write_model_meta(local_path, base_model, model_name, family, version.get("name"), trained_words, target)
    return local_path, base_model, model_name, family, trained_words

# ============================================================================
# 5. UTILIDADES — VRAM budget (P2-13) / clamp / limpeza temp (P0-5)
# ============================================================================
def _enhance(prompt, style_preset="None"):
    if not prompt:
        return ""
    p = prompt.lower()
    if any(k in p for k in ["photorealistic", "hyperrealistic", "highly detailed", "8k resolution"]):
        enhanced = prompt
    elif any(w in p for w in ["person", "woman", "man", "girl", "boy", "portrait", "face"]):
        enhanced = prompt + ", 85mm lens, f/1.8, natural skin texture, professional studio lighting, highly detailed, 8k"
    elif any(w in p for w in ["cyberpunk", "futuristic", "sci-fi", "robot", "neon"]):
        enhanced = prompt + ", cyberpunk aesthetic, neon glow, volumetric fog, cinematic lighting, ray tracing, 8k"
    elif any(w in p for w in ["forest", "mountain", "ocean", "landscape", "lake", "river", "sky", "sunset"]):
        enhanced = prompt + ", golden hour, national geographic style, volumetric mist, wide-angle lens, 8k"
    elif any(w in p for w in ["dragon", "magic", "wizard", "elf", "castle", "fantasy"]):
        enhanced = prompt + ", glowing magical particles, ethereal light, highly detailed fantasy art, 8k"
    elif any(w in p for w in ["anime", "illustration", "drawing", "painting", "digital art"]):
        enhanced = prompt + ", vibrant color palette, clean line art, beautiful lighting, anime key visual, masterpiece"
    else:
        enhanced = prompt + ", highly detailed, photorealistic, cinematic lighting, masterpiece, 8k"
    styles = {
        "Cinematic": "cinematic lighting, dramatic shadows, film grain, masterpiece",
        "Photographic": "professional photography, 35mm lens, f/2.8, depth of field, photorealistic",
        "Anime": "anime key visual, vibrant color palette, clean line art, highly detailed",
        "Cyberpunk": "cyberpunk aesthetic, neon lights, volumetric smoke, high contrast",
        "Fantasy": "mythical atmosphere, glowing particles, ethereal light, digital painting",
        "Pixel Art": "pixel art, 16-bit, dithering, retro game graphics",
        "Oil Painting": "oil painting, impasto, canvas texture, classical art style",
    }
    if style_preset and style_preset != "None" and style_preset in styles:
        enhanced += ", " + styles[style_preset]
    return enhanced

def _parse_px(value, default=1024):
    try:
        if isinstance(value, (int, float)):
            return int(value)
        m = re.search(r"(\d+)", str(value))
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return default

def _dims(aspect, base_px):
    b = int(base_px)
    if aspect == "1:1 Square":
        return b, b
    if aspect == "16:9 Landscape":
        return b, max(16, int(b * 9 / 16 // 16 * 16))
    if aspect == "9:16 Portrait":
        return max(16, int(b * 9 / 16 // 16 * 16)), b
    if aspect == "4:3 Standard":
        return b, max(16, int(b * 3 / 4 // 16 * 16))
    if aspect == "3:4 Portrait":
        return max(16, int(b * 3 / 4 // 16 * 16)), b
    return b, b

def vram_info():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        total = info.total / 1024 ** 3
        free = info.free / 1024 ** 3
        return free, total
    except Exception:
        try:
            if torch.cuda.is_available():
                # total_mem foi renomeado p/ total_memory no PyTorch >= 2.9;
                # mem_get_info() e estavel em todas as versoes
                free, total = torch.cuda.mem_get_info()
                return free / 1024 ** 3, total / 1024 ** 3
        except Exception:
            pass
        return None, None

def clamp_resolution(base_px, family):
    """Reduz resolucao automaticamente se a VRAM livre estiver apertada."""
    free, _ = vram_info()
    if free is None:
        return base_px
    if family in ("flux_dev", "flux_schnell", "sd3", "sd35", "hidream", "chroma"):
        if free < 4.5:
            return 512
        if free < 7.0:
            return 768
        return base_px
    if family in ("sdxl", "pony", "illustrious", "noobai", "animagine"):
        if free < 4.0:
            return 768
        if free < 6.0:
            return 768 if base_px > 768 else base_px
        return base_px
    if free < 3.0:
        return 512
    return base_px

def _cleanup_tmp():
    try:
        for f in Path(APP_DIR).glob("init_tmp_*.png"):
            try:
                f.unlink()
            except Exception:
                pass
    except Exception:
        pass

# ============================================================================
# 6. MOTOR DIFFUSERS
# ============================================================================
def _load_te_and_tok(family, dtype, hf_token):
    """Text encoders/VAE por familia, usando mirrors NAO-gated quando possivel (P0-3)."""
    from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection, T5EncoderModel, T5TokenizerFast
    from diffusers import AutoencoderKL
    comps = {}
    hf_kwargs = {"token": hf_token} if hf_token else {}
    if family in ("sdxl", "pony", "illustrious", "noobai", "animagine", "sdxl_lightning", "sdxl_lcm", "sdxl_hyper"):
        repo = "stabilityai/stable-diffusion-xl-base-1.0"
        comps["text_encoder"] = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder", torch_dtype=dtype, **hf_kwargs)
        comps["text_encoder_2"] = CLIPTextModelWithProjection.from_pretrained(repo, subfolder="text_encoder_2", torch_dtype=dtype, **hf_kwargs)
        comps["tokenizer"] = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer", **hf_kwargs)
        comps["tokenizer_2"] = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer_2", **hf_kwargs)
        comps["vae"] = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype, **hf_kwargs)
    elif family in ("sd15", "sd15_hyper", "sd15_lcm"):
        repo = "runwayml/stable-diffusion-v1-5"
        comps["text_encoder"] = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder", torch_dtype=dtype, **hf_kwargs)
        comps["tokenizer"] = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer", **hf_kwargs)
        comps["vae"] = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype, **hf_kwargs)
    elif family in ("sd2", "sd2_768"):
        repo = "stabilityai/stable-diffusion-2-1"
        comps["text_encoder"] = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder", torch_dtype=dtype, **hf_kwargs)
        comps["tokenizer"] = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer", **hf_kwargs)
        comps["vae"] = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype, **hf_kwargs)
    elif family in ("flux_dev", "flux_schnell", "flux2_klein"):
        comps = _flux_components(dtype, hf_token)
    return comps

def _flux_components(dtype, hf_token):
    """Componentes FLUX de mirrors publicos (comfyanonymous/flux_text_encoders) — P0-3."""
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
    from diffusers import AutoencoderKL
    comps = {}
    comp_dir = os.path.join(HF_CACHE, "flux_text_encoders")
    Path(comp_dir).mkdir(parents=True, exist_ok=True)

    def mirror_file(fname, url, dest_sub="text_encoders"):
        dest = os.path.join(comp_dir, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            return dest
        print("  Baixando componente FLUX (mirror): " + fname)
        try:
            r = requests.get(url, stream=True, timeout=600)
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            return dest
        except Exception as e:
            print("  WARN mirror " + fname + " falhou: " + str(e)[:120])
            return None

    BASE_M = "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/"
    t5_file = mirror_file("t5xxl_fp8_e4m3fn.safetensors", BASE_M + "t5xxl_fp8_e4m3fn.safetensors")
    clip_file = mirror_file("clip_l.safetensors", BASE_M + "clip_l.safetensors")
    try:
        comps["text_encoder"] = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype, token=hf_token or None)
        comps["tokenizer"] = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", token=hf_token or None)
    except Exception:
        pass
    if t5_file:
        try:
            from safetensors import safe_open
            from transformers import T5Config
            cfg = T5Config.from_pretrained("google/t5-xxl-encoder", torch_dtype=dtype)
            model = T5EncoderModel(cfg)
            with safe_open(t5_file, framework="pt") as sf:
                sd = {k: sf.get_tensor(k) for k in sf.keys()}
            # converte fp8 -> dtype do modelo p/ load_state_dict nao falhar
            sd = {k: (v.to(dtype) if v.is_floating_point() and v.dtype != dtype else v) for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
            model = model.to(dtype)
            comps["text_encoder_2"] = model
        except Exception as e:
            print("  WARN t5 fp8 load: " + str(e)[:120])
            comps["text_encoder_2"] = T5EncoderModel.from_pretrained("google/t5-xxl-encoder", torch_dtype=dtype, token=hf_token or None)
    else:
        comps["text_encoder_2"] = T5EncoderModel.from_pretrained("google/t5-xxl-encoder", torch_dtype=dtype, token=hf_token or None)
    comps["tokenizer_2"] = T5TokenizerFast.from_pretrained("google/t5-xxl-encoder", token=hf_token or None)
    # VAE ae.safetensors — tenta BFL (gated, precisa token) e mirrors
    ae = None
    for url, tok in [
        ("https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors", hf_token),
        ("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors", hf_token),
    ]:
        try:
            headers = {"Authorization": "Bearer " + tok} if tok else {}
            r = requests.get(url, headers=headers, stream=True, timeout=600)
            if r.status_code == 200:
                ae_dest = os.path.join(comp_dir, "ae.safetensors")
                with open(ae_dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                ae = ae_dest
                break
        except Exception:
            continue
    if ae:
        try:
            comps["vae"] = AutoencoderKL.from_single_file(ae, torch_dtype=dtype)
        except Exception:
            pass
    else:
        print("  WARN ae.safetensors indisponivel (FLUX precisa de token HF para o VAE). Tentando SDXL VAE compat.")
        try:
            comps["vae"] = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype, token=hf_token or None)
        except Exception:
            pass
    return comps

def _install_scheduler(pipe, family):
    try:
        sched_name = FAMILY_PRESETS[family].get("scheduler", "DPM++ 2M Karras")
        spec = SCHEDULERS.get(sched_name)
        if not spec or pipe.scheduler is None:
            return
        cls_name, kwargs = spec
        import diffusers.schedulers as sch
        cls = getattr(sch, cls_name)
        pipe.scheduler = cls.from_config(pipe.scheduler.config, **kwargs)
    except Exception as e:
        print("  (scheduler fallback: " + str(e) + ")")

def _set_pipe_offload(pipe, mode="model"):
    if mode == "sequential":
        try:
            pipe.enable_sequential_cpu_offload()
            return
        except Exception:
            pass
    try:
        pipe.enable_model_cpu_offload()
        return
    except Exception:
        pass
    try:
        pipe.to("cuda")
    except Exception:
        pass
    try:
        if hasattr(pipe, "vae") and pipe.vae is not None:
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
    except Exception:
        pass

def _quantize_model(pipe, method, family):
    """Quantizacao opcional (P2-14): torchao int8 / fp8_e4m3fn / bitsandbytes 4-bit / 8-bit."""
    if not method or method == "none":
        return False
    model = None
    if hasattr(pipe, "transformer"):
        model = pipe.transformer
    elif hasattr(pipe, "unet"):
        model = pipe.unet
    if model is None:
        return False
    try:
        if method == "torchao_int8":
            try:
                from torchao.quantization import quantize_, weight_only_int8
                quantize_(model, weight_only_int8())
            except Exception:
                from torchao.quantization.quant_api import quantize_ as _q2
                from torchao.quantization.quant_api import weight_only_int8 as _wo8
                _q2(model, _wo8())
            print("  Quantizado INT8 (torchao) — " + family)
            return True
        if method == "fp8_e4m3fn":
            model.to(torch.float8_e4m3fn)
            print("  Quantizado FP8 (fp8_e4m3fn) — " + family)
            return True
        if method == "bnb_8bit":
            import bitsandbytes as bnb
            from diffusers.utils.quantization import quantize_model_as_8bit
            quantize_model_as_8bit(model)
            print("  Quantizado 8-bit (bitsandbytes) — " + family)
            return True
        if method == "bnb_4bit":
            import bitsandbytes as bnb
            from diffusers.utils.quantization import quantize_model_as_4bit
            quantize_model_as_4bit(model, bnb_4bit_compute_dtype=torch.float16)
            print("  Quantizado 4-bit (bitsandbytes) — " + family)
            return True
    except Exception as e:
        print("  (quantizacao " + method + " indisponivel: " + str(e)[:150] + ")")
    return False

def _apply_compile(pipe, family):
    if not STATE.get("use_compile"):
        return
    try:
        model = None
        if hasattr(pipe, "transformer"):
            model = pipe.transformer
        elif hasattr(pipe, "unet"):
            model = pipe.unet
        if model is not None:
            model = torch.compile(model)
            if hasattr(pipe, "transformer"):
                pipe.transformer = model
            elif hasattr(pipe, "unet"):
                pipe.unet = model
            print("  torch.compile aplicado")
    except Exception as e:
        print("  (torch.compile falhou: " + str(e)[:120] + ")")

def load_diffusers_single_file(model_path, base_model, family, cfg):
    """Carga single-file diffusers com cadeia de fallbacks por familia + P0-3/P2-14."""
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusion3Pipeline, FluxPipeline
    dtype = get_torch_dtype(cfg.get("dtype", "float16"))
    is_st = str(model_path).lower().endswith(".safetensors")
    preset = FAMILY_PRESETS.get(family, FAMILY_PRESETS["other"])
    hf_token = STATE.get("hf_token")
    errors = []

    def try_inject(pipe_cls):
        try:
            pipe = pipe_cls.from_single_file(model_path, torch_dtype=dtype, use_safetensors=is_st,
                                             safety_checker=None, requires_safety_checker=False)
            return pipe, "OK"
        except Exception as e1:
            errors.append("direto: " + str(e1)[:200])
        comps = _load_te_and_tok(family, dtype, hf_token)
        if comps:
            try:
                pipe = pipe_cls.from_single_file(model_path, torch_dtype=dtype, use_safetensors=is_st,
                                                 safety_checker=None, requires_safety_checker=False, **comps)
                return pipe, "com TE/VAE injetados"
            except Exception as e2:
                errors.append("com injecao: " + str(e2)[:200])
        try:
            from diffusers import UNet2DConditionModel
            unet = UNet2DConditionModel.from_pretrained(preset["te_repo"], subfolder="unet", torch_dtype=dtype, token=hf_token or None)
            comps2 = _load_te_and_tok(family, dtype, hf_token)
            pipe = pipe_cls.from_single_file(model_path, unet=unet, **comps2, torch_dtype=dtype,
                                             use_safetensors=is_st, safety_checker=None, requires_safety_checker=False)
            return pipe, "com UNet+TE+VAE do base"
        except Exception as e3:
            errors.append("UNet base: " + str(e3)[:200])
        raise RuntimeError("Falha ao carregar " + family + ": " + " | ".join(errors))

    if family in ("sd15", "sd15_hyper", "sd15_lcm", "sd2", "sd2_768"):
        pipe, msg = try_inject(StableDiffusionPipeline)
    elif family in ("sdxl", "pony", "illustrious", "noobai", "animagine",
                    "sdxl_lightning", "sdxl_lcm", "sdxl_hyper"):
        pipe, msg = try_inject(StableDiffusionXLPipeline)
    elif family in ("flux_dev", "flux_schnell", "flux2_klein"):
        try:
            pipe = FluxPipeline.from_single_file(model_path, torch_dtype=dtype, use_safetensors=is_st)
            msg = "OK"
        except Exception as e1:
            errors.append("flux: " + str(e1)[:200])
            comps = _load_te_and_tok(family, dtype, hf_token)
            pipe = FluxPipeline.from_single_file(model_path, torch_dtype=dtype, use_safetensors=is_st, **comps)
            msg = "com T5/CLIP/VAE injetados (mirror)"
        _quantize_model(pipe, STATE.get("quant_method", preset.get("quantize", "none")), family)
    elif family in ("sd3", "sd35"):
        pipe, msg = try_inject(StableDiffusion3Pipeline)
    elif family in ("zimage", "zimage_turbo"):
        from diffusers import ZImagePipeline, ZImageTransformer2DModel
        transformer = ZImageTransformer2DModel.from_single_file(model_path, torch_dtype=dtype)
        pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", transformer=transformer, torch_dtype=dtype)
        msg = "OK (transformer single-file + pipeline base)"
    elif family == "qwen":
        from diffusers import QwenImagePipeline
        pipe = QwenImagePipeline.from_single_file(model_path, torch_dtype=dtype, use_safetensors=is_st)
        msg = "OK"
    elif family == "chroma":
        from diffusers import ChromaPipeline, ChromaTransformer2DModel
        transformer = ChromaTransformer2DModel.from_single_file(model_path, torch_dtype=dtype)
        pipe = ChromaPipeline.from_pretrained("lodestones/Chroma", transformer=transformer, torch_dtype=dtype)
        msg = "OK (transformer + pipeline Chroma)"
    elif family == "anima":
        try:
            from diffusers import AnimaPipeline
            pipe = AnimaPipeline.from_single_file(model_path, torch_dtype=dtype)
            msg = "OK (AnimaPipeline)"
        except Exception:
            try:
                from diffusers_anima import AnimaPipeline
                pipe = AnimaPipeline.from_single_file(model_path, torch_dtype=dtype)
                msg = "OK (diffusers-anima)"
            except Exception as e2:
                raise RuntimeError("Anima sem suporte diffusers (use ComfyUI): " + str(e2)[:150])
    else:
        raise RuntimeError("Familia sem loader diffusers direto: " + family)

    print("  Carga: " + msg)
    _install_scheduler(pipe, family)
    _apply_compile(pipe, family)
    _set_pipe_offload(pipe, cfg.get("cpu_offload", "model"))
    STATE["pipe"] = pipe
    STATE["backend"] = "diffusers"
    STATE["family"] = family
    STATE["loaded"] = True
    return pipe


KREA2_WORKER_PORT = 7862
KREA2_WORKER_PROGRESS = os.path.join(APP_DIR, "krea2_worker_progress.json")
KREA2_WORKER_SRC = r"""# -*- coding: utf-8 -*-
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


def do_generate(krea_model, prompt, negative, steps, width, height, cfg, seed, loras=None):
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
            loras_slists={"phase1": [(x["path"], float(x["scale"])) for x in (loras or []) if os.path.exists(x.get("path", ""))]},
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
                            loras=data.get("loras"),
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
"""


def _write_krea_worker_file():
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


def _ensure_wan2gp_deps(progress_cb=None):
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
    for _pin in ("mmgp==3.7.12", "gradio==5.29.0", "optimum-quanto==0.2.7",
                 "smplfitter==0.2.10", "torchao>=0.16.0"):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", _pin], timeout=600)
        except Exception as _e:
            print("  WARN pip " + _pin + ":", str(_e)[:100])


def _spawn_krea_worker(ckpt_path, model_name, progress_cb=None):
    """Sobe o worker Krea-2-Turbo em processo limpo (padrao dos notebooks validados).
    Se o OOM memcg 12Gi matar o worker, SO ele morre — kernel/sessao/app continuam vivos."""
    if not os.path.exists(ckpt_path):
        raise RuntimeError("Checkpoint Krea2 nao encontrado: " + ckpt_path)
    te = os.path.join(WAN2GP_DIR, "models", "Qwen3-VL-4B-Instruct", "Qwen3-VL-4B-Instruct_quanto_bf16_int8.safetensors")
    if not os.path.exists(WAN2GP_DIR):
        if progress_cb:
            progress_cb(0.10, 1.0, "Clonando Wan2GP...")
        subprocess.check_call(["git", "clone", "-q", "https://github.com/DeepBeepMeep/Wan2GP.git", WAN2GP_DIR], timeout=900)
    _ensure_wan2gp_deps(progress_cb=progress_cb)
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
    env["WAN2GP_DIR"] = WAN2GP_DIR
    logf = open(wlog, "w", encoding="utf-8")
    cmd = [sys.executable, "-u", os.path.join(APP_DIR, "krea2_worker.py"),
           "--ckpt", ckpt_path, "--te", te, "--model", str(model_name), "--port", str(port),
           "--wan2gp", WAN2GP_DIR]
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
        raise RuntimeError("Worker Krea2 nao subiu:\n" + tail)
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
    loras_payload = [{"path": p, "scale": float(w)} for (p, w) in (STATE.get("lora_stack") or []) if os.path.exists(p)]
    payload = {
        "prompt": prompt, "negative": negative or "", "steps": int(steps),
        "width": int(width), "height": int(height), "cfg": float(cfg or 0),
        "seed": int(seed), "loras": loras_payload,
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


# ============================================================================
# 8. MOTOR COMFYUI (fallback universal + inpaint/controlnet/hires) (P1-9/10)
# ============================================================================
COMFY_PORT = 8188
COMFY_BASE = "http://127.0.0.1:" + str(COMFY_PORT)
_comfy_proc = None
_comfy_lock = threading.Lock()



def _safetensors_valid(path, min_size=1024, verbose=False):
    """True se o arquivo parece um safetensors valido (header JSON parseavel).
    verbose=True imprime o motivo da falha (diagnostico)."""
    try:
        if not path or not os.path.exists(path):
            if verbose:
                print("  [check] arquivo nao existe:", path)
            return False
        size = os.path.getsize(path)
        if size < min_size:
            if verbose:
                print("  [check] arquivo muito pequeno:", size, "bytes")
            return False
        with open(path, "rb") as f:
            raw = f.read(8)
            n = struct.unpack("<Q", raw)[0]
            if n <= 0 or n > 512 * 1024 * 1024:
                if verbose:
                    print("  [check] tamanho do header invalido:", n)
                return False
            hdr = f.read(n)
            txt = hdr.decode("utf-8", errors="replace")
            if txt.startswith("﻿"):
                txt = txt[1:]
            data = json.loads(txt)
        if not isinstance(data, dict):
            if verbose:
                print("  [check] header JSON nao e um objeto")
            return False
        return True
    except Exception as e:
        if verbose:
            print("  [check] falha ao ler header: " + str(e)[:120])
        return False


def _read_log_tail(path, n=40):
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().strip().splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return "(log indisponivel)"

def ensure_comfyui():
    global _comfy_proc
    with _comfy_lock:
        if _comfy_proc is not None and _comfy_proc.poll() is None:
            return True
        if not os.path.exists(os.path.join(COMFY_DIR, "main.py")):
            # Diretorio invalido/incompleto (clone quebrado ou criado por engano, ex.: repair_vae
            # em VM sem ComfyUI) — remove ANTES do clone: 'git clone' recusa destino nao vazio (exit 128).
            if os.path.exists(COMFY_DIR):
                print("  Removendo /content/ComfyUI invalido/incompleto antes de clonar...")
                try:
                    shutil.rmtree(COMFY_DIR)
                except Exception as e:
                    print("  (falha ao remover: " + str(e)[:100] + " — tentando clone mesmo assim)")
            print("  Instalando ComfyUI (universal engine)...")
            last_err = ""
            cloned = False
            for attempt in range(3):
                try:
                    r = subprocess.run(
                        ["git", "clone", "--depth", "1", "-q",
                         "https://github.com/comfyanonymous/ComfyUI.git", COMFY_DIR],
                        capture_output=True, text=True, timeout=900)
                    if r.returncode == 0 and os.path.exists(os.path.join(COMFY_DIR, "main.py")):
                        cloned = True
                        break
                    last_err = ((r.stderr or "") + " " + (r.stdout or "")).strip()[-400:]
                    print("  git clone tentativa " + str(attempt + 1) + "/3 falhou: " + last_err[:180])
                except Exception as e:
                    last_err = str(e)
                    print("  git clone tentativa " + str(attempt + 1) + "/3 erro: " + str(e)[:180])
                time.sleep(5)
            if not cloned:
                raise RuntimeError("Falha ao clonar ComfyUI (3 tentativas): " + (last_err or "sem detalhes do git"))
        # Libs de sistema comuns em Colab limpo (libGL.so.1, etc.) — matam o ComfyUI no startup
        try:
            subprocess.run(["apt-get", "update", "-qq"], timeout=180, capture_output=True)
            subprocess.run(["apt-get", "install", "-y", "-qq", "libgl1", "libglib2.0-0",
                            "libsm6", "libxext6", "libxrender1"], timeout=300, capture_output=True)
        except Exception as e:
            print("  (apt libs: " + str(e)[:100] + ")")
        req = os.path.join(COMFY_DIR, "requirements.txt")
        if os.path.exists(req):
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-input",
                                       "--timeout", "180", "-r", req], timeout=1800)
            except Exception as e:
                print("  WARN pip ComfyUI: " + str(e)[:200] + " (continuando — core deps geralmente ja existem)")
        # Mata processo ComfyUI anterior (evita 'address already in use' na porta)
        try:
            subprocess.run(["pkill", "-f", "ComfyUI.*main.py"], capture_output=True)
            time.sleep(2)
        except Exception:
            pass
        log = open(os.path.join(APP_DIR, "comfy.log"), "w")
        env = dict(os.environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        _comfy_proc = subprocess.Popen(
            [sys.executable, "main.py", "--listen", "127.0.0.1",
             "--port", str(COMFY_PORT), "--disable-auto-launch"],
            cwd=COMFY_DIR, stdout=log, stderr=subprocess.STDOUT, env=env)
        t0 = time.time()
        while time.time() - t0 < 300:
            try:
                r = requests.get(COMFY_BASE + "/system_stats", timeout=3)
                if r.status_code == 200:
                    print("  ComfyUI pronto (porta " + str(COMFY_PORT) + ")")
                    return True
            except Exception:
                pass
            if _comfy_proc.poll() is not None:
                raise RuntimeError("ComfyUI morreu. Ultimas linhas de comfy.log:\n" +
                                   _read_log_tail(os.path.join(APP_DIR, "comfy.log")))
            time.sleep(2)
        raise RuntimeError("Timeout aguardando ComfyUI (300s). Log:\n" +
                           _read_log_tail(os.path.join(APP_DIR, "comfy.log")))
    return True

def comfy_ensure_aux_files(template, family):
    """Aux files: text encoders (salvos em clip e text_encoders), VAE e ControlNet (P0-3/P1-10). Multi-URL com fallback."""
    base_te = "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/"
    needs = []
    if template in ("flux", "sd3"):
        needs.append(("clip_l.safetensors", "clip", [base_te + "clip_l.safetensors"]))
        needs.append(("t5xxl_fp8_e4m3fn.safetensors", "clip", [base_te + "t5xxl_fp8_e4m3fn.safetensors"]))
    if template == "flux":
        # ae (VAE do flux). BFL e GATED (licenca); Kijai/flux-fp8 flux-vae-bf16 e PUBLICO —
        # primeiro mirror = sempre disponivel (167 MB, bf16, 100% compativel com VAEDecode).
        needs.append(("ae.safetensors", "vae", [
            "https://huggingface.co/Kijai/flux-fp8/resolve/main/flux-vae-bf16.safetensors",
            "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors",
            "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors",
        ]))
    if template == "sd3":
        needs.append(("clip_g.safetensors", "clip", [base_te + "clip_g.safetensors"]))
        needs.append(("sd_vae.safetensors", "vae", [
            "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors"]))
    if template == "anima":
        # TE (Qwen3-0.6B) e VAE (Wan21) — TE best-effort; VAE com ultima linha de defesa:
        # se /studio nao tiver um WanVAE valido, baixa o oficial direto no ComfyUI.
        te_src = STATE.get("anima_te_path")
        vae_src = STATE.get("anima_vae_path")
        if te_src and os.path.exists(te_src):
            needs.append((os.path.basename(te_src), "clip", [te_src]))
        vae_ok = bool(vae_src) and os.path.exists(vae_src) and _anima_vae_valid(vae_src)
        if vae_ok:
            needs.append(("anima_vae.safetensors", "vae", [vae_src]))
        else:
            print("  ComfyUI aux: VAE Anima ausente/invalido em /studio — baixando oficial agora...")
            needs.append(("anima_vae.safetensors", "vae", [ANIMA_VAE_URL]))
        if not te_src and not vae_src:
            # modelo carregado de arquivo local (sem dados do Civitai) — mensagem clara
            needs.append(("anima_te.safetensors", "clip", []))
    preset = FAMILY_PRESETS.get(family, {})
    cn = preset.get("controlnet")
    if cn:
        if "control_v11p_sd15" in cn:
            needs.append((cn, "controlnet", ["https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/" + cn]))
        elif cn == "diffusion_pytorch_model.safetensors":
            needs.append((cn, "controlnet", ["https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors"]))
    # Tamanho minimo por arquivo: um download truncado/pagina de erro passa no check >1024 bytes.
    # t5xxl_fp8 real = 4.87 GB; clip_l = 244 MB; clip_g = 1.75 GB; ae = 335 MB; sd_vae = 330 MB.
    MIN_SIZES = {"t5xxl_fp8_e4m3fn.safetensors": 2 << 30, "clip_l.safetensors": 200 << 20,
                 "clip_g.safetensors": 1 << 30, "ae.safetensors": 100 << 20,
                 "sd_vae.safetensors": 200 << 20}
    for fname, sub, urls in needs:
        dest = os.path.join(COMFY_DIR, "models", sub, fname)
        # VAE Anima: exige formato WanVAE 2.1 (header-check) — arquivo diffusers antigo (Qwen/Qwen-Image)
        # passaria no check de tamanho mas quebraria o ComfyUI no load_state_dict.
        already = os.path.exists(dest) and os.path.getsize(dest) > 1024
        if already and fname == "anima_vae.safetensors":
            already = _anima_vae_valid(dest)
            if not already:
                print("  ComfyUI aux: anima_vae.safetensors invalido no destino, recopiando...")
        minb = MIN_SIZES.get(fname, 1024)
        if already and fname != "anima_vae.safetensors" and os.path.getsize(dest) < minb:
            print("  ComfyUI aux: " + fname + " truncado (" + fmt_bytes(os.path.getsize(dest)) + "), rebaixando...")
            already = False
        if already:
            # Sincronizar tambem com pasta secundaria (ex: text_encoders <-> clip)
            if sub in ("clip", "text_encoders"):
                other_sub = "text_encoders" if sub == "clip" else "clip"
                alt = os.path.join(COMFY_DIR, "models", other_sub, fname)
                Path(os.path.dirname(alt)).mkdir(parents=True, exist_ok=True)
                if not os.path.exists(alt):
                    try:
                        os.symlink(dest, alt)
                    except Exception:
                        shutil.copy(dest, alt)
            continue
        Path(os.path.dirname(dest)).mkdir(parents=True, exist_ok=True)
        ok = False
        for url in urls:
            print("  ComfyUI aux: " + fname + " <- " + str(url).split("/")[-1])
            try:
                if os.path.exists(url):
                    # url local (ja baixado pelo app) — copia direto
                    shutil.copy(url, dest)
                    ok = (fname != "anima_vae.safetensors") or _anima_vae_valid(dest)
                    if ok and fname != "anima_vae.safetensors":
                        ok = os.path.getsize(dest) >= minb
                    if ok:
                        break
                    try:
                        os.remove(dest)
                    except Exception:
                        pass
                    continue
                hdrs = {}
                tok = STATE.get("hf_token")
                if tok:
                    hdrs["Authorization"] = "Bearer " + tok
                r = requests.get(url, headers=hdrs, stream=True, timeout=900)
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                if fname == "anima_vae.safetensors":
                    ok = _anima_vae_valid(dest)
                else:
                    ok = os.path.getsize(dest) >= minb
                if ok:
                    break
            except Exception as e:
                print("  WARN aux " + fname + " (" + str(url).split("/")[-1] + "): " + str(e)[:120])
        if ok and sub in ("clip", "text_encoders"):
            other_sub = "text_encoders" if sub == "clip" else "clip"
            alt = os.path.join(COMFY_DIR, "models", other_sub, fname)
            Path(os.path.dirname(alt)).mkdir(parents=True, exist_ok=True)
            if not os.path.exists(alt):
                try:
                    os.symlink(dest, alt)
                except Exception:
                    shutil.copy(dest, alt)
        if not ok:
            extra = (" Carregue o modelo via URL do Civitai para baixar TE/VAE automaticamente."
                     if fname.startswith("anima") else "")
            raise RuntimeError("Arquivo auxiliar do ComfyUI indisponivel: " + fname +
                               " (nenhum mirror). Se for gated, preencha o HF token na aba Modelo/Configuracao." + extra)


def comfy_build_workflow(family, ckpt, prompt, negative, width, height, steps, cfg,
                         sampler, scheduler, seed, denoise=1.0, loras=None,
                         init_image_path=None, mask_path=None, controlnet_path=None):
    valid_samplers = ["euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral",
                      "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
                      "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "ddim", "uni_pc", "uni_pc_bh2",
                      "lcm", "ipndm", "ipndm_v", "deis"]
    if sampler not in valid_samplers:
        sampler = "euler" if sampler in ("krea", "custom") else "dpmpp_2m"

    valid_schedulers = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta"]
    if scheduler not in valid_schedulers:
        scheduler = "karras" if "karras" in str(scheduler).lower() else "normal"

    nodes = {}
    nid = [1]

    def add(cls_name, inputs):
        i = nid[0]; nid[0] += 1
        nodes[str(i)] = {"class_type": cls_name, "inputs": inputs}
        return i

    # SEGURANCA (auto-correcao): FLUX/SD3/Anima NAO tem CLIP dentro do safetensors.
    # Se a familia vier errada do Civitai (labels inconsistentes), forca o branch correto
    # pelo CONTEUDO do arquivo. Sem isso, cairia no CheckpointLoaderSimple e o CLIPTextEncode
    # falharia com 'clip input is invalid: None'.
    _snf = _sniff_family_from_file(ckpt, "other")
    if _snf in ("flux_dev", "flux", "flux_schnell", "flux2_klein"):
        family = "flux_dev"
    elif _snf == "sd3":
        family = "sd3"
    elif _snf == "anima":
        family = "anima"

    if family in ("flux", "flux_dev", "flux_schnell", "flux2_klein"):
        # NOTA: a familia real e 'flux_dev'/'flux_schnell' (preset com comfy_template='flux');
        # comparar so com 'flux' faria o FLUX cair no else (CheckpointLoaderSimple) e o
        # CLIPTextEncode falhar com clip=None (flux nao tem CLIP dentro do safetensors).
        ckpt_name = os.path.basename(ckpt)
        unet = add("UNETLoader", {"unet_name": ckpt_name, "weight_dtype": "default"})
        clip = add("DualCLIPLoader", {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp8_e4m3fn.safetensors", "type": "flux"})
        vae = add("VAELoader", {"vae_name": "ae.safetensors"})
        pos = add("CLIPTextEncode", {"text": prompt, "clip": [str(clip), 0]})
        neg = add("CLIPTextEncode", {"text": negative, "clip": [str(clip), 0]})
        if init_image_path:
            img = add("LoadImage", {"image": os.path.basename(init_image_path)})
            latent = add("VAEEncode", {"pixels": [str(img), 0], "vae": [str(vae), 0]})
        else:
            latent = add("EmptySD3LatentImage", {"width": width, "height": height, "batch_size": 1})
        guidance = add("FluxGuidance", {"conditioning": [str(pos), 0], "guidance": cfg})
        ksample = add("KSampler", {
            "model": [str(unet), 0], "positive": [str(guidance), 0], "negative": [str(neg), 0],
            "latent_image": [str(latent), 0], "seed": seed, "steps": steps, "cfg": 1.0,
            "sampler_name": sampler, "scheduler": scheduler, "denoise": denoise})
        decoded = add("VAEDecode", {"samples": [str(ksample), 0], "vae": [str(vae), 0]})
    elif family in ("sd3", "sd35"):
        ckpt_name = os.path.basename(ckpt)
        unet = add("UNETLoader", {"unet_name": ckpt_name, "weight_dtype": "default"})
        clip = add("TripleCLIPLoader", {"clip_name1": "clip_l.safetensors", "clip_name2": "clip_g.safetensors", "clip_name3": "t5xxl_fp8_e4m3fn.safetensors", "type": "sd3"})
        vae = add("VAELoader", {"vae_name": "sd_vae.safetensors"})
        pos = add("CLIPTextEncode", {"text": prompt, "clip": [str(clip), 0]})
        neg = add("CLIPTextEncode", {"text": negative, "clip": [str(clip), 0]})
        latent = add("EmptySD3LatentImage", {"width": width, "height": height, "batch_size": 1})
        ksample = add("KSampler", {
            "model": [str(unet), 0], "positive": [str(pos), 0], "negative": [str(neg), 0],
            "latent_image": [str(latent), 0], "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": sampler, "scheduler": scheduler, "denoise": denoise})
        decoded = add("VAEDecode", {"samples": [str(ksample), 0], "vae": [str(vae), 0]})
    elif family == "anima":
        # Anima (CircleStone): UNET (DiT) + text encoder Qwen3-0.6B + VAE Wan21, latente 16ch
        # Workflow oficial do blueprint ComfyUI ('Text to Image (Anima).json'):
        #   UNETLoader weight_dtype='default' (fp16 NAO e mais valor valido no git main)
        #   CLIPLoader clip_name=<qwen3_06b> type='stable_diffusion' (entrada singular clip_name)
        ckpt_name = os.path.basename(ckpt)
        unet = add("UNETLoader", {"unet_name": ckpt_name, "weight_dtype": "default"})
        te_name = os.path.basename(STATE.get("anima_te_path") or "anima_te.safetensors")
        clip = add("CLIPLoader", {"clip_name": te_name, "type": "stable_diffusion"})
        vae = add("VAELoader", {"vae_name": "anima_vae.safetensors"})
        pos = add("CLIPTextEncode", {"text": prompt, "clip": [str(clip), 0]})
        neg = add("CLIPTextEncode", {"text": negative, "clip": [str(clip), 0]})
        latent = add("EmptyLatentImage", {"width": width, "height": height, "batch_size": 1})
        ksample = add("KSampler", {
            "model": [str(unet), 0], "positive": [str(pos), 0], "negative": [str(neg), 0],
            "latent_image": [str(latent), 0], "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": sampler, "scheduler": scheduler, "denoise": denoise})
        decoded = add("VAEDecode", {"samples": [str(ksample), 0], "vae": [str(vae), 0]})
    else:
        ckpt_name = os.path.basename(ckpt)
        ckpt_node = add("CheckpointLoaderSimple", {"ckpt_name": ckpt_name})
        model_out = [str(ckpt_node), 0]
        clip_out = [str(ckpt_node), 1]
        vae_out = [str(ckpt_node), 2]
        for (lora_path, w) in (loras or []):
            lora_node = add("LoraLoader", {"model": model_out, "clip": clip_out,
                                           "lora_name": os.path.basename(lora_path),
                                           "strength_model": w, "strength_clip": w})
            model_out = [str(lora_node), 0]
            clip_out = [str(lora_node), 1]
        pos = add("CLIPTextEncode", {"text": prompt, "clip": clip_out})
        neg = add("CLIPTextEncode", {"text": negative, "clip": clip_out})
        # ControlNet (P1-10)
        cn_model = None
        if controlnet_path:
            preset = FAMILY_PRESETS.get(family, {})
            cn_name = preset.get("controlnet") or "control_v11p_sd15_canny.pth"
            cn_node = add("ControlNetLoader", {"control_net_name": cn_name})
            cn_model = [str(cn_node), 0]
        if init_image_path and mask_path:
            img = add("LoadImage", {"image": os.path.basename(init_image_path)})
            enc = add("VAEEncode", {"pixels": [str(img), 0], "vae": vae_out})
            mask = add("LoadMask", {"mask": os.path.basename(mask_path)})
            latent = add("SetLatentNoiseMask", {"samples": [str(enc), 0], "mask": [str(mask), 0]})
        elif init_image_path:
            img = add("LoadImage", {"image": os.path.basename(init_image_path)})
            latent = add("VAEEncode", {"pixels": [str(img), 0], "vae": vae_out})
        else:
            latent = add("EmptyLatentImage", {"width": width, "height": height, "batch_size": 1})
        if cn_model:
            cn_img = add("LoadImage", {"image": os.path.basename(controlnet_path)})
            cn_app = add("ControlNetApplyAdvanced", {
                "positive": [str(pos), 0], "negative": [str(neg), 0],
                "control_net": cn_model, "image": [str(cn_img), 0],
                "strength": 0.8, "start_percent": 0.0, "end_percent": 1.0})
            pos_use = [str(cn_app), 0]
            neg_use = [str(cn_app), 1]
        else:
            pos_use = [str(pos), 0]
            neg_use = [str(neg), 0]
        ksample = add("KSampler", {
            "model": model_out, "positive": pos_use, "negative": neg_use,
            "latent_image": [str(latent), 0], "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": sampler, "scheduler": scheduler, "denoise": denoise})
        decoded = add("VAEDecode", {"samples": [str(ksample), 0], "vae": vae_out})
    save = add("SaveImage", {"images": [str(decoded), 0], "filename_prefix": "studio_out"})
    return nodes

def comfy_run(family, ckpt, prompt, negative, width, height, steps, cfg,
              sampler, scheduler, seed, denoise=1.0, loras=None, init_image_path=None,
              mask_path=None, controlnet_path=None, progress_cb=None):
    ensure_comfyui()
    # REDE DE SEGURANCA: metadata 'Other'/antigo mas arquivo FLUX (sem CLIP/VAE dentro —
    # CheckpointLoaderSimple devolveria clip=None e CLIPTextEncode falharia). Detectar por
    # conteudo ANTES do template/aux/symlink para baixar clip_l/t5xxl/ae e usar UNETLoader.
    # REDE DE SEGURANCA (robusta): detecta FLUX/SD3/Anima pelo CONTEUDO do arquivo,
    # ignorando a familia vinda do Civitai. FLUX/SD3 nao tem CLIP dentro do safetensors,
    # entao PRECISAM de UNETLoader + DualCLIPLoader/TripleCLIPLoader (senao CLIPTextEncode
    # falha com 'clip input is invalid: None').
    sniffed = _sniff_family_from_file(ckpt, "other")
    if sniffed in ("flux_dev", "flux", "flux_schnell", "flux2_klein"):
        print("  comfy_run: FLUX detectado por conteudo (meta '" + str(family) +
              "') — usando branch flux (UNETLoader + DualCLIP).")
        family = "flux_dev"
    elif sniffed == "sd3":
        print("  comfy_run: SD3 detectado por conteudo — usando branch sd3.")
        family = "sd3"
    elif sniffed == "anima":
        family = "anima"
    template = (FAMILY_PRESETS.get(family, {}) or {}).get("comfy_template", "checkpoint")
    if template == "checkpoint" and not _safetensors_valid(ckpt):
        print("  [check] checkpoint aparenta invalido em " + str(ckpt))
        _safetensors_valid(ckpt, verbose=True)
        mid = STATE.get("civitai_model_id")
        if mid:
            print("  [auto-reparo] rebaixando modelo " + str(mid) + " automaticamente...")
            try:
                new_local, _, _, _, _ = download_from_civitai(mid, STATE.get("civitai_token"), "Model", 0, None, True)
                STATE["model_path"] = new_local
                ckpt = new_local
                print("  [auto-reparo] re-download OK: " + os.path.basename(new_local))
            except Exception as e:
                print("  [auto-reparo] falhou: " + str(e)[:150])
        if not _safetensors_valid(ckpt):
            print("  [aviso] checkpoint ainda aparenta invalido — tentando mesmo assim (o ComfyUI mostrara o erro real se o arquivo estiver corrompido).")
    comfy_ensure_aux_files(template, family)
    # UNETLoader (flux/sd3/anima) le de models/diffusion_models; CheckpointLoaderSimple de models/checkpoints
    sub_dir = "diffusion_models" if template in ("flux", "sd3", "anima") else "checkpoints"
    ckpt_dest = os.path.join(COMFY_DIR, "models", sub_dir, os.path.basename(ckpt))
    if os.path.abspath(ckpt) != os.path.abspath(ckpt_dest):
        Path(os.path.dirname(ckpt_dest)).mkdir(parents=True, exist_ok=True)
        # CORRECAO: se o destino ja existe mas esta INVALIDO (copia antiga corrompida) ou nao
        # aponta para o arquivo original atual, RECRIA — o ComfyUI le o ckpt_dest, nao o original.
        _dest_ok = os.path.exists(ckpt_dest) and _safetensors_valid(ckpt_dest)
        _dest_points_original = False
        if os.path.exists(ckpt_dest):
            try:
                _dest_points_original = os.path.abspath(os.path.realpath(ckpt_dest)) == os.path.abspath(ckpt)
            except Exception:
                _dest_points_original = False
        if os.path.exists(ckpt_dest) and (not _dest_ok or not _dest_points_original):
            print("  [comfy] ckpt_dest invalido/desatualizado — recriando: " + os.path.basename(ckpt_dest))
            try:
                os.remove(ckpt_dest)
            except Exception:
                pass
        if not os.path.exists(ckpt_dest):
            try:
                os.symlink(ckpt, ckpt_dest)
            except Exception:
                shutil.copy(ckpt, ckpt_dest)
    for (lora_path, w) in (loras or []):
        lora_dest = os.path.join(COMFY_DIR, "models", "loras", os.path.basename(lora_path))
        Path(os.path.dirname(lora_dest)).mkdir(parents=True, exist_ok=True)
        if os.path.abspath(lora_path) != os.path.abspath(lora_dest) and not os.path.exists(lora_dest):
            try:
                os.symlink(lora_path, lora_dest)
            except Exception:
                shutil.copy(lora_path, lora_dest)
    for p in [init_image_path, mask_path, controlnet_path]:
        if p and os.path.abspath(p) != os.path.abspath(os.path.join(COMFY_DIR, "input", os.path.basename(p))):
            d2 = os.path.join(COMFY_DIR, "input", os.path.basename(p))
            Path(os.path.dirname(d2)).mkdir(parents=True, exist_ok=True)
            if not os.path.exists(d2):
                shutil.copy(p, d2)
    workflow = comfy_build_workflow(family, ckpt, prompt, negative, width, height, steps, cfg,
                                    sampler, scheduler, seed, denoise, loras, init_image_path,
                                    mask_path, controlnet_path)
    client_id = "studio-" + str(random.randint(0, 10 ** 9))
    r = requests.post(COMFY_BASE + "/prompt", json={"prompt": workflow, "client_id": client_id}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError("ComfyUI rejeitou workflow: " + r.text[:500])
    pid = r.json().get("prompt_id")
    if not pid:
        raise RuntimeError("Sem prompt_id do ComfyUI.")
    t0 = time.time()
    while time.time() - t0 < 2400:
        time.sleep(1.5)
        try:
            h = requests.get(COMFY_BASE + "/history/" + pid, timeout=10).json()
        except Exception:
            continue
        if pid in h:
            entry = h[pid]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = status.get("messages", [])
                detail = ""
                for ev in msgs:
                    if isinstance(ev, (list, tuple)) and len(ev) >= 2 and ev[0] == "execution_error":
                        info = ev[1] or {}
                        em = info.get("exception_message") or ""
                        tb = info.get("traceback") or []
                        detail = ("exception: " + str(em) + "\n" +
                                  "traceback:\n" + "\n".join(str(x) for x in tb[-14:])[:1600])
                        break
                log_tail = _read_log_tail(os.path.join(APP_DIR, "comfy.log"), n=30)
                raise RuntimeError("ComfyUI erro: " + str(msgs)[:400] + "\n\n" + detail +
                                   "\n\n--- tail comfy.log ---\n" + log_tail)
            outputs = entry.get("outputs", {})
            for node_id, out in outputs.items():
                for img in out.get("images", []):
                    fname = img.get("filename")
                    if fname:
                        img_r = requests.get(COMFY_BASE + "/view", params={"filename": fname, "subfolder": img.get("subfolder", ""), "type": img.get("type", "output")}, timeout=60)
                        img_r.raise_for_status()
                        from io import BytesIO
                        return Image.open(BytesIO(img_r.content)).convert("RGB"), seed
            if status.get("completed") is True:
                raise RuntimeError("ComfyUI terminou sem imagem.")
            continue
        try:
            q = requests.get(COMFY_BASE + "/queue", timeout=5).json()
            if q.get("queue_running") and progress_cb:
                progress_cb(0.5, 1.0, "ComfyUI gerando...")
        except Exception:
            pass
    raise RuntimeError("Timeout no ComfyUI (2400s).")

def comfy_upscale(image_path, upscale_by=4):
    ensure_comfyui()
    # copia a imagem para o diretorio input do ComfyUI
    in_dir = os.path.join(COMFY_DIR, "input")
    Path(in_dir).mkdir(parents=True, exist_ok=True)
    img_dest = os.path.join(in_dir, os.path.basename(image_path))
    if os.path.abspath(image_path) != os.path.abspath(img_dest) and not os.path.exists(img_dest):
        shutil.copy(image_path, img_dest)
    src = Image.open(image_path)
    w0, h0 = src.size
    tw = int(w0 * upscale_by); th = int(h0 * upscale_by)
    nodes = {}
    nid = [1]
    def add(cls_name, inputs):
        i = nid[0]; nid[0] += 1
        nodes[str(i)] = {"class_type": cls_name, "inputs": inputs}
        return i
    img = add("LoadImage", {"image": os.path.basename(image_path)})
    model = add("UpscaleModelLoader", {"model_name": "RealESRGAN_x4plus.pth"})
    up = add("ImageUpscaleWithModel", {"upscale_model": [str(model), 0], "image": [str(img), 0]})
    resized = add("ImageScale", {"image": [str(up), 0], "upscale_method": "lanczos", "width": tw, "height": th, "crop": "disabled"})
    save = add("SaveImage", {"images": [str(resized), 0], "filename_prefix": "studio_upscale"})
    client_id = "studio-" + str(random.randint(0, 10 ** 9))
    r = requests.post(COMFY_BASE + "/prompt", json={"prompt": nodes, "client_id": client_id}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError("ComfyUI upscale rejeitado: " + r.text[:300])
    pid = r.json().get("prompt_id")
    t0 = time.time()
    while time.time() - t0 < 600:
        time.sleep(1.5)
        try:
            h = requests.get(COMFY_BASE + "/history/" + pid, timeout=10).json()
        except Exception:
            continue
        if pid in h:
            for node_id, out in h[pid].get("outputs", {}).items():
                for imgf in out.get("images", []):
                    img_r = requests.get(COMFY_BASE + "/view", params={"filename": imgf.get("filename"), "type": imgf.get("type", "output")}, timeout=60)
                    from io import BytesIO
                    return Image.open(BytesIO(img_r.content)).convert("RGB")
    raise RuntimeError("Timeout no upscale.")

# ============================================================================
# 9. CORE DE GERACAO — compartilhado por UI e API (P1-6/7/8/11/12, P0-1)
# ============================================================================
def expand_matrix(prompt):
    """Prompt matrix (P1-11): 'a;b|b2;c|c2' -> todas as combinacoes."""
    if not prompt or (";" not in prompt and "|" not in prompt):
        return [prompt]
    segments = [s.strip() for s in prompt.split(";") if s.strip()]
    if not segments:
        return [prompt]
    options = [seg.split("|") for seg in segments]
    import itertools
    combos = [", ".join([opt.strip() for opt in combo]) for combo in itertools.product(*options)]
    return combos[:16]

def build_png_info(prompt, neg, steps, cfg, seed, w, h, model_label, family):
    info = str(prompt)
    if neg:
        info += "\nNegative prompt: " + str(neg)
    info += "\nSteps: " + str(steps) + ", CFG scale: " + str(cfg) + ", Seed: " + str(seed) + \
            ", Size: " + str(w) + "x" + str(h) + ", Model: " + str(model_label) + ", Family: " + str(family)
    return info

def save_image_pnginfo(img, path, info):
    from PIL import PngImagePlugin
    try:
        if img.mode not in ("RGB", "RGBA", "L", "P", "I"):
            img = img.convert("RGB")
    except Exception:
        pass
    pnginfo = PngImagePlugin.PngInfo()
    try:
        # PNG tEXt e latin-1: chars fora do latin-1 (emoji, etc) quebram o add_text
        safe = str(info).encode("latin-1", "replace").decode("latin-1")
        pnginfo.add_text("parameters", safe)
    except Exception:
        pass
    try:
        # format='PNG' OBRIGATORIO quando path e um BytesIO (a API salva em buffer):
        # sem ele, PIL nao infere extensao e lanca KeyError 'unknown file extension'.
        img.save(path, format="PNG", pnginfo=pnginfo)
    except Exception:
        # Nunca deixe o PNGInfo impedir o salvamento da imagem
        img.save(path, format="PNG")

def apply_templates(family, prompt, negative, use_template, use_trigger):
    preset = FAMILY_PRESETS.get(family, FAMILY_PRESETS["other"])
    p = prompt or ""
    n = negative or ""
    if use_template:
        p = (preset.get("prompt_prefix") or "") + p + (preset.get("prompt_suffix") or "")
        neg_prefix = preset.get("neg_prefix") or ""
        if neg_prefix:
            n = (neg_prefix + (", " + n if n else "")) if n else neg_prefix
    if use_trigger and STATE.get("trained_words"):
        tw = [t for t in STATE["trained_words"] if t and "<" not in t][:4]
        if tw:
            p = p + ", " + ", ".join(tw)
    return p, n

def _gen_diffusers(prompt, negative, steps, width, height, cfg, seed, family, init_image=None, mask_image=None, strength=1.0, hires=False, hires_denoise=0.45, hires_scale=2.0, progress_cb=None):
    pipe = STATE["pipe"]
    preset = FAMILY_PRESETS.get(family, {})
    if mask_image is not None:
        # Inpaint via from_pipe (P1-9); se falhar, erro claro sugerindo ComfyUI
        from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionInpaintPipeline
        cls = StableDiffusionXLInpaintPipeline if family in ("sdxl", "pony", "illustrious", "noobai", "animagine", "sdxl_lightning", "sdxl_lcm", "sdxl_hyper") else StableDiffusionInpaintPipeline
        ipipe = cls.from_pipe(pipe)
        gen = torch.Generator(device="cuda").manual_seed(int(seed))
        kwargs = {"prompt": prompt, "image": init_image, "mask_image": mask_image,
                  "num_inference_steps": int(steps), "strength": float(strength), "generator": gen}
        if negative:
            kwargs["negative_prompt"] = negative
        if preset.get("cfg", 0.0) != 0.0:
            kwargs["guidance_scale"] = float(cfg)
        return ipipe(**kwargs).images[0]

    def one_pass(pr, ne, ww, hh, ss, img_in=None, strn=1.0):
        gen = torch.Generator(device="cuda").manual_seed(int(seed))
        kwargs = {"prompt": pr, "num_inference_steps": int(ss), "width": int(ww), "height": int(hh), "generator": gen}
        if ne:
            kwargs["negative_prompt"] = ne
        if family.startswith("flux"):
            kwargs["guidance_scale"] = float(cfg) if cfg and cfg > 0 else 0.0
        else:
            kwargs["guidance_scale"] = float(cfg)
        rescale = preset.get("cfg_rescale", 0.0)
        if rescale and cfg and cfg > 1.0 and "sdxl" in family or rescale and family in ("pony", "illustrious", "noobai", "animagine"):
            kwargs["guidance_rescale"] = float(rescale)
        if img_in is not None:
            kwargs.pop("width", None)
            kwargs.pop("height", None)
            kwargs["image"] = img_in
            kwargs["strength"] = float(strn)
        import inspect
        try:
            sig = inspect.signature(pipe.__call__)
            for k in list(kwargs.keys()):
                if k not in sig.parameters and "kwargs" not in str(sig):
                    kwargs.pop(k)
        except Exception:
            pass
        if progress_cb:
            try:
                if "callback_on_step_end" in inspect.signature(pipe.__call__).parameters:
                    def step_cb(pipe_obj, step_index, timestep, callback_kwargs):
                        progress_cb(step_index, int(ss), "Passo " + str(step_index) + "/" + str(ss))
                        return callback_kwargs
                    kwargs["callback_on_step_end"] = step_cb
            except Exception:
                pass
        return pipe(**kwargs).images[0]

    if hires:
        w1 = max(256, round(width / hires_scale / 64) * 64)
        h1 = max(256, round(height / hires_scale / 64) * 64)
        img1 = one_pass(prompt, negative, w1, h1, max(8, int(steps * 0.6)))
        img2 = one_pass(prompt, negative, width, height, int(steps * 0.6), img_in=img1, strn=hires_denoise)
        return img2
    return one_pass(prompt, negative, width, height, steps)

def _gen_wan2gp(prompt, negative, steps, width, height, cfg, seed, progress_cb=None):
    # Worker isolado: geracao via proxy HTTP (OOM mata o worker, nao a sessao)
    if STATE.get("krea_worker"):
        return _proxy_worker_generate(prompt, negative, steps, width, height, cfg, seed, progress_cb)
    raise RuntimeError("Krea-2 nao carregado (use o botao Krea-2-Turbo na aba Modelo).")

def _mk_cb(cb):
    """Adapta QUALQUER callback para a convencao (done, total, desc) — nunca quebra por aridade nem kwargs."""
    if not cb:
        return None
    def _w(*a, **kwargs):
        try:
            done = a[0] if len(a) > 0 else 0
            total = a[1] if len(a) > 1 else 1
            desc = a[2] if len(a) > 2 else kwargs.get("desc", "")
            cb(done, total, desc)
        except Exception:
            pass
    return _w

def gen_single(family, prompt, negative, steps, width, height, cfg, seed, init_image=None,
               mask_image=None, strength=1.0, hires=False, hires_denoise=0.45, hires_scale=2.0, progress_cb=None):
    """Gera UMA imagem no backend atual; inpaint/controlnet roteiam para ComfyUI."""
    progress_cb = _mk_cb(progress_cb)
    backend = STATE.get("backend")
    preset = FAMILY_PRESETS.get(family, {})
    # Upscaler: aplica em imagem enviada via ComfyUI (RealESRGAN)
    if family == "upscaler":
        if init_image is None:
            raise RuntimeError("Upscaler requer uma imagem de entrada (Img2Img).")
        scale = max(int(width), int(height))
        if scale < 2:
            scale = 4
        p = os.path.join(APP_DIR, "up_tmp_" + str(seed) + ".png")
        if isinstance(init_image, Image.Image):
            init_image.save(p)
        else:
            Image.fromarray(init_image).save(p)
        try:
            if progress_cb:
                progress_cb(0.5, 1.0, "Upscaling x" + str(scale) + "...")
            return comfy_upscale(p, upscale_by=scale)
        finally:
            try:
                os.remove(p)
            except Exception:
                pass
    # Inpaint e ControlNet: rota ComfyUI universal (P1-9/10)
    if (mask_image is not None or (STATE.get("ctrl_image") and backend != "diffusers")) or (mask_image is not None and backend == "diffusers" and family in ("flux_dev", "flux_schnell", "sd3", "sd35", "anima")):
        ensure_comfyui()
        if STATE.get("model_path") is None:
            raise RuntimeError("Inpaint/ControlNet requerem checkpoint em disco (use ComfyUI).")
        init_path = None
        if init_image is not None:
            init_path = os.path.join(APP_DIR, "init_tmp_" + str(seed) + ".png")
            if isinstance(init_image, Image.Image):
                init_image.save(init_path)
            else:
                Image.fromarray(init_image).save(init_path)
        mask_path = None
        if mask_image is not None:
            mask_path = os.path.join(APP_DIR, "mask_tmp_" + str(seed) + ".png")
            if isinstance(mask_image, Image.Image):
                mask_image.save(mask_path)
            else:
                Image.fromarray(mask_image).save(mask_path)
        ctrl_path = None
        ctrl = STATE.get("ctrl_image")
        if ctrl is not None:
            ctrl_path = os.path.join(APP_DIR, "ctrl_tmp_" + str(seed) + ".png")
            if isinstance(ctrl, Image.Image):
                ctrl.save(ctrl_path)
            else:
                Image.fromarray(ctrl).save(ctrl_path)
        img, _ = comfy_run(family, STATE["model_path"], prompt, negative, width, height, steps, cfg,
                           preset.get("sampler", "dpmpp_2m"),
                           "simple" if preset.get("scheduler") == "simple" else ("karras" if "Karras" in preset.get("scheduler", "") else "normal"),
                           seed, strength, STATE.get("lora_stack"), init_path, mask_path, ctrl_path,
                           progress_cb=lambda a, b, d: progress_cb(a, b, d) if progress_cb else None)
        _cleanup_tmp()
        return img
    if backend == "diffusers":
        return _gen_diffusers(prompt, negative, steps, width, height, cfg, seed, family,
                              init_image, mask_image, strength, hires, hires_denoise, hires_scale, progress_cb)
    if backend == "wan2gp":
        if init_image is not None:
            raise RuntimeError("Krea-2-Turbo nao suporta img2img diretamente.")
        return _gen_wan2gp(prompt, negative, steps, width, height, cfg, seed, progress_cb)
    if backend == "comfy":
        init_path = None
        if init_image is not None:
            init_path = os.path.join(APP_DIR, "init_tmp_" + str(seed) + ".png")
            if isinstance(init_image, Image.Image):
                init_image.save(init_path)
            else:
                Image.fromarray(init_image).save(init_path)
        if hires:
            w1 = max(256, round(width / hires_scale / 64) * 64)
            h1 = max(256, round(height / hires_scale / 64) * 64)
            img1, _ = comfy_run(family, STATE["model_path"], prompt, negative, w1, h1,
                                max(8, int(steps * 0.6)), cfg,
                                preset.get("sampler", "dpmpp_2m"),
                                "simple" if preset.get("scheduler") == "simple" else ("karras" if "Karras" in preset.get("scheduler", "") else "normal"),
                                seed, 1.0, STATE.get("lora_stack"), None, None, None)
            p1 = os.path.join(APP_DIR, "hires_tmp_" + str(seed) + ".png")
            img1.save(p1)
            img2, _ = comfy_run(family, STATE["model_path"], prompt, negative, width, height,
                                int(steps * 0.6), cfg,
                                preset.get("sampler", "dpmpp_2m"),
                                "simple" if preset.get("scheduler") == "simple" else ("karras" if "Karras" in preset.get("scheduler", "") else "normal"),
                                seed + 1, hires_denoise, STATE.get("lora_stack"), p1, None, None)
            try:
                os.remove(p1)
            except Exception:
                pass
            return img2
        img, _ = comfy_run(family, STATE["model_path"], prompt, negative, width, height, steps, cfg,
                           preset.get("sampler", "dpmpp_2m"),
                           "simple" if preset.get("scheduler") == "simple" else ("karras" if "Karras" in preset.get("scheduler", "") else "normal"),
                           seed, 1.0, STATE.get("lora_stack"), init_path, None, None,
                           progress_cb=lambda a, b, d: progress_cb(a, b, d) if progress_cb else None)
        _cleanup_tmp()
        return img
    raise RuntimeError("Nenhum backend carregado.")

def run_generation(params, progress_cb=None):
    """Core de geracao compartilhado por UI e API. params: dict com tudo."""
    with GEN_LOCK:
        progress_cb = _mk_cb(progress_cb)
        if params.get("model_url"):
            with LOAD_LOCK:
                if STATE.get("last_loaded_url") != params["model_url"]:
                    st, fam = load_model_from_civitai(params["model_url"], params.get("civitai_token"),
                                                      params.get("hf_token"), 0, bool(params.get("force_comfy")),
                                                      progress_cb=progress_cb)
                    if fam is None:
                        raise RuntimeError(st)
        if not STATE.get("loaded"):
            raise RuntimeError("Nenhum modelo carregado.")
        family = STATE.get("family", "other")
        preset = FAMILY_PRESETS.get(family, FAMILY_PRESETS["other"])
        prompt = params.get("prompt", "")
        negative = params.get("negative_prompt", "")
        use_template = bool(params.get("use_template", True))
        use_trigger = bool(params.get("use_trigger", True))
        p_prompt, p_neg = apply_templates(family, prompt, negative, use_template, use_trigger)
        width = int(params.get("width", 1024))
        height = int(params.get("height", 1024))
        base_px = max(width, height)
        base_px = clamp_resolution(base_px, family)
        if max(width, height) > base_px:
            scale = base_px / max(width, height)
            width = max(64, round(width * scale / 64) * 64)
            height = max(64, round(height * scale / 64) * 64)
        steps = int(params.get("steps", preset.get("steps", {}).get("default", 25)))
        cfg = float(params.get("cfg", preset.get("cfg", 0.0)))
        init_seed = random.randint(0, 2 ** 32 - 1) if (params.get("seed") is None or params.get("seed") < 0) else int(params["seed"])
        num_images = max(1, int(params.get("num_images", 1)))
        hires = bool(params.get("hires_fix", False))
        hires_denoise = float(params.get("hires_denoise", 0.45))
        matrix = bool(params.get("prompt_matrix", False))
        combos = expand_matrix(p_prompt) if matrix else [p_prompt]
        results = []
        model_label = (STATE.get("config") or {}).get("label", "model")
        hires_scale = float(params.get("hires_scale", 2.0))
        total = len(combos) * num_images
        idx = 0
        for ci, combo in enumerate(combos):
            for i in range(num_images):
                s = (init_seed + idx) % (2 ** 32)
                idx += 1
                if progress_cb:
                    progress_cb(idx, total, "Gerando " + str(idx) + "/" + str(total) + " (seed " + str(s) + ")")
                img = gen_single(family, combo, p_neg, steps, width, height, cfg, s,
                                 init_image=params.get("init_image"), mask_image=params.get("mask_image"),
                                 strength=float(params.get("strength", 1.0)),
                                 hires=hires, hires_denoise=hires_denoise, hires_scale=hires_scale,
                                 progress_cb=progress_cb)
                info = build_png_info(combo, p_neg, steps, cfg, s, width, height, model_label, family)
                results.append((img, s, info, width, height))
                gc.collect(); torch.cuda.empty_cache()
        return results

# ============================================================================
# 10. LORA / VAE / TEXTUAL INVERSION (Civitai) — P3-24
# ============================================================================
LORA_EXAMPLES = [
    ("Detail Tweaker XL (SDXL)", "https://civitai.com/models/122359"),
    ("Add More Details (SD1.5/SDXL)", "https://civitai.com/models/82098"),
    ("Detail Tweaker LoRA", "https://civitai.com/models/58390"),
    ("Not Artists Styles for Pony", "https://civitai.com/models/264290"),
    ("blindbox", "https://civitai.com/models/25995"),
    ("Hands XL + SD1.5 + Pony + F1D", "https://civitai.com/models/200255"),
    ("Doll Likeness - EDG", "https://civitai.com/models/42903"),
    ("SXZ Will Murai BlizzCon Keyart (Krea2)", "https://civitai.com/models/2846061"),
    ("Dmitry Prozorov TamplierPainter Artist", "https://civitai.com/models/1276570"),
    ("SXZ Warcraft Animated Short (Krea2)", "https://civitai.com/models/2828764"),
    ("SXZ Bayard Wu (Krea2)", "https://civitai.com/models/2828610"),
    ("World of Warcraft", "https://civitai.com/models/693378"),
    ("Satoshi Urushihara Style", "https://civitai.com/models/7227"),
    ("Comic Book Page Style (multi-base)", "https://civitai.com/models/462611"),
    ("Hades 2 Isometric Map", "https://civitai.com/models/1677773"),
    ("Tensura (Slime) Anime Style", "https://civitai.com/models/430181"),
    ("RimWorld Art Style", "https://civitai.com/models/411781"),
    ("Gag RPG Potions LoRA XL", "https://civitai.com/models/22591"),
    ("MohawkAddon Comics", "https://civitai.com/models/148394"),
    ("Game Icon InstituteANFGV3", "https://civitai.com/models/143301"),
    ("Fake Books Cthulhu Mythos", "https://civitai.com/models/139226"),
    ("Tensura Style LoRA", "https://civitai.com/models/29829"),
    ("Hades Style", "https://civitai.com/models/26681"),
    ("PrintableHeroes TTRPG PaperMinis", "https://civitai.com/models/15535"),
    ("Painted Miniature", "https://civitai.com/models/7718"),
]
VAE_EXAMPLES = [
    ("vae-ft-mse-840000 (SD1.5)", "https://civitai.com/models/276082"),
    ("SDXL VAE", "https://civitai.com/models/296576"),
    ("sdxl-vae-fp16-fix (menos VRAM)", "https://civitai.com/models/140686"),
    ("XL VAE C", "https://civitai.com/models/152040"),
    ("VAE-kl-f8-anime2", "https://civitai.com/models/23906"),
    ("ClearVAE (SD1.5)", "https://civitai.com/models/22354"),
    ("Flux VAE", "https://civitai.com/models/636193"),
]
TI_EXAMPLES = [
    ("EasyNegative", "https://civitai.com/models/7808"),
    ("Deep Negative V1.x", "https://civitai.com/models/4629"),
    ("negative_hand", "https://civitai.com/models/56519"),
    ("badhandv4", "https://civitai.com/models/16993"),
    ("veryBadImageNegative", "https://civitai.com/models/11772"),
    ("BadDream + UnrealisticDream", "https://civitai.com/models/72437"),
]
AUX_KIND_DIR = {"lora": LORA_DIR, "vae": VAE_DIR, "ti": TI_DIR}


def aux_query(url, token, wanted_type):
    """Consulta modelo LoRA/VAE/TI no Civitai.
    Retorna (card_md, choices_versoes, lista_versoes, msg)."""
    model_id = parse_civitai_id(url)
    if not model_id:
        raise RuntimeError("URL/ID invalido.")
    data = civitai_get_model(model_id, token)
    versions = civitai_sorted_versions(data)
    choices = []
    for i, v in enumerate(versions):
        total = sum((f.get("sizeKB") or 0) for f in v.get("files", [])) * 1024
        choices.append(str(i) + " | " + str(v.get("name", "?")) + " | base: " + str(v.get("baseModel")) + " | " + fmt_bytes(total))
    tw = ", ".join((versions[0].get("trainedWords") or [])[:6]) if versions else ""
    card = ("## " + str(data.get("name", "?")) + " (" + str(wanted_type) + ")\n\n"
            + "**Tipo:** " + str(wanted_type) + " | **Versoes:** " + str(len(versions)) + "\n"
            + "**Trigger words (ultima versao):** `" + tw + "`\n\n"
            + str(data.get("description") or "")[:400])
    return card, choices, versions, ("Consultado: " + str(data.get("name")) + " | " + str(len(versions)) + " versoes.")


def aux_download(url, token, wanted_type, version_index=0, progress_cb=None):
    """Baixa um LoRA/VAE/TI (versao especifica; 0 = mais recente). Retorna (local, base, name, family, tw)."""
    model_id = parse_civitai_id(url)
    if not model_id:
        raise RuntimeError("URL/ID invalido.")
    idx = int(version_index or 0)
    return download_from_civitai(model_id, token, wanted_type, idx, progress_cb, use_latest=(idx == 0))


def list_aux_by_kind(kind):
    """Lista arquivos baixados de um tipo (lora/vae/ti) -> [(path, label)]."""
    d = AUX_KIND_DIR.get(str(kind).lower())
    if not d or not os.path.isdir(d):
        return []
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith((".safetensors", ".pt", ".bin", ".ckpt")):
            label = f
            meta = os.path.join(d, f + ".meta.json")
            if os.path.exists(meta):
                try:
                    md = json.load(open(meta, encoding="utf-8"))
                    label = f + " | " + str(md.get("model_name", ""))[:42]
                except Exception:
                    pass
            out.append((os.path.join(d, f), label))
    return out


def delete_aux_file(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
            for mp in (path + ".meta.json",):
                if os.path.exists(mp):
                    os.remove(mp)
            return "Deletado: " + os.path.basename(path)
        return "Arquivo nao encontrado."
    except Exception as e:
        return "Erro ao deletar: " + str(e)


def lora_remove_one(path):
    STATE["lora_stack"] = [(p, w) for (p, w) in STATE["lora_stack"] if p != path]
    return "LoRA removido da pilha: " + os.path.basename(path)




def apply_lora_local(path, weight=1.0):
    """Aplica um LoRA JA baixado (biblioteca) — sem re-download."""
    try:
        if not path or not os.path.exists(path):
            return "Arquivo nao encontrado: " + str(path)
        STATE["lora_stack"] = [(p, w) for (p, w) in STATE["lora_stack"] if p != path]
        STATE["lora_stack"].append((path, float(weight)))
        STATE["lora_scale"] = float(weight)
        if STATE.get("backend") == "diffusers" and STATE.get("pipe") is not None:
            _ensure_torchao()
            try:
                STATE["pipe"].load_lora_weights(path)
                try:
                    STATE["pipe"].set_adapters(["default"], adapter_weights=[float(weight)])
                except Exception:
                    pass
                return "LoRA aplicado da biblioteca: " + os.path.basename(path) + " (peso " + str(weight) + ")"
            except Exception as e:
                STATE["lora_stack"] = [(p, w) for (p, w) in STATE["lora_stack"] if p != path]
                return "Falha ao aplicar no pipe (usa ComfyUI no load): " + str(e)[:150]
        return "LoRA adicionado a pilha (backend " + str(STATE.get("backend")) + "): " + os.path.basename(path)
    except Exception as e:
        return "Erro: " + str(e)


def apply_vae_local(path):
    """Aplica um VAE JA baixado (biblioteca) ao modelo atual."""
    try:
        if not path or not os.path.exists(path):
            return "Arquivo nao encontrado: " + str(path)
        if STATE.get("backend") == "diffusers" and STATE.get("pipe") is not None:
            from diffusers import AutoencoderKL
            vae = AutoencoderKL.from_single_file(path, torch_dtype=torch.float16)
            STATE["pipe"].vae = vae
            try:
                STATE["pipe"].vae.enable_tiling()
                STATE["pipe"].vae.enable_slicing()
            except Exception:
                pass
            STATE["active_vae"] = os.path.basename(path)
            return "VAE aplicado da biblioteca: " + os.path.basename(path)
        return "VAE baixado (backend " + str(STATE.get("backend")) + "): " + os.path.basename(path)
    except Exception as e:
        return "Erro VAE: " + str(e)


def load_ti_local(path):
    """Registra um TI JA baixado (biblioteca)."""
    try:
        if not path or not os.path.exists(path):
            return "Arquivo nao encontrado: " + str(path)
        token_name = Path(path).stem
        if STATE.get("backend") == "diffusers" and STATE.get("pipe") is not None:
            try:
                STATE["pipe"].load_textual_inversion(path)
                STATE.setdefault("active_ti", [])
                if token_name not in STATE["active_ti"]:
                    STATE["active_ti"].append(token_name)
                return "TI carregado da biblioteca: " + os.path.basename(path) + " | use <" + token_name + "> no prompt"
            except Exception as e:
                return "Falha ao registrar no pipe: " + str(e)[:150]
        d2 = os.path.join(COMFY_DIR, "models", "embeddings")
        Path(d2).mkdir(parents=True, exist_ok=True)
        shutil.copy(path, os.path.join(d2, os.path.basename(path)))
        return "TI copiado p/ ComfyUI: " + os.path.basename(path) + " | use embedding:" + token_name
    except Exception as e:
        return "Erro TI: " + str(e)




def _ensure_torchao():
    """Garante torchao >= 0.16 (diffusers 0.36 exige p/ aplicar LoRA no pipe)."""
    try:
        import torchao
        _v = [int(x) for x in str(getattr(torchao, "__version__", "0")).split(".")[:2]]
        if _v and _v[0] == 0 and _v[1] < 16:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torchao>=0.16.0"], timeout=600)
    except Exception:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torchao>=0.16.0"], timeout=600)
        except Exception:
            pass
def components_status():
    """Estado integrado dos componentes (LoRA + VAE + TI) para a proxima geracao."""
    parts = []
    ls = STATE.get("lora_stack") or []
    if ls:
        parts.append("LoRAs(" + str(len(ls)) + "): " + ", ".join(os.path.basename(p)[:22] + " w=" + str(w) for (p, w) in ls))
    else:
        parts.append("LoRAs: nenhum")
    parts.append("VAE: " + str(STATE.get("active_vae") or "(padrao do modelo)"))
    tis = STATE.get("active_ti") or []
    parts.append("TI(" + str(len(tis)) + "): " + (", ".join(str(x) for x in tis) if tis else "nenhum"))
    return " | ".join(parts)
def lora_active_choices():
    return [os.path.basename(p) + " (w=" + str(w) + ")" for (p, w) in (STATE.get("lora_stack") or [])]


def load_lora_from_civitai(url_or_id, token, weight=1.0, version_index=0):
    """Baixa e ATIVA o LoRA (versao especifica; 0 = mais recente)."""
    try:
        model_id = parse_civitai_id(url_or_id)
        if not model_id:
            return "URL/ID de LoRA invalido."
        idx = int(version_index or 0)
        local, base, name, family, tw = download_from_civitai(model_id, token, "LoRA", idx, None, use_latest=(idx == 0))
        STATE["lora_stack"].append((local, float(weight)))
        STATE["lora_scale"] = float(weight)
        if STATE.get("backend") == "diffusers" and STATE.get("pipe") is not None:
            _ensure_torchao()
            try:
                STATE["pipe"].load_lora_weights(local)
                try:
                    STATE["pipe"].set_adapters(["default"], adapter_weights=[float(weight)])
                except Exception:
                    pass
                return "LoRA ativado: " + name + " (peso " + str(weight) + ")"
            except Exception as e:
                STATE["lora_stack"] = [(p, w) for (p, w) in STATE["lora_stack"] if p != local]
                return "LoRA baixado mas falha ao aplicar (usa ComfyUI): " + str(e)[:200]
        return "LoRA adicionado a pilha (backend " + str(STATE.get("backend")) + "): " + name
    except Exception as e:
        traceback.print_exc()
        return "Erro LoRA: " + str(e)

def load_vae_from_civitai(url_or_id, token, version_index=0):
    try:
        model_id = parse_civitai_id(url_or_id)
        if not model_id:
            return "URL/ID de VAE invalido."
        idx = int(version_index or 0)
        local, base, name, family, tw = download_from_civitai(model_id, token, "VAE", idx, None, use_latest=(idx == 0))
        if STATE.get("backend") == "diffusers" and STATE.get("pipe") is not None:
            from diffusers import AutoencoderKL
            try:
                vae = AutoencoderKL.from_single_file(local, torch_dtype=torch.float16)
                STATE["pipe"].vae = vae
                try:
                    STATE["pipe"].vae.enable_tiling(); STATE["pipe"].vae.enable_slicing()
                except Exception:
                    pass
                STATE["active_vae"] = name
                return "VAE aplicado: " + name
            except Exception as e:
                return "VAE baixado mas falha ao aplicar: " + str(e)[:200]
        return "VAE baixado: " + local
    except Exception as e:
        traceback.print_exc()
        return "Erro VAE: " + str(e)

def load_ti_from_civitai(url_or_id, token, version_index=0):
    """TextualInversion (P3-24): baixa e registra; uso via <nome> ou embedding:nome."""
    try:
        model_id = parse_civitai_id(url_or_id)
        if not model_id:
            return "URL/ID de TextualInversion invalido."
        idx = int(version_index or 0)
        local, base, name, family, tw = download_from_civitai(model_id, token, "TextualInversion", idx, None, use_latest=(idx == 0))
        token_name = Path(local).stem
        if STATE.get("backend") == "diffusers" and STATE.get("pipe") is not None:
            try:
                STATE["pipe"].load_textual_inversion(local)
                STATE.setdefault("active_ti", [])
                if token_name not in STATE["active_ti"]:
                    STATE["active_ti"].append(token_name)
                return "TI carregado: " + name + " | use <" + token_name + "> no prompt"
            except Exception as e:
                return "TI baixado mas falha ao registrar: " + str(e)[:200]
        d2 = os.path.join(COMFY_DIR, "models", "embeddings")
        Path(d2).mkdir(parents=True, exist_ok=True)
        shutil.copy(local, os.path.join(d2, os.path.basename(local)))
        return "TI copiado p/ ComfyUI: " + name + " | use embedding:" + token_name + " no prompt"
    except Exception as e:
        traceback.print_exc()
        return "Erro TI: " + str(e)

def clear_lora():
    try:
        if STATE.get("backend") == "diffusers" and STATE.get("pipe") is not None:
            try:
                STATE["pipe"].unload_lora_weights()
            except Exception:
                pass
        STATE["lora_stack"] = []
        STATE["lora_scale"] = 1.0
        return "LoRAs removidos."
    except Exception as e:
        return "Erro: " + str(e)

# ============================================================================
# 11. ORQUESTRADOR DE CARGA
# ============================================================================
def unload_current_model():
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
    STATE["krea_worker"] = None
    STATE["active_vae"] = None
    STATE["active_ti"] = []
    STATE["config"] = None
    STATE["backend"] = None
    STATE["loaded"] = False
    STATE["family"] = None
    STATE["lora_stack"] = []
    STATE["lora_scale"] = 1.0
    STATE["trained_words"] = []
    gc.collect()
    torch.cuda.empty_cache()

def load_model_from_civitai(url_or_id, token, hf_token, version_index=0, force_comfy=False, progress_cb=None):
    """Fluxo definitivo: consulta -> versao mais recente -> arquivo recomendado -> motor."""
    try:
        model_id = parse_civitai_id(url_or_id)
        if not model_id:
            return "URL/ID invalido.", None
        STATE["civitai_model_id"] = model_id
        STATE["civitai_wanted_type"] = "Model"
        if hf_token:
            STATE["hf_token"] = str(hf_token).strip()
        if token:
            STATE["civitai_token"] = str(token).strip()
        if progress_cb:
            progress_cb(0, 1, "Consultando API do Civitai...")
        local, base_model, model_name, family, trained_words = download_from_civitai(
            model_id, STATE.get("civitai_token"), "Model", version_index, progress_cb, use_latest=(version_index == 0))
        # FALLBACK: baseModel 'Other'/desconhecido mas arquivo FLUX/SDXL — sniff pelo conteudo
        sniffed = _sniff_family_from_file(local, family)
        if sniffed != family:
            print("  Metadata '" + str(family) + "' — detectado por conteudo: " + sniffed)
            family = sniffed

        # AUTO-ROUTING INTELIGENTE: ativa ComfyUI e quantizacao automaticamente para modelos complexos
        if family in ("flux", "flux_dev", "flux_schnell", "flux2_klein", "sd3", "sd35", "anima", "chroma", "auraflow", "hunyuan", "lumina", "ernie"):
            force_comfy = True
            if not STATE.get("quant_method") or STATE.get("quant_method") == "none":
                STATE["quant_method"] = "torchao_int8"
            print("  Auto-routing Inteligente: ativado ComfyUI + quantizacao (" + STATE["quant_method"] + ") para " + family)
        preset = FAMILY_PRESETS.get(family, FAMILY_PRESETS["other"])
        print("  Base: " + str(base_model) + " | familia: " + family + " | " + os.path.basename(local))
        unload_current_model()
        backend_used = None
        error_log = []
        if family in ("grok",):
            return "Grok (xAI) e hospedado — sem pesos abertos.", None
        if family in ("wan_video",):
            return "Wan Video gera video, nao imagem.", None
        if not force_comfy and preset.get("single_file") and preset.get("diffusers_cls"):
            try:
                load_diffusers_single_file(local, base_model, family, {"dtype": "float16", "cpu_offload": "model"})
                backend_used = "diffusers"
            except Exception as e:
                error_log.append("diffusers: " + str(e)[:200])
                unload_current_model()
        if any(k in str(base_model).lower() for k in ["krea 2", "krea2", "krea-2"]):
            # EVOLUCAO v2.5: roteia checkpoints Krea-2 customizados para o fluxo
            # worker com selecao inteligente de arquivo (int8/fp8, nunca bf16 23.9GB).
            if progress_cb:
                progress_cb(0.01, 1.0, "Roteando Krea-2 custom (selecao int8/fp8)...")
            return load_krea_custom(url_or_id, progress_cb=progress_cb)

        if backend_used is None:
            if family in ("hidream",):
                return "HiDream (17B) excede a T4. Erros: " + " | ".join(error_log), None
            try:
                if progress_cb:
                    progress_cb(0.2, 1, "Usando motor universal ComfyUI...")
                ensure_comfyui()
                STATE["config"] = {"label": model_name, "backend": "comfy"}
                STATE["model_path"] = local
                STATE["backend"] = "comfy"
                STATE["family"] = family
                STATE["loaded"] = True
                backend_used = "comfy"
            except Exception as e:
                error_log.append("comfy: " + str(e)[:200])
                unload_current_model()
        if backend_used is None:
            return "Falha em todos os motores: " + " | ".join(error_log), None
        STATE["config"] = {"label": model_name, "backend": backend_used, "family": family, "base_model": base_model}
        STATE["trained_words"] = list(trained_words or [])
        STATE["model_name"] = model_name
        STATE["model_path"] = local
        STATE["last_loaded_url"] = url_or_id
        if progress_cb:
            progress_cb(1, 1, "Pronto!")
        tw_line = (" | Trigger words: " + ", ".join(STATE["trained_words"][:4])) if STATE["trained_words"] else ""
        return ("Modelo carregado: " + model_name + "\nBase: " + str(base_model) + "\nFamilia: " + family +
                "\nMotor: " + backend_used + "\nArquivo: " + os.path.basename(local) + tw_line), family
    except Exception as e:
        traceback.print_exc()
        return "Erro ao carregar do Civitai: " + str(e), None

def _ensure_krea_te_vae(progress_cb=None):
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
                "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                "preprocessor_config.json"]
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


def download_krea_official(progress_cb=None):
    """Baixa o Krea-2-Turbo oficial INT8 (DeepBeepMeep/krea-2) — hf_hub_download com resume."""
    from huggingface_hub import hf_hub_download
    repo = "DeepBeepMeep/krea-2"
    qwen_repo = "DeepBeepMeep/Qwen_image"
    if not os.path.exists(WAN2GP_DIR):
        raise RuntimeError("Wan2GP nao clonado (chame load_krea para setup completo).")
    model_dir = os.path.join(WAN2GP_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    _ensure_krea_te_vae(progress_cb=progress_cb)
    tf_path = os.path.join(model_dir, "Krea2Turbo_quanto_bf16_int8.safetensors")
    if not os.path.exists(tf_path):
        if progress_cb:
            progress_cb(0.15, 1.0, "Baixando Krea2-Turbo Transformer INT8 (12.5GB)...")
        hf_hub_download(repo_id=repo, filename="Krea2Turbo_quanto_bf16_int8.safetensors",
                        local_dir=model_dir, local_dir_use_symlinks=False)
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
        return "Erro ao carregar Krea-2-Turbo: " + str(e), None

def _pick_krea_file(version):
    """Escolhe o melhor arquivo Krea-2 para a T4 por FORMATO (nao por tamanho):
    prioridade int8 > fp8/fp8_mixed; bf16 (>20GB) recusado com aviso claro.
    resolve o OOM: o picker antigo escolhia o primary bf16 de 23.9GB."""
    files = [f for f in version.get("files", [])
             if (f.get("type") or "") in ("Model", "Diffusion Model")
             and str(f.get("name") or "").endswith(".safetensors")]
    if not files:
        files = [f for f in version.get("files", []) if str(f.get("name") or "").endswith(".safetensors")]
    if not files:
        raise RuntimeError("Nenhum arquivo safetensors nessa versao Krea-2.")
    def rank(f):
        fmt = str(((f.get("metadata") or {}).get("fp")) or "").lower()
        size = f.get("sizeKB") or 0
        if "int8" in fmt:
            return (0, size)
        if "fp8" in fmt:
            return (1, size)
        if "bf16" in fmt or "bfloat16" in fmt:
            return (3, size)
        return (2, size)
    best = sorted(files, key=rank)[0]
    fmt = str(((best.get("metadata") or {}).get("fp")) or "").lower()
    size_gb = (best.get("sizeKB") or 0) / 1024 / 1024
    if "bf16" in fmt or "bfloat16" in fmt or size_gb > 20:
        raise RuntimeError(
            "A versao disponivel e BF16 de " + "%.1f" % size_gb +
            "GB — impossivel na T4 (12GB RAM / 16GB VRAM). Procure uma versao fp8 ou int8 (Q8) do modelo.")
    return best


def load_krea_custom(url_or_id, progress_cb=None):
    """Carrega um checkpoint Krea-2 CUSTOMIZADO do Civitai na T4:
    1) consulta API e escolhe o arquivo certo (int8/fp8, nunca bf16);
    2) baixa; 3) sobe o worker isolado (quantizeTransformer=False)."""
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU indisponivel — use Runtime > T4 GPU.")
        model_id = parse_civitai_id(url_or_id)
        if not model_id:
            return "URL/ID invalido.", None
        unload_current_model()
        data = civitai_get_model(model_id, STATE.get("civitai_token"))
        model_name = data.get("name", "Krea-2 custom")
        versions = civitai_sorted_versions(data)
        if not versions:
            return "Modelo sem versoes com arquivos.", None
        # escolhe versao (latest) e arquivo por formato
        version = versions[0]
        target = _pick_krea_file(version)
        download_url = target.get("downloadUrl")
        if not download_url:
            return "URL de download ausente nessa versao.", None
        file_name = target.get("name", "krea2_custom.safetensors")
        safe_name = re.sub(r"[^\w\-.]", "_", file_name)
        os.makedirs(CIVITAI_DIR, exist_ok=True)
        local_path = os.path.join(CIVITAI_DIR, safe_name)
        expected = int((target.get("sizeKB") or 0) * 1024)
        if not (os.path.exists(local_path) and os.path.getsize(local_path) > 1024):
            if progress_cb:
                progress_cb(0, max(1, expected), "Baixando " + safe_name + " (" + fmt_bytes(expected) + ")...")
            download_file_stream(download_url, local_path, {}, desc=safe_name,
                                 progress_cb=progress_cb, expected_bytes=expected)
        base_model = version.get("baseModel") or "Krea 2"
        trained_words = version.get("trainedWords") or []
        write_model_meta(local_path, base_model, model_name, "krea2", version.get("name"), trained_words, target)
        if not os.path.exists(WAN2GP_DIR):
            if progress_cb:
                progress_cb(0.08, 1.0, "Clonando Wan2GP (1-2 min)...")
            subprocess.check_call(["git", "clone", "-q", "https://github.com/DeepBeepMeep/Wan2GP.git", WAN2GP_DIR], timeout=900)
        _ensure_krea_te_vae(progress_cb=progress_cb)
        w = _spawn_krea_worker(local_path, model_name, progress_cb=progress_cb)
        STATE["krea_worker"] = w
        STATE["backend"] = "wan2gp"
        STATE["family"] = "krea2"
        STATE["loaded"] = True
        STATE["krea_model"] = None
        STATE["pipe"] = None
        STATE["model_path"] = local_path
        STATE["trained_words"] = list(trained_words)
        STATE["config"] = {"label": model_name, "backend": "wan2gp", "family": "krea2", "base_model": base_model}
        if progress_cb:
            progress_cb(1.0, 1.0, model_name + " pronto (worker isolado)!")
        return model_name + " carregado com sucesso!", "krea2"
    except Exception as e:
        traceback.print_exc()
        return "Erro ao carregar Krea-2 custom: " + str(e), None


def load_krea_click(progress=gr.Progress()):
    status, family = load_krea(progress_cb=lambda a, b, d: progress(a / max(1, b), desc=d))
    if family:
        u = ui_apply_family(family)
        return [status] + list(u)
    return [status] + [gr.update()] * 8

def load_local_file(file_path, base_model="Other", force_comfy=False, progress_cb=None):
    try:
        if not file_path or not os.path.exists(file_path):
            return "Arquivo nao encontrado ou invalido.", ""
        family = _family_from_base(base_model)
        # Tenta ler sidecar metadata (P3-22)
        meta_file = str(file_path) + ".meta.json"
        trained = []
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if not base_model or base_model == "Other":
                    family = _family_from_base(meta.get("base_model"))
                trained = meta.get("trained_words") or []
            except Exception:
                pass
        # FALLBACK: metadata 'Other'/'desconhecido' mas arquivo FLUX/SDXL — sniff pelo conteudo
        sniffed = _sniff_family_from_file(file_path, family)
        if sniffed != family:
            print("  Metadata '" + str(family) + "' — detectado por conteudo: " + sniffed)
            family = sniffed

        # AUTO-ROUTING INTELIGENTE: ativa ComfyUI e quantizacao automaticamente para modelos complexos
        if family in ("flux", "flux_dev", "flux_schnell", "flux2_klein", "sd3", "sd35", "anima", "chroma", "auraflow", "hunyuan", "lumina", "ernie"):
            force_comfy = True
            if not STATE.get("quant_method") or STATE.get("quant_method") == "none":
                STATE["quant_method"] = "torchao_int8"
            print("  Auto-routing Inteligente: ativado ComfyUI + quantizacao (" + STATE["quant_method"] + ") para " + family)
        if not force_comfy and FAMILY_PRESETS.get(family, {}).get("diffusers_cls"):
            try:
                load_diffusers_single_file(file_path, base_model, family, {"dtype": "float16", "cpu_offload": "model"})
                STATE["trained_words"] = list(trained)
                STATE["model_path"] = file_path
                return "Modelo local carregado via Diffusers (" + family + ").", family
            except Exception as e:
                unload_current_model()
                print("Diffusers falhou, tentando ComfyUI: " + str(e)[:200])
        ensure_comfyui()
        STATE["model_path"] = file_path
        STATE["backend"] = "comfy"
        STATE["family"] = family
        STATE["loaded"] = True
        STATE["trained_words"] = list(trained)
        STATE["config"] = {"label": os.path.basename(file_path), "backend": "comfy"}
        return "Modelo local carregado via ComfyUI (" + family + ").", family
    except Exception as e:
        traceback.print_exc()
        return "Erro local: " + str(e), ""

# ============================================================================
# 12. GESTAO DE DISCO (P2-19) e BIBLIOTECA LOCAL (P3-22)
# ============================================================================
def list_downloaded_models():
    out = []
    for d in [CIVITAI_DIR, LORA_DIR, VAE_DIR, TI_DIR]:
        for f in sorted(Path(d).glob("*.safetensors")):
            try:
                sz = f.stat().st_size
                label = os.path.basename(str(f))
                if str(f).endswith(".meta.json"):
                    continue
                out.append((str(f), label + " (" + fmt_bytes(sz) + ")"))
            except Exception:
                continue
    out.sort(key=lambda x: -os.path.getsize(x[0]) if os.path.exists(x[0]) else 0)
    return out

def list_history_images():
    """Lista todas as imagens salvas no disco (/content/studio/outputs), ordenadas da mais recente para a mais antiga."""
    files = []
    if os.path.exists(OUTPUTS_DIR):
        for f in os.listdir(OUTPUTS_DIR):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                fp = os.path.join(OUTPUTS_DIR, f)
                try:
                    mtime = os.path.getmtime(fp)
                    files.append((fp, mtime))
                except Exception:
                    pass
    files.sort(key=lambda x: x[1], reverse=True)
    return [fp for fp, mtime in files]

def load_image_metadata(img_path):
    """Carrega metadados do sidecar .meta.json ou do PNGInfo."""
    if not img_path or not os.path.exists(img_path):
        return {}
    meta_path = str(img_path) + ".meta.json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    try:
        from PIL import Image as PILImage
        with PILImage.open(img_path) as im:
            params = im.info.get("parameters") or im.info.get("prompt") or ""
            return {"prompt": params}
    except Exception:
        pass
    return {}

def delete_downloaded(path):
    try:
        if not path or not os.path.exists(path):
            return "Nada para deletar."
        meta = str(path) + ".meta.json"
        os.remove(path)
        if os.path.exists(meta):
            os.remove(meta)
        return "Removido: " + os.path.basename(path)
    except Exception as e:
        return "Erro ao deletar: " + str(e)

# ============================================================================
# 13. API HTTP EXTERNA (P3-26) — stdlib, sem dependencias
# ============================================================================
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

def _check_api_auth(handler, body=None):
    key = STATE.get("api_key") or ""
    if not key:
        return True
    auth = handler.headers.get("Authorization", "")
    return auth == "Bearer " + key or (body and body.get("api_key") == key)

def start_api_server(port=API_PORT):
    class ApiHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send_json(self, code, obj):
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            try:
                path = urllib.parse.urlparse(self.path).path
                if path == "/api/health":
                    free, total = vram_info()
                    self._send_json(200, {"status": "ok", "loaded": STATE.get("loaded"),
                                          "model": (STATE.get("config") or {}).get("label"),
                                          "backend": STATE.get("backend"),
                                          "family": STATE.get("family"),
                                          "app_ver": APP_VER,
                                          "krea_worker": bool(STATE.get("krea_worker")),
                                          "anima_vae_valid": _anima_vae_valid(os.path.join(VAE_DIR, "anima_vae.safetensors")),
                                          "comfy_vae_valid": _anima_vae_valid(os.path.join(COMFY_DIR, "models", "vae", "anima_vae.safetensors")),
                                          "vram_free_gb": free, "vram_total_gb": total})
                elif path == "/api/models":
                    self._send_json(200, {"models": [{"path": p, "label": l} for p, l in list_downloaded_models()]})
                else:
                    self._send_json(404, {"error": "rota nao encontrada"})
            except Exception as e:
                self._send_json(500, {"error": str(e)})

        def _read_body(self):
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            if length > 100 * 1024 * 1024:
                raise ValueError("Payload HTTP excede o limite maximo permitido (100 MB).")
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception:
                return {}

        def do_POST(self):
            try:
                body = self._read_body()
                if not _check_api_auth(self, body):
                    self._send_json(401, {"error": "API key invalida"})
                    return
                path = urllib.parse.urlparse(self.path).path
                if path == "/api/generate":
                    images = run_generation(body, progress_cb=None)
                    out = []
                    for img, seed, info, w, h in images:
                        buf = io.BytesIO()
                        try:
                            save_image_pnginfo(img, buf, info)
                        except Exception as e:
                            self._send_json(500, {"status": "error",
                                                  "message": "Falha ao codificar PNG: " + str(e),
                                                  "seed": seed, "info": info})
                            return
                        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                        out.append({"image": b64, "seed": seed, "info": info, "width": w, "height": h})
                    self._send_json(200, {"status": "ok", "images": out})
                elif path == "/api/load_model":
                    with LOAD_LOCK:
                        st, fam = load_model_from_civitai(body.get("model_url", ""), body.get("civitai_token"),
                                                          body.get("hf_token"), int(body.get("version_index", 0)),
                                                          bool(body.get("force_comfy")))
                    if fam is None:
                        self._send_json(400, {"status": "error", "message": st})
                    else:
                        self._send_json(200, {"status": "ok", "family": fam, "message": st})
                elif path == "/api/repair_vae":
                    # Self-heal: garante WanVAE 2.1 oficial em /studio + copiado p/ o ComfyUI
                    vae = os.path.join(VAE_DIR, "anima_vae.safetensors")
                    if not _anima_vae_valid(vae):
                        if os.path.exists(vae):
                            try:
                                os.remove(vae)
                            except Exception:
                                pass
                        try:
                            download_file_stream(ANIMA_VAE_URL, vae, {}, desc="Baixando qwen_image_vae (repair)")
                        except Exception as e:
                            self._send_json(500, {"status": "error", "message": "reparo VAE falhou: " + str(e)[:200]})
                            return
                    dest = os.path.join(COMFY_DIR, "models", "vae", "anima_vae.safetensors")
                    if _anima_vae_valid(vae):
                        # IMPORTANTE: so copia se o ComfyUI ja esta instalado. Criar /content/ComfyUI
                        # antes do clone faria o 'git clone' falhar (destino nao vazio -> exit 128).
                        if os.path.exists(os.path.join(COMFY_DIR, "main.py")):
                            Path(os.path.dirname(dest)).mkdir(parents=True, exist_ok=True)
                            if not _anima_vae_valid(dest):
                                try:
                                    shutil.copy(vae, dest)
                                except Exception as e:
                                    self._send_json(500, {"status": "error", "message": "copia p/ ComfyUI falhou: " + str(e)[:200]})
                                    return
                        else:
                            print("  repair_vae: ComfyUI nao instalado ainda — copia adiada p/ comfy_ensure_aux_files")
                    self._send_json(200, {"status": "ok",
                                          "app_ver": APP_VER,
                                          "studio_vae_valid": _anima_vae_valid(vae),
                                          "comfy_vae_valid": _anima_vae_valid(dest),
                                          "studio_vae_size": os.path.getsize(vae) if os.path.exists(vae) else 0})
                elif path == "/api/unload":
                    unload_current_model()
                    self._send_json(200, {"status": "ok"})
                else:
                    self._send_json(404, {"error": "rota nao encontrada"})
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})

    srv = ThreadingHTTPServer(("0.0.0.0", port), ApiHandler)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    print("API externa ativa: http://0.0.0.0:" + str(port) + "/api/health")

# ============================================================================
# 14. UI GRADIO
# ============================================================================
CSS = """
* { font-family: 'Inter', 'Segoe UI', sans-serif !important; }
.gradio-container { max-width: 1240px !important; margin: auto !important; }
.brand-header { text-align: center; background: linear-gradient(135deg,#0f0c29 0%,#302b63 55%,#24243e 100%); padding: 26px; border-radius: 16px; margin-bottom: 18px; box-shadow: 0 10px 25px rgba(0,0,0,0.3); }
.brand-title { color: white; font-size: 1.9em; font-weight: 700; margin: 0 0 6px 0; }
.brand-subtitle { color: rgba(255,255,255,0.88); font-size: 0.95em; }
"""

def ui_family_info(family):
    if not family or family == "None":
        return "Selecione um modelo primeiro."
    preset = FAMILY_PRESETS.get(family, FAMILY_PRESETS["other"])
    steps = preset.get("steps", {})
    tw = ", ".join(STATE.get("trained_words") or [])[:200]
    label = str(preset.get("label", family))
    cfg_val = preset.get("cfg", 3.5)
    def_steps = steps.get("default", 25)
    def_res = preset.get("default_res", "768px (Balanced)")
    def_aspect = preset.get("default_aspect", "1:1 Square")
    txt = f"### ✨ Configuração Otimizada Automática: **{label}**\n\n"
    txt += f"✅ **Passos (Steps)**: `{def_steps}` (faixa recomendada: {steps.get('min', 1)}-{steps.get('max', 50)})\n"
    txt += f"✅ **Guidance (CFG)**: `{cfg_val}` | **Scheduler**: `{preset.get('scheduler', 'padrao')}`\n"
    txt += f"✅ **Resolução Ideal**: `{def_res}` | **Proporção**: `{def_aspect}`\n"
    if preset.get("prompt_prefix"):
        txt += f"📌 **Prefixo de Prompt**: `{preset.get('prompt_prefix')}`\n"
    if tw:
        txt += f"⚡ **Trigger Words ativas**: `{tw}`\n"
    if preset.get("notes"):
        txt += f"\n💡 *Dica do Modelo*: {preset.get('notes')}\n"
    return txt

def ui_apply_family(family):
    preset = FAMILY_PRESETS.get(family, FAMILY_PRESETS["other"])
    steps = preset.get("steps", {})
    neg_rec = preset.get("neg_prefix") or ""
    if not neg_rec and family in ("sd15", "sd15_hyper", "sd15_lcm"):
        neg_rec = "worst quality, low quality, bad anatomy, bad hands, text, watermark, signature, blurry"
    elif not neg_rec and family in ("sdxl", "sdxl_lightning", "sdxl_lcm", "sdxl_hyper"):
        neg_rec = "worst quality, low quality, bad anatomy, bad hands, distorted, disfigured, blurry, watermark"
    return (
        gr.update(minimum=steps.get("min", 1), maximum=steps.get("max", 50), value=steps.get("default", 25)),
        gr.update(value=float(preset.get("cfg", 3.5))),
        gr.update(choices=preset.get("resolutions", ["768px (Balanced)"]), value=preset.get("default_res", "768px (Balanced)")),
        gr.update(choices=preset.get("aspects", ["1:1 Square"]), value=preset.get("default_aspect", "1:1 Square")),
        gr.update(value=neg_rec),
        gr.update(value=True),
        gr.update(value=True),
        ui_family_info(family),
    )

def query_civitai_click(url, civ_token, hf_token):
    try:
        model_id = parse_civitai_id(url)
        if not model_id:
            return "URL/ID invalido.", gr.update(choices=[], value=None), ""
        if hf_token:
            STATE["hf_token"] = str(hf_token).strip()
        if civ_token:
            STATE["civitai_token"] = str(civ_token).strip()
        data = civitai_get_model(model_id, STATE.get("civitai_token"))
        versions = civitai_sorted_versions(data)
        STATE["civitai_model"] = data
        STATE["civitai_versions"] = versions
        choices = []
        for i, v in enumerate(versions):
            total = sum((f.get("sizeKB") or 0) for f in v.get("files", [])) * 1024
            choices.append(str(i) + " | " + str(v.get("name", "?")) + " | base: " + str(v.get("baseModel")) + " | " + fmt_bytes(total))
        preview = ""
        images = data.get("images") or []
        if images:
            preview = "![preview](" + str(images[0].get("url", "")) + ")"
        tw = ", ".join((versions[0].get("trainedWords") or [])[:6]) if versions else ""
        card = ("## " + str(data.get("name", "?")) + "\n\n"
                + str(data.get("description") or "")[:600] + "\n\n"
                + "**Tipo:** " + str(data.get("type")) + " | **Versoes:** " + str(len(versions)) + "\n"
                + "**Trigger words (ultima versao):** `" + tw + "`\n\n"
                + preview)
        msg = "Modelo consultado: " + str(data.get("name")) + " | " + str(len(versions)) + " versoes."
        return card, gr.update(choices=choices, value=choices[0] if choices else None), msg
    except Exception as e:
        traceback.print_exc()
        return "Erro ao consultar: " + str(e), gr.update(choices=[], value=None), ""

def load_civitai_click(url, civ_token, hf_token, version_choice, force_comfy, quant_method, use_compile, progress=gr.Progress()):
    try:
        if quant_method and quant_method != "auto":
            STATE["quant_method"] = str(quant_method)
        else:
            STATE["quant_method"] = "none"
        STATE["use_compile"] = bool(use_compile)
        idx = 0
        if version_choice:
            try:
                idx = int(str(version_choice).split("|")[0].strip())
            except Exception:
                idx = 0
        status, family = load_model_from_civitai(url, civ_token, hf_token, idx, bool(force_comfy),
                                                 progress_cb=lambda a, b, d: progress(a / max(1, b), desc=d))
        if family:
            u = ui_apply_family(family)
            return [status] + list(u)
        return [status] + [gr.update()] * 8
    except Exception as e:
        traceback.print_exc()
        return ["Erro: " + str(e)] + [gr.update()] * 8

def _parse_image_input(img_input):
    """Converte QUALQUER retorno do Gradio (PIL.Image, dict com 'image'/'mask'/'composite'/'background', numpy array ou string path) para PIL.Image."""
    if img_input is None:
        return None
    if isinstance(img_input, Image.Image):
        return img_input
    if isinstance(img_input, dict):
        for k in ("composite", "image", "mask", "background"):
            val = img_input.get(k)
            if val is not None:
                if isinstance(val, Image.Image):
                    return val
                if np is not None and isinstance(val, np.ndarray):
                    return Image.fromarray(val)
                if isinstance(val, str) and os.path.exists(val):
                    return Image.open(val)
        layers = img_input.get("layers")
        if layers and len(layers) > 0:
            layer = layers[0]
            if isinstance(layer, Image.Image):
                return layer
            if np is not None and isinstance(layer, np.ndarray):
                return Image.fromarray(layer)
    if np is not None and isinstance(img_input, np.ndarray):
        return Image.fromarray(img_input)
    if isinstance(img_input, str) and os.path.exists(img_input):
        return Image.open(img_input)
    return None

def generate_ui(prompt, negative, steps, aspect, resolution, seed, num_images, cfg,
                style, use_template, use_trigger, init_image, mask_image, denoise,
                hires_fix, hires_denoise, hires_scale, prompt_matrix,
                progress=gr.Progress(track_tqdm=True)):
    try:
        if not STATE.get("loaded"):
            yield gr.update(value=None, selected_index=None), "Nenhum modelo carregado.", gr.update(visible=False)
            return
        p_prompt = _enhance(prompt, style) if style and style != "None" else (prompt or "")
        base_px = _parse_px(resolution, 1024)
        w, h = _dims(aspect, base_px)
        m = re.search(r"[\d.]+", str(hires_scale or "2x"))
        hscale = float(m.group()) if m else 2.0
        init_pil = _parse_image_input(init_image)
        mask_pil = _parse_image_input(mask_image)
        params = {
            "prompt": p_prompt, "negative_prompt": negative or "",
            "steps": steps, "cfg": cfg, "width": w, "height": h,
            "seed": -1 if (seed is None or seed < 0) else int(seed),
            "num_images": int(num_images), "use_template": use_template, "use_trigger": use_trigger,
            "init_image": init_pil, "mask_image": mask_pil, "strength": float(denoise),
            "hires_fix": hires_fix, "hires_denoise": float(hires_denoise), "hires_scale": hscale,
            "prompt_matrix": prompt_matrix,
        }
        yield gr.update(value=None, selected_index=None), "Iniciando...", gr.update(visible=False)
        t0 = time.time()
        images = run_generation(params, progress_cb=lambda a, b, d: progress(a / max(1, b), desc=d))
        gallery_imgs = []
        import datetime
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        for idx2, (img, s, inf, ww, hh) in enumerate(images):
            out_name = f"gen_{now_str}_s{s}_{idx2}.png"
            out_path = os.path.join(OUTPUTS_DIR, out_name)
            save_image_pnginfo(img, out_path, inf)
            meta = {
                "filename": out_name, "path": out_path,
                "prompt": params.get("prompt", ""), "negative_prompt": params.get("negative_prompt", ""),
                "seed": s, "steps": params.get("steps"), "cfg": params.get("cfg"),
                "width": ww, "height": hh,
                "model": (STATE.get("config") or {}).get("label", "Desconhecido"),
                "family": STATE.get("family", ""),
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            try:
                with open(out_path + ".meta.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            gallery_imgs.append(out_path)

        zip_path = None
        if len(gallery_imgs) > 1:
            import zipfile
            tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
            zip_path = tmp.name
            label = re.sub(r"[^\w\-.]", "_", str((STATE.get("config") or {}).get("label", "model")))
            with zipfile.ZipFile(zip_path, "w") as zf:
                for idx2, p_saved in enumerate(gallery_imgs):
                    zf.write(p_saved, label + "_" + str(idx2 + 1) + ".png")
        elapsed = time.time() - t0
        seeds = [s for img, s, inf, ww, hh in images]
        status = ("OK " + str(len(images)) + " imagem(ns) em " + "{:.1f}".format(elapsed) +
                  "s | Seeds: " + str(seeds) + " | " + str(w) + "x" + str(h))
        yield gr.update(value=gallery_imgs, selected_index=0), status, gr.update(value=zip_path, visible=(zip_path is not None))
    except Exception as e:
        traceback.print_exc()
        gc.collect(); torch.cuda.empty_cache()
        yield gr.update(value=None, selected_index=None), "Erro: " + str(e), gr.update(visible=False)

def cmp_refresh():
    items = list_downloaded_models()
    choices = [l for p, l in items]
    val_a = choices[0] if choices else None
    val_b = choices[1] if len(choices) > 1 else val_a
    return gr.update(choices=choices, value=val_a), gr.update(choices=choices, value=val_b)

def compare_click(choice_a, choice_b, prompt, negative, steps, cfg, seed, aspect, resolution, style,
                  civ_token, hf_token, progress=gr.Progress()):
    try:
        items = list_downloaded_models()
        path_a = None
        path_b = None
        for p, l in items:
            if l == choice_a:
                path_a = p
            if l == choice_b:
                path_b = p
        if not path_a and choice_a:
            path_a = choice_a
        if not path_b and choice_b:
            path_b = choice_b

        if not path_a or not path_b:
            return gr.update(value=None), "Selecione ou informe dois modelos instalados para comparar."

        base_px = _parse_px(resolution, 1024)
        w, h = _dims(aspect, base_px)
        p_prompt = _enhance(prompt, style) if style and style != "None" else (prompt or "")
        target_seed = 12345 if (seed is None or seed < 0) else int(seed)

        out = []
        for label, model_target in [("Modelo A: " + str(choice_a or path_a), path_a),
                                   ("Modelo B: " + str(choice_b or path_b), path_b)]:
            progress(0.1, desc="Carregando " + label)
            with LOAD_LOCK:
                unload_current_model()
                if os.path.exists(model_target):
                    st, fam = load_local_file(model_target, "Other", False)
                else:
                    st, fam = load_model_from_civitai(model_target, civ_token, hf_token, 0, False)
                if fam is None:
                    return gr.update(value=None), "Falha ao carregar " + label + ": " + str(st)

            params = {"prompt": p_prompt, "negative_prompt": negative or "", "steps": steps, "cfg": cfg,
                      "width": w, "height": h, "seed": target_seed,
                      "num_images": 1, "use_template": True, "use_trigger": True}
            res = run_generation(params, progress_cb=None)
            if res and len(res) > 0:
                out.append((label, res[0][0], res[0][1]))

        if not out:
            return gr.update(value=None), "Nenhuma imagem foi gerada na comparação."

        return gr.update(value=[img for l, img, s in out], columns=2), \
               "Comparação concluída! " + " | ".join([l + " (seed " + str(s) + ")" for l, img, s in out])
    except Exception as e:
        traceback.print_exc()
        return gr.update(value=None), "Erro na comparação: " + str(e)

def library_refresh():
    items = list_downloaded_models()
    if not items:
        return gr.update(choices=[], value=None), "Biblioteca vazia. Baixe modelos primeiro."
    return gr.update(choices=[l for p, l in items], value=items[0][1]), \
           "Biblioteca: " + str(len(items)) + " arquivo(s)."

def library_load_click(choice_label):
    try:
        items = list_downloaded_models()
        path = None
        for p, l in items:
            if l == choice_label:
                path = p
                break
        if not path:
            return ["Selecione um modelo da biblioteca."] + [gr.update()] * 8
        st, fam = load_local_file(path, "Other", False)
        if fam:
            u = ui_apply_family(fam)
            return [st] + list(u)
        return [st] + [gr.update()] * 8
    except Exception as e:
        return ["Erro: " + str(e)] + [gr.update()] * 8

def library_delete_click(choice_label):
    items = list_downloaded_models()
    for p, l in items:
        if l == choice_label:
            st = delete_downloaded(p)
            lib = library_refresh()
            return st, lib[0], lib[1]
    return "Nada para deletar.", gr.update(choices=[], value=None), ""

def set_api_key(key):
    STATE["api_key"] = (key or "").strip()
    return ("API key definida: " + ("SIM" if STATE["api_key"] else "NAO (aberta)") +
            " | API ativa em :" + str(API_PORT) + " — endpoints: /api/health, /api/load_model, /api/generate, /api/models, /api/unload")

CIVITAI_EXAMPLES = [
    ("Arthemy Comics Anima", "https://civitai.com/models/2700278"),
    ("Arthemy Comics Illustrious", "https://civitai.com/models/1273254"),
    ("Arthemy Western Art", "https://civitai.com/models/2241572"),
    ("Krea-2-Turbo Oficial (INT8, via botao)", "DeepBeepMeep/krea-2"),
    ("Arthemy Toons Illustrious", "https://civitai.com/models/1906150"),
    ("Arthemy Anime", "https://civitai.com/models/2442462"),
    ("Arthemy Comics", "https://civitai.com/models/54073"),
    ("A-Zovya RPG Artist Tools", "https://civitai.com/models/8124"),
    ("Childrens Stories Toolkit", "https://civitai.com/models/64544"),
    ("Kestral Flux Anime", "https://civitai.com/models/697877"),
    ("Kestral PNY", "https://civitai.com/models/623819"),
    ("Cor Epica", "https://civitai.com/models/552880"),
    ("Fantasy Classic Style", "https://civitai.com/models/2860726"),
    ("Krea-2 Tubro Q8 From BF16", "https://civitai.com/models/2792164"),
    ("Comic Book Illustrious Or Anima", "https://civitai.com/models/2836562"),
    ("Perfectdeliberate", "https://civitai.com/models/24350"),
    ("Oberon", "https://civitai.com/models/2107235"),
    ("Arthemy Comics Krea-2", "https://civitai.com/models/2759057"),
    ("El Pistolero", "https://civitai.com/models/2478806"),
    ("Tratto Nero", "https://civitai.com/models/1907563"),
    ("Hoseki Lustrousmix Anima V1.0", "https://civitai.com/models/941345"),
    ("Perfectdeliberate Anime", "https://civitai.com/models/111274"),
    ("Arthemy Painter Illustrious", "https://civitai.com/models/1598875"),
    ("Hyphoria", "https://civitai.com/models/1595884"),
    ("Genuine Illustrious From Hades", "https://civitai.com/models/2252910"),
    ("Bridge Toons Comic Mix", "https://civitai.com/models/1832088"),
    ("Toon Factory From Hades", "https://civitai.com/models/2226128"),
    ("Red Lily Or Illu", "https://civitai.com/models/2070771"),
    ("Pony Diffusion V6 XL", "https://civitai.com/models/257749"),
    ("Cat Citron Anime Treasure Illustrious And NoobAI", "https://civitai.com/models/131986"),
    ("New Era New Esthetic Retro Anime", "https://civitai.com/models/137781"),
    ("Cheyenne", "https://civitai.com/models/198051"),
    ("D&D Battlemaps SDXL 10", "https://civitai.com/models/1073005"),
    ("Arthemy Comics Pony", "https://civitai.com/models/1212908"),
    ("Fantasyland", "https://civitai.com/models/387841"),
    ("Arthemy Comics XL", "https://civitai.com/models/462532"),
    ("Meinamix", "https://civitai.com/models/7240"),
    ("Arthemy Comics FLUX", "https://civitai.com/models/799804"),
    ("Sxz Luma", "https://civitai.com/models/25831"),
    ("Everclear PNY By Zovya", "https://civitai.com/models/341433"),
    ("Sdxlnijiseven", "https://civitai.com/models/120765"),
    ("Realistic Fantasy Mix SDXL", "https://civitai.com/models/136220"),
    ("Rev Animated", "https://civitai.com/models/7371"),
    ("Animat Background V1", "https://civitai.com/models/270238"),
    ("SDXL Unstable Diffusers Yamermix", "https://civitai.com/models/84040"),
    ("Realities Edge XL Lightning Turbo", "https://civitai.com/models/129666"),
    ("Dreamshaper XL", "https://civitai.com/models/112902"),
    ("D&D Battlemaps", "https://civitai.com/models/23240"),
    ("Epicrealism", "https://civitai.com/models/25694"),
    ("RPG", "https://civitai.com/models/1116"),
    ("Arthemy Objects", "https://civitai.com/models/88128"),
    ("Fantassified Icons", "https://civitai.com/models/4713"),
    ("D&D Top Down Token", "https://civitai.com/models/23328"),
    ("DnD Map Generator", "https://civitai.com/models/5012"),
    ("Awesome RPG Icon 2000", "https://civitai.com/models/14483"),
    ("Fantasy World", "https://civitai.com/models/11031"),
    ("Dungeons And Diffusion V3", "https://civitai.com/models/90"),
    ("Handpainted RPG Icons", "https://civitai.com/models/4052"),
    ("Stylized RPG Game Icons", "https://civitai.com/models/1239"),
]

def _ex_url_aux(choice, examples):
    for _n, _u in examples:
        if choice == _n:
            return _u
    return ""
def _idx_from_dd(val):
    try:
        return int(str(val).split(" | ")[0])
    except Exception:
        return 0
def query_aux_click(url, civ_token, wanted_type):
    try:
        if civ_token:
            STATE["civitai_token"] = str(civ_token).strip()
        card, choices, versions, msg = aux_query(url, STATE.get("civitai_token"), wanted_type)
        return card, gr.update(choices=choices, value=choices[0] if choices else None), msg
    except Exception as e:
        traceback.print_exc()
        return "Erro: " + str(e), gr.update(choices=[], value=None), ""
def lora_load_click(url, token, ver, weight, progress=gr.Progress()):
    try:
        st = load_lora_from_civitai(url, token, float(weight or 1.0), _idx_from_dd(ver))
        return st + " | Ativos: " + (", ".join(lora_active_choices()) or "(nenhum)")
    except Exception as e:
        return "Erro: " + str(e)
def vae_load_click(url, token, ver, progress=gr.Progress()):
    try:
        return load_vae_from_civitai(url, token, _idx_from_dd(ver))
    except Exception as e:
        return "Erro: " + str(e)
def ti_load_click(url, token, ver, progress=gr.Progress()):
    try:
        return load_ti_from_civitai(url, token, _idx_from_dd(ver))
    except Exception as e:
        return "Erro: " + str(e)
def lora_active_refresh():
    return gr.update(choices=lora_active_choices(), value=None)
def lib_refresh_aux(kind):
    items = list_aux_by_kind(kind)
    return gr.update(choices=[(lbl, pth) for pth, lbl in items], value=None)
def lib_del_aux(path):
    return delete_aux_file(path)
def clear_lora2():
    clear_lora()
    return "LoRAs removidos.", gr.update(choices=[], value=None)
def lora_rm_click(path):
    if path:
        return lora_remove_one(path), gr.update(choices=lora_active_choices(), value=None)
    return "Selecione um LoRA ativo.", gr.update()
def lora_lib_load_click(path, weight):
    if path:
        return apply_lora_local(path, float(weight or 1.0))
    return "Selecione um LoRA da biblioteca."
def vae_lib_load_click(path):
    if path:
        return apply_vae_local(path)
    return "Selecione um VAE da biblioteca."
def ti_lib_load_click(path):
    if path:
        return load_ti_local(path)
    return "Selecione um TI da biblioteca."

with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="Advanced Multi-Model Image Studio (Definitivo v2)") as demo:
    gr.HTML(
        '<div class="brand-header">'
        '<div class="brand-title">🎨 Advanced Multi-Model Image Studio v2</div>'
        '<div class="brand-subtitle">Qualquer base model do Civitai | Diffusers + ComfyUI + Wan2GP | Hires, Inpaint, ControlNet, API externa</div>'
        '</div>'
    )
    with gr.Tabs():
        with gr.TabItem("Modelo / Configuracao"):
            gr.Markdown("### Civitai — qualquer modelo, versao mais recente, arquivo recomendado")
            with gr.Row():
                civitai_example = gr.Dropdown(label="Exemplos populares", choices=[""] + [e[0] for e in CIVITAI_EXAMPLES], value="")
            civitai_url = gr.Textbox(label="URL ou ID do modelo no Civitai", placeholder="https://civitai.com/models/2700278 ou apenas 2700278")
            with gr.Row():
                civitai_token = gr.Textbox(label="Civitai API Token (pre-preenchido)", type="password", value=STATE.get("civitai_token") or "")
                hf_token = gr.Textbox(label="HF Token (pre-preenchido — FLUX/gated)", type="password", value=STATE.get("hf_token") or "")
            with gr.Row():
                query_btn = gr.Button("🔎 Consultar modelo", variant="secondary")
                civitai_load_btn = gr.Button("⬇️ Baixar e Carregar (versao mais recente)", variant="primary", size="lg")
            version_dropdown = gr.Dropdown(label="Versao (mais recente primeiro)", choices=[], value=None)
            with gr.Accordion("⚙️ Sobrescrever Configurações Avançadas (Opcional — Auto por Padrão)", open=False):
                with gr.Row():
                    force_comfy = gr.Checkbox(False, label="Forçar ComfyUI (manual)")
                    quant_method = gr.Dropdown(["auto", "torchao_int8", "fp8_e4m3fn", "bnb_8bit", "bnb_4bit", "none"], value="auto", label="Quantização (auto = automático)")
                    use_compile = gr.Checkbox(False, label="torch.compile (JIT opt-in)")
            model_card = gr.Markdown("")
            civitai_status = gr.Textbox(label="Status", lines=8)

            gr.Markdown("---")


            gr.Markdown("---")
            gr.Markdown("### LoRA / VAE / TextualInversion")
            gr.Markdown("👉 Gestao completa (exemplos, versoes, consultar, baixar, biblioteca) na aba **LoRA / VAE / TI**.")

            gr.Markdown("---")
            gr.Markdown("### Biblioteca local e disco (troca sem re-download)")
            with gr.Row():
                lib_refresh_btn = gr.Button("Atualizar biblioteca", variant="secondary")
                lib_dropdown = gr.Dropdown(label="Modelos baixados", choices=[], value=None, scale=3)
                load_lib_btn = gr.Button("Carregar da biblioteca", variant="primary")
                delete_lib_btn = gr.Button("🗑️ Deletar", variant="stop")
            lib_info = gr.Textbox(label="Biblioteca", lines=2)

            gr.Markdown("---")
            gr.Markdown("### API externa (P3-26) — outra aplicacao faz requisicoes e recebe a imagem")
            api_key_box = gr.Textbox(label="API key (opcional; vazia = aberta)", type="password", value=STATE.get("api_key") or "")
            api_btn = gr.Button("Definir API key")
            api_info = gr.Markdown("Endpoints: `GET /api/health` · `POST /api/load_model` · `POST /api/generate` · `GET /api/models` · `POST /api/unload`")

            gr.Markdown("---")
            gr.Markdown("### Comparacao Lado a Lado (Modelos Instalados da Biblioteca)")
            with gr.Row():
                cmp_dropdown_a = gr.Dropdown(label="Modelo A (Instalado)", choices=[], value=None, scale=2)
                cmp_dropdown_b = gr.Dropdown(label="Modelo B (Instalado)", choices=[], value=None, scale=2)
                cmp_refresh_btn = gr.Button("🔄 Atualizar Lista", variant="secondary")
            cmp_btn = gr.Button("⚖️ Comparar Lado a Lado", variant="primary")
            cmp_status = gr.Textbox(label="Status comparacao", lines=2)

        with gr.TabItem("Gerar"):
            with gr.Row():
                with gr.Column(scale=3):
                    prompt = gr.Textbox(label="📝 Prompt", lines=4, placeholder="Descreva a imagem... (dica: use ';' para prompt matrix ex.: cidade;noite|dia;chuva)")
                    with gr.Row():
                        enhance_btn = gr.Button("✨ Enhance Prompt", variant="secondary")
                        style_preset = gr.Dropdown(STYLES, value="None", label="Estilo")
                        use_template = gr.Checkbox(True, label="Auto-tags da familia")
                        use_trigger = gr.Checkbox(True, label="Trigger words (Civitai)")
                    with gr.Row():
                        aspect_ratio = gr.Dropdown(["1:1 Square", "16:9 Landscape", "9:16 Portrait", "4:3 Standard", "3:4 Portrait"], value="1:1 Square", label="Proporcao")
                        resolution = gr.Dropdown(["512px (Fast)", "768px (Balanced)", "1024px (Standard)", "1152px (High)"], value="768px (Balanced)", label="Resolucao")
                    with gr.Row():
                        steps = gr.Slider(1, 50, 25, step=1, label="Steps")
                        num_images = gr.Slider(1, 6, 1, step=1, label="Imagens")
                    with gr.Accordion("Avançado", open=False):
                        negative_prompt = gr.Textbox(label="Negative Prompt", lines=2, placeholder="(se vazio, usa a recomendada da familia)")
                        seed = gr.Number(-1, label="Seed (-1 = aleatorio)", precision=0)
                        guidance_scale = gr.Slider(0.0, 20.0, 6.5, step=0.1, label="Guidance (CFG)")
                        with gr.Row():
                            init_image = gr.Image(label="Img2Img (opcional)", type="pil")
                            mask_image = gr.Image(label="Mascara Inpaint (opcional)", type="pil")
                            ctrl_image = gr.Image(label="ControlNet Canny (pose/contornos)", type="pil")
                        denoise = gr.Slider(0.0, 1.0, 0.7, step=0.05, label="Forca (denoise) img2img/inpaint")
                        with gr.Row():
                            hires_fix = gr.Checkbox(False, label="Hires fix (2 passos)")
                            hires_denoise = gr.Slider(0.2, 0.7, 0.45, step=0.05, label="Hires denoise")
                            hires_scale = gr.Dropdown(["2x", "1.5x"], value="2x", label="Hires scale")
                            prompt_matrix = gr.Checkbox(False, label="Prompt matrix")
                    with gr.Row():
                        gen_btn = gr.Button("🎨 Gerar", variant="primary", size="lg")
                        clear_btn = gr.Button("🧹 Limpar", variant="secondary", size="lg")
                    family_info = gr.Markdown("")
                with gr.Column(scale=1):
                    gallery = gr.Gallery(label="Saida", columns=2, rows=2, object_fit="contain", preview=True)
                    with gr.Row():
                        send_img2img_btn = gr.Button("⏩ Usar em Img2Img", variant="secondary", size="sm")
                        send_mask_btn = gr.Button("⏩ Usar em Máscara Inpaint", variant="secondary", size="sm")
                        send_ctrl_btn = gr.Button("⏩ Usar em ControlNet", variant="secondary", size="sm")
                    zip_out = gr.File(label="Download ZIP", visible=False)
                    status_out = gr.Textbox(label="Status", interactive=False)
                    cmp_gallery = gr.Gallery(label="Comparacao", columns=2, rows=1, object_fit="contain", preview=True)

        with gr.TabItem("🎯 LoRA"):
            gr.Markdown("### 🎯 LoRA — sessao independente (estilos/efeitos sobre o modelo)")
            gr.Markdown("_Integrado a geracao: ativos aplicados ao modelo + passados ao ComfyUI. Combina com VAE e TI ativos._")
            gr.Markdown("**Exemplos populares (pre-cadastrados)**")
            with gr.Row():
                lora_example = gr.Dropdown(label="Exemplos populares de LoRA", choices=[""] + [e[0] for e in LORA_EXAMPLES], value="", scale=3)
            lora_url = gr.Textbox(label="URL ou ID do LoRA no Civitai", placeholder="https://civitai.com/models/122359 ou 122359")
            with gr.Row():
                lora_query_btn = gr.Button("🔎 Consultar LoRA", variant="secondary")
                lora_ver_dd = gr.Dropdown(label="Versao (mais recente primeiro)", choices=[], value=None, scale=2)
                lora_weight = gr.Slider(0.0, 2.0, 1.0, step=0.05, label="Peso do LoRA", scale=1)
            lora_card = gr.Markdown("")
            with gr.Row():
                lora_dl_btn = gr.Button("⬇️ Baixar e Ativar LoRA", variant="primary")
                clear_lora_btn2 = gr.Button("Remover todos LoRAs", variant="secondary")
                lora_rm_btn = gr.Button("Remover selecionado", variant="secondary")
            with gr.Row():
                lora_active_dd = gr.Dropdown(label="LoRAs ativos (selecione p/ remover)", choices=[], value=None, scale=2)
                lora_comp_refresh = gr.Button("🔄 Estado dos componentes (LoRA+VAE+TI)", variant="secondary")
            with gr.Row():
                lora_lib_refresh = gr.Button("🔄 Biblioteca LoRA", variant="secondary")
                lora_lib_dd = gr.Dropdown(label="LoRAs baixados", choices=[], value=None, scale=2)
                lora_lib_load = gr.Button("⬆️ Carregar da biblioteca", variant="secondary")
                lora_del_btn = gr.Button("🗑️ Deletar", variant="stop")
            lora_status = gr.Textbox(label="Status LoRA / Componentes ativos", lines=3)

        with gr.TabItem("🎨 VAE"):
            gr.Markdown("### 🎨 VAE — sessao independente (decodificador: cor/qualidade)")
            gr.Markdown("_Integrado a geracao: aplicado ao modelo carregado (pipe.vae)._")
            gr.Markdown("**Exemplos populares (pre-cadastrados)**")
            with gr.Row():
                vae_example = gr.Dropdown(label="Exemplos populares de VAE", choices=[""] + [e[0] for e in VAE_EXAMPLES], value="", scale=3)
            vae_url = gr.Textbox(label="URL ou ID do VAE no Civitai", placeholder="https://civitai.com/models/296576 ou 296576")
            with gr.Row():
                vae_query_btn = gr.Button("🔎 Consultar VAE", variant="secondary")
                vae_ver_dd = gr.Dropdown(label="Versao (mais recente primeiro)", choices=[], value=None, scale=2)
            vae_card = gr.Markdown("")
            with gr.Row():
                vae_dl_btn = gr.Button("⬇️ Baixar e Aplicar VAE ao modelo atual", variant="primary")
                vae_comp_refresh = gr.Button("🔄 Estado dos componentes", variant="secondary")
            with gr.Row():
                vae_lib_refresh = gr.Button("🔄 Biblioteca VAE", variant="secondary")
                vae_lib_dd = gr.Dropdown(label="VAEs baixados", choices=[], value=None, scale=2)
                vae_lib_load = gr.Button("⬆️ Aplicar da biblioteca", variant="secondary")
                vae_del_btn = gr.Button("🗑️ Deletar", variant="stop")
            vae_status = gr.Textbox(label="Status VAE / Componentes ativos", lines=3)

        with gr.TabItem("🧠 TextualInversion"):
            gr.Markdown("### 🧠 TextualInversion — sessao independente (embeddings / trigger words)")
            gr.Markdown("_Integrado a geracao: registrado no modelo + copiado p/ ComfyUI. Uso: `<nome>` ou `embedding:nome` no prompt._")
            gr.Markdown("**Exemplos populares (pre-cadastrados)**")
            with gr.Row():
                ti_example = gr.Dropdown(label="Exemplos populares de TI", choices=[""] + [e[0] for e in TI_EXAMPLES], value="", scale=3)
            ti_url = gr.Textbox(label="URL ou ID do TI no Civitai", placeholder="https://civitai.com/models/7808 ou 7808")
            with gr.Row():
                ti_query_btn = gr.Button("🔎 Consultar TI", variant="secondary")
                ti_ver_dd = gr.Dropdown(label="Versao (mais recente primeiro)", choices=[], value=None, scale=2)
            ti_card = gr.Markdown("")
            with gr.Row():
                ti_dl_btn = gr.Button("⬇️ Baixar e Carregar TI", variant="primary")
                ti_comp_refresh = gr.Button("🔄 Estado dos componentes", variant="secondary")
            with gr.Row():
                ti_lib_refresh = gr.Button("🔄 Biblioteca TI", variant="secondary")
                ti_lib_dd = gr.Dropdown(label="TIs baixados", choices=[], value=None, scale=2)
                ti_lib_load = gr.Button("⬆️ Carregar da biblioteca", variant="secondary")
                ti_del_btn = gr.Button("🗑️ Deletar", variant="stop")
            ti_status = gr.Textbox(label="Status TI / Componentes ativos", lines=3)

        with gr.TabItem("🖼️ Histórico & Galeria"):
            gr.Markdown("### Histórico Geral de Imagens (Criadas & Salvas no Disco)")
            with gr.Row():
                hist_refresh_btn = gr.Button("🔄 Atualizar Histórico", variant="primary")
                hist_status = gr.Textbox(label="Status Histórico", interactive=False)
            with gr.Row():
                with gr.Column(scale=3):
                    history_gallery = gr.Gallery(label="Todas as Imagens no Disco (clique na imagem para selecionar)", columns=4, rows=3, object_fit="contain", preview=True)
                with gr.Column(scale=2):
                    hist_preview = gr.Image(label="Imagem Selecionada", type="pil")
                    hist_meta_card = gr.Markdown("Selecione uma imagem da galeria à esquerda para ver os metadados.")
                    with gr.Row():
                        hist_to_img2img_btn = gr.Button("⏩ Usar em Img2Img", variant="secondary", size="sm")
                        hist_to_mask_btn = gr.Button("⏩ Usar em Máscara Inpaint", variant="secondary", size="sm")
                        hist_to_ctrl_btn = gr.Button("⏩ Usar em ControlNet", variant="secondary", size="sm")
                    hist_delete_btn = gr.Button("🗑️ Deletar do Disco", variant="stop", size="sm")

    def update_example_url(choice):
        for name, url in CIVITAI_EXAMPLES:
            if choice == name:
                return url
        return ""

    def set_ctrl_image(img):
        STATE["ctrl_image"] = img
        return "ControlNet image definida (aplicada via ComfyUI)" if img is not None else "ControlNet removida"

    def send_to_img2img(gallery_data):
        if not gallery_data:
            return None, "Nenhuma imagem gerada para enviar."
        item = gallery_data[0]
        img = item[0] if isinstance(item, (list, tuple)) else (item.get("name") if isinstance(item, dict) else item)
        return img, "Imagem gerada enviada para Img2Img!"

    def send_to_mask(gallery_data):
        if not gallery_data:
            return None, "Nenhuma imagem gerada para enviar."
        item = gallery_data[0]
        img = item[0] if isinstance(item, (list, tuple)) else (item.get("name") if isinstance(item, dict) else item)
        return img, "Imagem gerada enviada para Máscara Inpaint!"

    def send_to_controlnet(gallery_data):
        if not gallery_data:
            return None, "Nenhuma imagem gerada para o ControlNet."
        item = gallery_data[0]
        img = item[0] if isinstance(item, (list, tuple)) else (item.get("name") if isinstance(item, dict) else item)
        STATE["ctrl_image"] = img
        return img, "Imagem gerada definida como referência do ControlNet!"

    def history_refresh():
        imgs = list_history_images()
        return gr.update(value=imgs), f"Histórico: {len(imgs)} imagem(ns) salvas no disco."

    def on_history_select(evt: gr.SelectData):
        imgs = list_history_images()
        idx = evt.index
        if idx < 0 or idx >= len(imgs):
            return None, "Imagem não encontrada.", "Erro"
        img_path = imgs[idx]
        meta = load_image_metadata(img_path)
        card = f"### 📷 Metadados da Imagem\n"
        card += f"**Modelo**: `{meta.get('model', 'Desconhecido')}` | **Família**: `{meta.get('family', '-')}`\n\n"
        card += f"**Prompt**: `{meta.get('prompt', '-')}`\n\n"
        if meta.get("negative_prompt"):
            card += f"**Negative**: `{meta.get('negative_prompt')}`\n\n"
        card += f"**Seed**: `{meta.get('seed', '-')}` | **Steps**: `{meta.get('steps', '-')}` | **CFG**: `{meta.get('cfg', '-')}`\n\n"
        card += f"**Resolução**: `{meta.get('width', '-')}x{meta.get('height', '-')}` | **Data**: `{meta.get('timestamp', '-')}`\n"
        STATE["selected_history_image"] = img_path
        try:
            pil_img = Image.open(img_path)
            return pil_img, card, f"Selecionada: {os.path.basename(img_path)}"
        except Exception:
            return None, card, f"Erro ao abrir {os.path.basename(img_path)}"

    def send_hist_img2img():
        p = STATE.get("selected_history_image")
        if not p or not os.path.exists(p):
            return None, "Selecione uma imagem do histórico primeiro."
        return Image.open(p), "Imagem do histórico enviada para Img2Img!"

    def send_hist_mask():
        p = STATE.get("selected_history_image")
        if not p or not os.path.exists(p):
            return None, "Selecione uma imagem do histórico primeiro."
        return Image.open(p), "Imagem do histórico enviada para Máscara Inpaint!"

    def send_hist_ctrl():
        p = STATE.get("selected_history_image")
        if not p or not os.path.exists(p):
            return None, "Selecione uma imagem do histórico primeiro."
        STATE["ctrl_image"] = p
        return Image.open(p), "Imagem do histórico definida no ControlNet!"

    def delete_hist_image():
        p = STATE.get("selected_history_image")
        if not p or not os.path.exists(p):
            return gr.update(value=[]), None, "Nenhuma imagem selecionada.", ""
        try:
            os.remove(p)
            if os.path.exists(p + ".meta.json"):
                os.remove(p + ".meta.json")
        except Exception:
            pass
        imgs = list_history_images()
        STATE["selected_history_image"] = None
        return gr.update(value=imgs), None, "Selecione uma imagem da galeria.", f"Deletada: {os.path.basename(p)}"

    ctrl_image.change(fn=set_ctrl_image, inputs=[ctrl_image], outputs=[status_out])

    send_img2img_btn.click(fn=send_to_img2img, inputs=[gallery], outputs=[init_image, status_out])
    send_mask_btn.click(fn=send_to_mask, inputs=[gallery], outputs=[mask_image, status_out])
    send_ctrl_btn.click(fn=send_to_controlnet, inputs=[gallery], outputs=[ctrl_image, status_out])
    civitai_example.change(fn=update_example_url, inputs=[civitai_example], outputs=[civitai_url])
    query_btn.click(fn=query_civitai_click, inputs=[civitai_url, civitai_token, hf_token],
                    outputs=[model_card, version_dropdown, civitai_status])
    civitai_load_btn.click(fn=load_civitai_click,
                           inputs=[civitai_url, civitai_token, hf_token, version_dropdown, force_comfy, quant_method, use_compile],
                           outputs=[civitai_status, steps, guidance_scale, resolution, aspect_ratio, negative_prompt, use_template, use_trigger, family_info])


    lib_refresh_btn.click(fn=library_refresh, outputs=[lib_dropdown, lib_info])
    load_lib_btn.click(fn=library_load_click, inputs=[lib_dropdown],
                       outputs=[lib_info, steps, guidance_scale, resolution, aspect_ratio, negative_prompt, use_template, use_trigger, family_info])
    delete_lib_btn.click(fn=library_delete_click, inputs=[lib_dropdown], outputs=[lora_status, lib_dropdown, lib_info])
    api_btn.click(fn=set_api_key, inputs=[api_key_box], outputs=[api_info])
    cmp_refresh_btn.click(fn=cmp_refresh, outputs=[cmp_dropdown_a, cmp_dropdown_b])
    cmp_btn.click(fn=compare_click,
                  inputs=[cmp_dropdown_a, cmp_dropdown_b, prompt, negative_prompt, steps, guidance_scale, seed, aspect_ratio, resolution, style_preset, civitai_token, hf_token],
                  outputs=[cmp_gallery, cmp_status])
    hist_refresh_btn.click(fn=history_refresh, outputs=[history_gallery, hist_status])
    history_gallery.select(fn=on_history_select, outputs=[hist_preview, hist_meta_card, hist_status])
    hist_to_img2img_btn.click(fn=send_hist_img2img, outputs=[init_image, status_out])
    hist_to_mask_btn.click(fn=send_hist_mask, outputs=[mask_image, status_out])
    hist_to_ctrl_btn.click(fn=send_hist_ctrl, outputs=[ctrl_image, status_out])
    hist_delete_btn.click(fn=delete_hist_image, outputs=[history_gallery, hist_preview, hist_meta_card, hist_status])
    enhance_btn.click(fn=lambda p, s: _enhance(p, s), inputs=[prompt, style_preset], outputs=[prompt])
    gen_btn.click(fn=generate_ui,
                  inputs=[prompt, negative_prompt, steps, aspect_ratio, resolution, seed, num_images,
                          guidance_scale, style_preset, use_template, use_trigger, init_image, mask_image,
                          denoise, hires_fix, hires_denoise, hires_scale, prompt_matrix],
                  outputs=[gallery, status_out, zip_out])
    clear_btn.click(
        fn=lambda: ("", "", 25, "1:1 Square", "768px (Balanced)", -1, 1, 6.5, "None", True, True, None, None, 0.7,
                    False, 0.45, "2x", False,
                    gr.update(value=None, selected_index=None), "", gr.update(visible=False)),
        outputs=[prompt, negative_prompt, steps, aspect_ratio, resolution, seed, num_images,
                 guidance_scale, style_preset, use_template, use_trigger, init_image, mask_image,
                 denoise, hires_fix, hires_denoise, hires_scale, prompt_matrix,
                 gallery, status_out, zip_out])

    lora_comp_refresh.click(fn=components_status, outputs=[lora_status])
    vae_comp_refresh.click(fn=components_status, outputs=[vae_status])
    ti_comp_refresh.click(fn=components_status, outputs=[ti_status])
    lora_example.change(fn=lambda c: _ex_url_aux(c, LORA_EXAMPLES), inputs=[lora_example], outputs=[lora_url])
    lora_query_btn.click(fn=lambda u, t: query_aux_click(u, t, "LoRA"), inputs=[lora_url, civitai_token], outputs=[lora_card, lora_ver_dd, lora_status])
    lora_dl_btn.click(fn=lora_load_click, inputs=[lora_url, civitai_token, lora_ver_dd, lora_weight], outputs=[lora_status])
    lora_dl_btn.click(fn=lora_active_refresh, outputs=[lora_active_dd])
    clear_lora_btn2.click(fn=clear_lora2, outputs=[lora_status, lora_active_dd])
    lora_rm_btn.click(fn=lora_rm_click, inputs=[lora_active_dd], outputs=[lora_status, lora_active_dd])
    lora_lib_refresh.click(fn=lambda: lib_refresh_aux("lora"), outputs=[lora_lib_dd])
    lora_del_btn.click(fn=lib_del_aux, inputs=[lora_lib_dd], outputs=[lora_status])
    lora_lib_load.click(fn=lora_lib_load_click, inputs=[lora_lib_dd, lora_weight], outputs=[lora_status])
    lora_lib_load.click(fn=lora_active_refresh, outputs=[lora_active_dd])
    vae_example.change(fn=lambda c: _ex_url_aux(c, VAE_EXAMPLES), inputs=[vae_example], outputs=[vae_url])
    vae_query_btn.click(fn=lambda u, t: query_aux_click(u, t, "VAE"), inputs=[vae_url, civitai_token], outputs=[vae_card, vae_ver_dd, vae_status])
    vae_dl_btn.click(fn=vae_load_click, inputs=[vae_url, civitai_token, vae_ver_dd], outputs=[vae_status])
    vae_lib_refresh.click(fn=lambda: lib_refresh_aux("vae"), outputs=[vae_lib_dd])
    vae_del_btn.click(fn=lib_del_aux, inputs=[vae_lib_dd], outputs=[vae_status])
    vae_lib_load.click(fn=vae_lib_load_click, inputs=[vae_lib_dd], outputs=[vae_status])
    ti_example.change(fn=lambda c: _ex_url_aux(c, TI_EXAMPLES), inputs=[ti_example], outputs=[ti_url])
    ti_query_btn.click(fn=lambda u, t: query_aux_click(u, t, "TextualInversion"), inputs=[ti_url, civitai_token], outputs=[ti_card, ti_ver_dd, ti_status])
    ti_dl_btn.click(fn=ti_load_click, inputs=[ti_url, civitai_token, ti_ver_dd], outputs=[ti_status])
    ti_lib_refresh.click(fn=lambda: lib_refresh_aux("ti"), outputs=[ti_lib_dd])
    ti_del_btn.click(fn=lib_del_aux, inputs=[ti_lib_dd], outputs=[ti_status])
    ti_lib_load.click(fn=ti_lib_load_click, inputs=[ti_lib_dd], outputs=[ti_status])
if __name__ == "__main__":
    print("Launching Advanced Multi-Model Image Studio v2.1 (" + APP_VER + ")...")
    start_api_server(API_PORT)
    demo.queue()
    demo.launch(share=True, inline=False, debug=False, show_error=True)
