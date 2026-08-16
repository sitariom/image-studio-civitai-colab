# -*- coding: utf-8 -*-
"""Remocao DEFINITIVA e PERMANENTE de todas as implementacoes Krea-2/Wan2GP.
v2.3.20260817 — arquitetura final: 2 motores (Diffusers + ComfyUI)."""
import sys, os, re

APP = "universal_app.py"
src = open(APP, encoding="utf-8").read()
orig_len = len(src)

def remove_between(start_anchor, end_anchor, label):
    """Remove do inicio de start_anchor (inclusive) ate o inicio de end_anchor (exclusive)."""
    global src
    i = src.find(start_anchor)
    j = src.find(end_anchor, i + 1)
    assert i != -1 and j != -1 and i < j, f"boundary {label} nao encontrada"
    src = src[:i] + src[j:]
    print(f"  [OK] removido bloco {label} ({j - i} chars)")

def replace_once(old, new, label):
    global src
    assert old in src, f"replace {label} nao encontrado"
    src = src.replace(old, new, 1)
    print(f"  [OK] {label}")

# ---------------------------------------------------------------------------
# 1. APP_VER bump
# ---------------------------------------------------------------------------
replace_once('APP_VER = "v2.2.20260817"', 'APP_VER = "v2.3.20260817"', "APP_VER v2.3.20260817")

# ---------------------------------------------------------------------------
# 2. Constantes globais
# ---------------------------------------------------------------------------
replace_once('COMFY_DIR = "/content/ComfyUI"\nWAN2GP_DIR = "/content/Wan2GP"\nAPI_PORT',
             'COMFY_DIR = "/content/ComfyUI"\nAPI_PORT', "WAN2GP_DIR")

# ---------------------------------------------------------------------------
# 3. BASE_MODEL_MAP — remover "Krea 2" (manter FLUX.1 Krea -> flux_dev)
# ---------------------------------------------------------------------------
replace_once('"FLUX.2 Klein 4B-base": "flux2_klein", "Krea 2": "krea2",',
             '"FLUX.2 Klein 4B-base": "flux2_klein",', "BASE_MODEL_MAP Krea 2")

# ---------------------------------------------------------------------------
# 4. FAMILY_PRESETS — remover preset krea2
# ---------------------------------------------------------------------------
preset_krea2 = '''    "krea2": {
        "label": "Krea-2 (Wan2GP)", "diffusers_cls": None, "single_file": False,
        "backend": "wan2gp", "steps": {"min": 1, "max": 20, "default": 8}, "cfg": 3.5,
        "scheduler": "krea", "sampler": "krea",
        "resolutions": ["1024px (Standard)", "1536px (High)"],
        "default_res": "1024px (Standard)",
        "aspects": ["16:9 Landscape", "9:16 Portrait", "1:1 Square", "4:3 Standard", "3:4 Portrait"],
        "default_aspect": "16:9 Landscape",
        "prompt_prefix": "", "prompt_suffix": "", "neg_prefix": "",
        "comfy_template": "none", "controlnet": None, "cfg_rescale": 0.0,
        "notes": "Krea-2-Turbo via Wan2GP INT8 (mantido).",
    },
'''
replace_once(preset_krea2, "", "preset krea2 FAMILY_PRESETS")

# ---------------------------------------------------------------------------
# 5. STATE init — remover krea_model
# ---------------------------------------------------------------------------
replace_once('"krea_model": None, "loaded": False,', '"loaded": False,', "STATE krea_model")

# ---------------------------------------------------------------------------
# 6. Remover Seção 7 (MOTOR WAN2GP) inteira — worker + helpers + downloads + loads
# ---------------------------------------------------------------------------
sec7_start = src.find("# ============================================================================\n# 7. MOTOR WAN2GP")
sec8_start = src.find("# ============================================================================\n# 8. MOTOR COMFYUI")
assert sec7_start != -1 and sec8_start != -1 and sec7_start < sec8_start
src = src[:sec7_start] + src[sec8_start:]
print("  [OK] removida Seção 7 MOTOR WAN2GP inteira")

# ---------------------------------------------------------------------------
# 7. Remover _wan2gp_result_to_image + _gen_wan2gp
# ---------------------------------------------------------------------------
wan_start = src.find("def _wan2gp_result_to_image(result):")
mk_cb_start = src.find("def _mk_cb(cb):")
assert wan_start != -1 and mk_cb_start != -1 and wan_start < mk_cb_start
src = src[:wan_start] + src[mk_cb_start:]
print("  [OK] removido _wan2gp_result_to_image + _gen_wan2gp")

# ---------------------------------------------------------------------------
# 8. Branch wan2gp na geracao
# ---------------------------------------------------------------------------
branch_wan = '''    if backend == "wan2gp":
        if init_image is not None:
            raise RuntimeError("Krea-2 nao suporta img2img.")
        return _gen_wan2gp(prompt, negative, steps, width, height, cfg, seed, progress_cb)
'''
replace_once(branch_wan, "", "branch wan2gp na geracao")

# ---------------------------------------------------------------------------
# 9. unload_current_model — remover worker/krea_model
# ---------------------------------------------------------------------------
unload_old = '''def unload_current_model():
    # NOVO: derruba o worker Krea-2 isolado (se ativo) antes de limpar o estado
    w = STATE.get("krea_worker")
    if w:
        try:
            requests.post("http://127.0.0.1:" + str(w.get("port") or KREA2_WORKER_PORT) + "/unload", timeout=5)
        except Exception:
            pass
        _kill_krea_worker()
    for key in ["pipe", "krea_model"]:
        if STATE.get(key) is not None:
            try:
                del STATE[key]
            except Exception:
                pass
    STATE["pipe"] = None
    STATE["krea_model"] = None'''
unload_new = '''def unload_current_model():
    for key in ["pipe"]:
        if STATE.get(key) is not None:
            try:
                del STATE[key]
            except Exception:
                pass
    STATE["pipe"] = None'''
replace_once(unload_old, unload_new, "unload_current_model")

# ---------------------------------------------------------------------------
# 10. load_model_from_civitai — guard + rota wan2gp
# ---------------------------------------------------------------------------
# Guard atualizado para mensagem definitiva
guard_old = 'if any(k in str(base_model).lower() for k in ["krea 2", "krea2", "krea-2"]):\n            return "Checkpoints Krea-2 customizados do Civitai (24GB FP8) exigem +30GB RAM e nao rodam na T4 (12GB RAM). Para usar o Krea-2 na GPU T4 do Colab, utilize o modelo oficial Krea-2-Turbo INT8 (disponivel no botao na aba Modelo).", None'
guard_new = 'if any(k in str(base_model).lower() for k in ["krea 2", "krea2", "krea-2"]):\n            return "O suporte ao modelo Krea-2 foi DESCONTINUADO permanentemente (causava desconexao por OOM no Colab). Use SDXL, Pony, Illustrious, NoobAI, Anima ou FLUX.", None'
replace_once(guard_old, guard_new, "guard Krea-2 definitivo (civitai)")

# Rota wan2gp (custom do civitai) — remover
civitai_route_old = '''        if backend_used is None and family == "krea2" and not force_comfy:
            # Krea2 (DiT Wan-based, sem CLIP/VAE no safetensors): rota Wan2GP com o
            # checkpoint custom + TE/VAE base. Se falhar, faz fallback para motor ComfyUI.
            try:
                if progress_cb:
                    progress_cb(0.2, 1, "Usando motor Wan2GP (Krea2)...")
                _load_wan2gp_custom(local, base_model, model_name, progress_cb=progress_cb)
                backend_used = "wan2gp"
            except Exception as e:
                error_log.append("wan2gp: " + str(e)[:200])
                unload_current_model()
                print("  Wan2GP custom falhou (" + str(e)[:150] + ") — Krea2 so e suportado via Wan2GP (motor ComfyUI nao suporta Qwen-Image).")
                return "Falha ao carregar Krea2 via Wan2GP: " + str(e)[:200] + ". O motor ComfyUI nao suporta Krea2 (modelo Qwen-Image, nao FLUX).", None
'''
replace_once(civitai_route_old, "", "rota wan2gp custom no load civitai")

# ---------------------------------------------------------------------------
# 11. load_krea / load_krea_click — mensagem definitiva
# ---------------------------------------------------------------------------
load_krea_old = src[src.find("def load_krea(progress_cb=None):"):src.find("def load_krea_click")]
assert "load_krea" in load_krea_old, "load_krea_old nao achado"
load_krea_new = '''def load_krea(progress_cb=None):
    return "O suporte ao modelo Krea-2 foi DESCONTINUADO permanentemente (causava desconexao por OOM no Colab). Use SDXL, Pony, Illustrious, NoobAI, Anima ou FLUX.", None

'''
src = src.replace(load_krea_old, load_krea_new)
print("  [OK] load_krea com mensagem definitiva")

# ---------------------------------------------------------------------------
# 12. load_local_file — rota krea2
# ---------------------------------------------------------------------------
local_krea = '''        if family == "krea2" and not force_comfy:
            try:
                _load_wan2gp_custom(file_path, base_model or "Krea 2", os.path.basename(file_path), progress_cb=progress_cb)
                STATE["trained_words"] = list(trained)
                STATE["model_path"] = file_path
                return "Modelo local Krea2 carregado via Wan2GP.", family
            except Exception as e:
                unload_current_model()
                print("Wan2GP krea2 falhou: " + str(e)[:200])
                return "Falha ao carregar Krea2 via Wan2GP: " + str(e), ""
'''
replace_once(local_krea, "", "rota krea2 no load_local_file")

# ---------------------------------------------------------------------------
# 13. Health API — remover krea_worker
# ---------------------------------------------------------------------------
replace_once('                                          "krea_worker": bool(STATE.get("krea_worker")),\n', '', "health krea_worker")

# ---------------------------------------------------------------------------
# 14. neg_rec krea2
# ---------------------------------------------------------------------------
replace_once('    elif not neg_rec and family in ("krea2",):\n        neg_rec = "worst quality, low quality, blurry, distorted, watermark, signature"\n', '', "neg_rec krea2")

# ---------------------------------------------------------------------------
# 15. CIVITAI_EXAMPLES — remover 2 linhas Krea2
# ---------------------------------------------------------------------------
replace_once('    ("Arthemy Comics Krea2 (Aviso: +30GB RAM)", "https://civitai.com/models/2759057"),\n', '', "exemplo Krea2 (aviso)")
replace_once('    ("Arthemy Comics Krea2", "https://civitai.com/models/2759057"),\n', '', "exemplo Krea2")

# ---------------------------------------------------------------------------
# 16. Subtitle — remover Wan2GP
# ---------------------------------------------------------------------------
replace_once('Diffusers + ComfyUI + Wan2GP', 'Diffusers + ComfyUI', "subtitle")

# ---------------------------------------------------------------------------
# 17. UI: remover botao/status Krea-2-Turbo
# ---------------------------------------------------------------------------
ui_old = '''            gr.Markdown("### Krea-2-Turbo (Wan2GP INT8)")
            krea_btn = gr.Button("Baixar e Carregar Krea-2-Turbo", variant="secondary")
            krea_status = gr.Textbox(label="Status Krea", lines=3)'''
replace_once(ui_old, '', "UI botao Krea-2-Turbo")

# remover o .click do krea_btn
click_old = '''    krea_btn.click(fn=load_krea_click,
                   outputs=[krea_status, steps, guidance_scale, resolution, aspect_ratio, negative_prompt, use_template, use_trigger, family_info])'''
replace_once(click_old, '', "UI krea_btn.click")

# ---------------------------------------------------------------------------
# 18. Docstring topo do arquivo
# ---------------------------------------------------------------------------
replace_once('Motor triplo: DIFFUSERS (nativo) + COMFYUI (fallback universal) + WAN2GP (Krea-2)',
             'Motor duplo: DIFFUSERS (nativo) + COMFYUI (fallback universal)', "docstring topo")

with open(APP, "w", encoding="utf-8") as f:
    f.write(src)

print(f"\n=== REMOCAO DEFINITIVA CONCLUIDA ===")
print(f"Tamanho: {orig_len} -> {len(src)} chars (removidos {orig_len - len(src)})")
