# -*- coding: utf-8 -*-
"""EVOLUCAO: rodar checkpoints Krea-2 CUSTOMIZADOS do Civitai na T4.
Causa do OOM antigo: pick_file_from_version escolhia o arquivo de MAIOR tamanho
(= bf16 23.9GB do Arthemy Krea2 v1.1) — impossivel em 12GB RAM/16GB VRAM.
Fix: picker por FORMATO (int8 > fp8 > fp8_mixed; bf16 recusado com aviso) +
routing do guard para o fluxo worker (download + spawn isolado)."""
import sys, os, re

APP = "universal_app.py"
src = open(APP, encoding="utf-8").read()

# ===========================================================================
# 1. APP_VER bump
# ===========================================================================
src = src.replace('APP_VER = "v2.4.20260817"', 'APP_VER = "v2.5.20260817"')

# ===========================================================================
# 2. BASE_MODEL_MAP: re-adicionar "Krea 2" (se ausente)
# ===========================================================================
if '"Krea 2": "krea2"' not in src:
    src = src.replace('"FLUX.2 Klein 4B-base": "flux2_klein",',
                      '"FLUX.2 Klein 4B-base": "flux2_klein", "Krea 2": "krea2",')
    print("[OK] BASE_MODEL_MAP Krea 2")

# ===========================================================================
# 3. _pick_krea_file + load_krea_custom (inseridos antes de def load_krea_click)
# ===========================================================================
anchor = src.find("def load_krea_click")
assert anchor != -1, "load_krea_click nao encontrado"
new_funcs = '''def _pick_krea_file(version):
    """Escolhe o melhor arquivo Krea-2 para a T4 por FORMATO (nao por tamanho):
    prioridade int8 > fp8/fp8_mixed; bf16 (>20GB) recusado com aviso claro.
    resolve o OOM: o picker antigo escolhia o primary bf16 de 23.9GB."""
    files = [f for f in version.get("files", [])
             if (f.get("type") or "") in ("Model", "Diffusion Model")]
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
        safe_name = re.sub(r"[^\\w\\-.]", "_", file_name)
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


'''
src = src[:anchor] + new_funcs + src[anchor:]

# ===========================================================================
# 4. Guard: routing para load_krea_custom (em vez de bloquear)
# ===========================================================================
old_guard = '''        if any(k in str(base_model).lower() for k in ["krea 2", "krea2", "krea-2"]):
            return "Checkpoints Krea-2 customizados do Civitai exigem +30GB RAM. Para rodar Krea-2 na T4, use o botao \\"Krea-2-Turbo\\" (modelo oficial INT8) na aba Modelo.", None'''
new_guard = '''        if any(k in str(base_model).lower() for k in ["krea 2", "krea2", "krea-2"]):
            # EVOLUCAO v2.5: roteia checkpoints Krea-2 customizados para o fluxo
            # worker com selecao inteligente de arquivo (int8/fp8, nunca bf16 23.9GB).
            if progress_cb:
                progress_cb(0.01, 1.0, "Roteando Krea-2 custom (selecao int8/fp8)...")
            return load_krea_custom(url_or_id, progress_cb=progress_cb)'''
assert old_guard in src, "guard antigo nao encontrado"
src = src.replace(old_guard, new_guard)
print("[OK] guard -> routing load_krea_custom")

# ===========================================================================
# 5. download_from_civitai: picker por formato quando familia krea2
# ===========================================================================
old_pick = '''    target = pick_file_from_version(version, wanted_type)'''
new_pick = '''    if family == "krea2" and wanted_type in (None, "Model"):
        target = _pick_krea_file(version)
    else:
        target = pick_file_from_version(version, wanted_type)'''
assert old_pick in src, "pick nao encontrado"
src = src.replace(old_pick, new_pick, 1)
print("[OK] download_from_civitai usa _pick_krea_file para krea2")

# ===========================================================================
# 6. Botao na UI para carregar custom por URL (usar o campo existente de URL)
# ===========================================================================
open(APP, "w", encoding="utf-8").write(src)
print("universal_app.py evoluido: Krea-2 custom na T4 (v2.5.20260817)")
print("tamanho:", len(src), "chars")
