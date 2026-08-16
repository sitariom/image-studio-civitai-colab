# -*- coding: utf-8 -*-
"""Evolucao: gestao unificada LoRA/VAE/TI com o mesmo padrao dos checkpoints.
- Exemplos populares (Civitai) por tipo
- Consultar (card + trigger words) + versoes (mais recente primeiro)
- Baixar versao especifica + ativar/aplicar/carregar
- Biblioteca por tipo (listar/deletar) + gestao de LoRAs ativos (remover)
"""
import re

APP = "universal_app.py"
src = open(APP, encoding="utf-8").read()

# ===========================================================================
# 1. BACKEND: exemplos + funcoes de gestao (inseridos antes de load_lora_from_civitai)
# ===========================================================================
anchor = src.find("def load_lora_from_civitai(url_or_id, token, weight=1.0):")
assert anchor != -1, "anchor lora"

backend = '''LORA_EXAMPLES = [
    ("Detail Tweaker XL (SDXL)", "https://civitai.com/models/122359"),
    ("Add More Details (SD1.5/SDXL)", "https://civitai.com/models/82098"),
    ("Detail Tweaker LoRA", "https://civitai.com/models/58390"),
    ("Not Artists Styles for Pony", "https://civitai.com/models/264290"),
    ("blindbox", "https://civitai.com/models/25995"),
    ("Hands XL + SD1.5 + Pony + F1D", "https://civitai.com/models/200255"),
    ("Doll Likeness - EDG", "https://civitai.com/models/42903"),
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
    card = ("## " + str(data.get("name", "?")) + " (" + str(wanted_type) + ")\\n\\n"
            + "**Tipo:** " + str(wanted_type) + " | **Versoes:** " + str(len(versions)) + "\\n"
            + "**Trigger words (ultima versao):** `" + tw + "`\\n\\n"
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


def lora_active_choices():
    return [os.path.basename(p) + " (w=" + str(w) + ")" for (p, w) in (STATE.get("lora_stack") or [])]


'''
src = src[:anchor] + backend + src[anchor:]
print("[OK] backend: exemplos + aux_query/aux_download/list_aux/delete/lora_remove/lora_active")

# ===========================================================================
# 2. load_lora_from_civitai / load_vae / load_ti: suportar version_index
# ===========================================================================
old_lora = '''def load_lora_from_civitai(url_or_id, token, weight=1.0):
    """Baixa e ATIVA o LoRA automaticamente (versao mais recente)."""
    try:
        model_id = parse_civitai_id(url_or_id)
        if not model_id:
            return "URL/ID de LoRA invalido."
        local, base, name, family, tw = download_from_civitai(model_id, token, "LoRA")'''
new_lora = '''def load_lora_from_civitai(url_or_id, token, weight=1.0, version_index=0):
    """Baixa e ATIVA o LoRA (versao especifica; 0 = mais recente)."""
    try:
        model_id = parse_civitai_id(url_or_id)
        if not model_id:
            return "URL/ID de LoRA invalido."
        idx = int(version_index or 0)
        local, base, name, family, tw = download_from_civitai(model_id, token, "LoRA", idx, None, use_latest=(idx == 0))'''
assert old_lora in src, "load_lora"
src = src.replace(old_lora, new_lora)

old_vae = '''def load_vae_from_civitai(url_or_id, token):
    try:
        model_id = parse_civitai_id(url_or_id)
        if not model_id:
            return "URL/ID de VAE invalido."
        local, base, name, family, tw = download_from_civitai(model_id, token, "VAE")'''
new_vae = '''def load_vae_from_civitai(url_or_id, token, version_index=0):
    try:
        model_id = parse_civitai_id(url_or_id)
        if not model_id:
            return "URL/ID de VAE invalido."
        idx = int(version_index or 0)
        local, base, name, family, tw = download_from_civitai(model_id, token, "VAE", idx, None, use_latest=(idx == 0))'''
assert old_vae in src, "load_vae"
src = src.replace(old_vae, new_vae)

old_ti = '''def load_ti_from_civitai(url_or_id, token):
    """TextualInversion (P3-24): baixa e registra; uso via <nome> ou embedding:nome."""
    try:
        model_id = parse_civitai_id(url_or_id)
        if not model_id:
            return "URL/ID de TextualInversion invalido."
        local, base, name, family, tw = download_from_civitai(model_id, token, "TextualInversion")'''
new_ti = '''def load_ti_from_civitai(url_or_id, token, version_index=0):
    """TextualInversion (P3-24): baixa e registra; uso via <nome> ou embedding:nome."""
    try:
        model_id = parse_civitai_id(url_or_id)
        if not model_id:
            return "URL/ID de TextualInversion invalido."
        idx = int(version_index or 0)
        local, base, name, family, tw = download_from_civitai(model_id, token, "TextualInversion", idx, None, use_latest=(idx == 0))'''
assert old_ti in src, "load_ti"
src = src.replace(old_ti, new_ti)
print("[OK] load_lora/vae/ti suportam version_index")

# ===========================================================================
# 3. UI Modelo: substituir bloco antigo LoRA/VAE/TI por ponteiro para a nova aba
# ===========================================================================
old_ui = '''            gr.Markdown("### LoRA / VAE / TextualInversion (Civitai) — baixa e ativa automaticamente")
            with gr.Row():
                lora_url = gr.Textbox(label="URL/ID do LoRA", placeholder="https://civitai.com/models/xxx")
                lora_weight = gr.Slider(0.0, 2.0, 1.0, step=0.05, label="Peso")
            with gr.Row():
                lora_btn = gr.Button("⬇️ Baixar e Ativar LoRA", variant="secondary")
                clear_lora_btn = gr.Button("Remover LoRAs", variant="secondary")
                vae_url = gr.Textbox(label="URL/ID do VAE", placeholder="https://civitai.com/models/xxx")
                vae_btn = gr.Button("Aplicar VAE", variant="secondary")
            with gr.Row():
                ti_url = gr.Textbox(label="URL/ID do TextualInversion", placeholder="https://civitai.com/models/xxx")
                ti_btn = gr.Button("Carregar TI", variant="secondary")
            lora_status = gr.Textbox(label="Status LoRA/VAE/TI", lines=3)'''
new_ui = '''            gr.Markdown("### LoRA / VAE / TextualInversion")
            gr.Markdown("👉 Gestao completa (exemplos, versoes, consultar, baixar, biblioteca) na aba **LoRA / VAE / TI**.")'''
assert old_ui in src, "old_ui"
src = src.replace(old_ui, new_ui)
print("[OK] bloco antigo da aba Modelo substituido")

# ===========================================================================
# 4. UI: nova aba "LoRA / VAE / TI" antes de Historico
# ===========================================================================
anchor_hist = src.find('        with gr.TabItem("🖼️ Histórico & Galeria"):')
assert anchor_hist != -1, "anchor historico"
nova_aba = '''        with gr.TabItem("LoRA / VAE / TI"):
            gr.Markdown("### Gestao unificada — LoRA / VAE / TextualInversion (mesmo padrao dos checkpoints)")
            with gr.Tabs():
                with gr.TabItem("LoRA"):
                    with gr.Row():
                        lora_example = gr.Dropdown(label="Exemplos populares de LoRA", choices=[""] + [e[0] for e in LORA_EXAMPLES], value="")
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
                        lora_lib_refresh = gr.Button("🔄 Biblioteca LoRA", variant="secondary")
                        lora_lib_dd = gr.Dropdown(label="LoRAs baixados (biblioteca)", choices=[], value=None, scale=2)
                        lora_del_btn = gr.Button("🗑️ Deletar", variant="stop")
                    lora_status = gr.Textbox(label="Status LoRA", lines=2)
                with gr.TabItem("VAE"):
                    with gr.Row():
                        vae_example = gr.Dropdown(label="Exemplos populares de VAE", choices=[""] + [e[0] for e in VAE_EXAMPLES], value="")
                    vae_url = gr.Textbox(label="URL ou ID do VAE no Civitai", placeholder="https://civitai.com/models/296576 ou 296576")
                    with gr.Row():
                        vae_query_btn = gr.Button("🔎 Consultar VAE", variant="secondary")
                        vae_ver_dd = gr.Dropdown(label="Versao (mais recente primeiro)", choices=[], value=None, scale=2)
                    vae_card = gr.Markdown("")
                    vae_dl_btn = gr.Button("⬇️ Baixar e Aplicar VAE", variant="primary")
                    with gr.Row():
                        vae_lib_refresh = gr.Button("🔄 Biblioteca VAE", variant="secondary")
                        vae_lib_dd = gr.Dropdown(label="VAEs baixados (biblioteca)", choices=[], value=None, scale=2)
                        vae_del_btn = gr.Button("🗑️ Deletar", variant="stop")
                    vae_status = gr.Textbox(label="Status VAE", lines=2)
                with gr.TabItem("TextualInversion"):
                    with gr.Row():
                        ti_example = gr.Dropdown(label="Exemplos populares de TI", choices=[""] + [e[0] for e in TI_EXAMPLES], value="")
                    ti_url = gr.Textbox(label="URL ou ID do TI no Civitai", placeholder="https://civitai.com/models/7808 ou 7808")
                    with gr.Row():
                        ti_query_btn = gr.Button("🔎 Consultar TI", variant="secondary")
                        ti_ver_dd = gr.Dropdown(label="Versao (mais recente primeiro)", choices=[], value=None, scale=2)
                    ti_card = gr.Markdown("")
                    ti_dl_btn = gr.Button("⬇️ Baixar e Carregar TI", variant="primary")
                    with gr.Row():
                        ti_lib_refresh = gr.Button("🔄 Biblioteca TI", variant="secondary")
                        ti_lib_dd = gr.Dropdown(label="TIs baixados (biblioteca)", choices=[], value=None, scale=2)
                        ti_del_btn = gr.Button("🗑️ Deletar", variant="stop")
                    ti_status = gr.Textbox(label="Status TI", lines=2)

'''
src = src[:anchor_hist] + nova_aba + src[anchor_hist:]
print("[OK] nova aba LoRA/VAE/TI inserida")

# ===========================================================================
# 5. Handlers (substituir os antigos + adicionar novos antes de if __name__)
# ===========================================================================
old_handlers = '''    lora_btn.click(fn=load_lora_from_civitai, inputs=[lora_url, civitai_token, lora_weight], outputs=[lora_status])
    clear_lora_btn.click(fn=clear_lora, outputs=[lora_status])
    vae_btn.click(fn=load_vae_from_civitai, inputs=[vae_url, civitai_token], outputs=[lora_status])
    ti_btn.click(fn=load_ti_from_civitai, inputs=[ti_url, civitai_token], outputs=[lora_status])'''
assert old_handlers in src, "old_handlers"
src = src.replace(old_handlers, "")

new_handlers = '''
    # ---- Handlers da aba LoRA / VAE / TI ----
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

    lora_example.change(fn=lambda c: _ex_url_aux(c, LORA_EXAMPLES), inputs=[lora_example], outputs=[lora_url])
    lora_query_btn.click(fn=query_aux_click, inputs=[lora_url, civitai_token], outputs=[lora_card, lora_ver_dd, lora_status])
    lora_dl_btn.click(fn=lora_load_click, inputs=[lora_url, civitai_token, lora_ver_dd, lora_weight], outputs=[lora_status])
    lora_dl_btn.click(fn=lora_active_refresh, outputs=[lora_active_dd])
    clear_lora_btn2.click(fn=clear_lora2, outputs=[lora_status, lora_active_dd])
    lora_rm_btn.click(fn=lora_rm_click, inputs=[lora_active_dd], outputs=[lora_status, lora_active_dd])
    lora_lib_refresh.click(fn=lib_refresh_aux, inputs=[gr.State("lora")], outputs=[lora_lib_dd])
    lora_del_btn.click(fn=lib_del_aux, inputs=[lora_lib_dd], outputs=[lora_status])

    vae_example.change(fn=lambda c: _ex_url_aux(c, VAE_EXAMPLES), inputs=[vae_example], outputs=[vae_url])
    vae_query_btn.click(fn=query_aux_click, inputs=[vae_url, civitai_token], outputs=[vae_card, vae_ver_dd, vae_status])
    vae_dl_btn.click(fn=vae_load_click, inputs=[vae_url, civitai_token, vae_ver_dd], outputs=[vae_status])
    vae_lib_refresh.click(fn=lib_refresh_aux, inputs=[gr.State("vae")], outputs=[vae_lib_dd])
    vae_del_btn.click(fn=lib_del_aux, inputs=[vae_lib_dd], outputs=[vae_status])

    ti_example.change(fn=lambda c: _ex_url_aux(c, TI_EXAMPLES), inputs=[ti_example], outputs=[ti_url])
    ti_query_btn.click(fn=query_aux_click, inputs=[ti_url, civitai_token], outputs=[ti_card, ti_ver_dd, ti_status])
    ti_dl_btn.click(fn=ti_load_click, inputs=[ti_url, civitai_token, ti_ver_dd], outputs=[ti_status])
    ti_lib_refresh.click(fn=lib_refresh_aux, inputs=[gr.State("ti")], outputs=[ti_lib_dd])
    ti_del_btn.click(fn=lib_del_aux, inputs=[ti_lib_dd], outputs=[ti_status])

'''
anchor_main = src.find('if __name__ == "__main__":')
assert anchor_main != -1, "anchor main"
src = src[:anchor_main] + new_handlers + src[anchor_main:]
print("[OK] handlers da nova aba adicionados")

open(APP, "w", encoding="utf-8").write(src)
print("universal_app.py atualizado | tamanho:", len(src), "chars")
