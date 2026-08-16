# -*- coding: utf-8 -*-
import sys, os, re

APP = "universal_app.py"
src = open(APP, encoding="utf-8").read()

# 1. APP_VER bump para v2.2.20260817
src = src.replace('APP_VER = "v2.1.20260816"', 'APP_VER = "v2.2.20260817"')

# 2. Guard no load_model_from_civitai para modelos Krea2 customizados do Civitai (24GB FP8)
old_civitai_routing = '        if backend_used is None and family == "krea2" and not force_comfy:'
new_civitai_routing = '''        if any(k in str(base_model).lower() for k in ["krea 2", "krea2", "krea-2"]):
            return "Checkpoints Krea-2 customizados do Civitai (24GB FP8) exigem +30GB RAM e nao rodam na T4 (12GB RAM). Para usar o Krea-2 na GPU T4 do Colab, utilize o modelo oficial Krea-2-Turbo INT8 (disponivel no botao na aba Modelo).", None

        if backend_used is None and family == "krea2" and not force_comfy:'''
assert old_civitai_routing in src, "old_civitai_routing not found"
src = src.replace(old_civitai_routing, new_civitai_routing)

# 3. Atualizar exemplo do Civitai para incluir o aviso
old_ex = '    ("Arthemy Western Art", "https://civitai.com/models/2241572"),'
new_ex = '    ("Arthemy Western Art", "https://civitai.com/models/2241572"),\n    ("Arthemy Comics Krea2 (Aviso: +30GB RAM)", "https://civitai.com/models/2759057"),'
if old_ex in src:
    src = src.replace(old_ex, new_ex)

with open(APP, "w", encoding="utf-8") as f:
    f.write(src)

print("universal_app.py atualizado com v2.2.20260817! Tamanho:", len(src), "chars")
EOF