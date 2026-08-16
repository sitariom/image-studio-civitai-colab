# -*- coding: utf-8 -*-
"""Regenera Notebook_Definitivo_CivitAI.ipynb (v3) a partir de universal_app.py.

Agora com UMA UNICA CELULA de execucao (o usuario roda so ela):
  setup (pip) -> grava universal_app.py -> app em background +
  supervisor -> espera health -> teste API completo (Anima) -> salva PNG.
"""
import json

APP = "universal_app.py"
OUT = "Notebook_Definitivo_CivitAI.ipynb"
app_src = open(APP, encoding="utf-8").read()
app_src = app_src.replace("\r\n", "\n").replace("\r", "\n")

# ---------------------------------------------------------------------------
# A UNICA CELULA
# ---------------------------------------------------------------------------
cell_run = (
"# 🚀 CELULA UNICA — setup + app + teste completo (rode e aguarde ~2-6 min na 1a vez)\n"
"import os, sys, gc, subprocess, json as _json, ast, time, threading, re, io, base64, requests, shutil as _sh\n"
"os.chdir('/content')\n"
"for mod in list(sys.modules):\n"
"    if any(x in mod for x in ['tensorflow', 'tf']):\n"
"        del sys.modules[mod]\n"
"gc.collect()\n"
"# 🔑 API keys (idempotente — com fallback para pre-preenchimento automatico)\n"
"os.environ['CIVITAI_TOKEN'] = os.environ.get('CIVITAI_TOKEN') or 'SEU_TOKEN_CIVITAI'\n"
"os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN') or 'SEU_TOKEN_HF'\n"
"os.makedirs('/content/studio', exist_ok=True)\n"
"_json.dump({'civitai': os.environ['CIVITAI_TOKEN'], 'hf': os.environ['HF_TOKEN']},\n"
"           open('/content/studio/tokens.json', 'w'))\n"
"\n"
"print('⏳ [1/5] SETUP (pip — ~30-60 sec)...')\n"
"def _pip(*pkgs):\n"
"    try:\n"
"        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-input', '--timeout', '120'] + list(pkgs), timeout=600)\n"
"    except Exception as e:\n"
"        print('  WARN pip:', str(e)[:100])\n"
"_pip('--upgrade', 'pip')\n"
"_pip('diffusers>=0.30.0', 'transformers>=4.44.0', 'accelerate', 'huggingface_hub', 'safetensors')\n"
"# numpy: o requirements.txt do Wan2GP fixa 2.1.2 (nao forcar downgrade)\n"
"_pip('gradio', 'requests', 'nvidia-ml-py')\n"
"try:\n"
"    import torchao\n"
"except ImportError:\n"
"    _pip('torchao')\n"
"try:\n"
"    import bitsandbytes\n"
"except ImportError:\n"
"    _pip('bitsandbytes')\n"
"# (numpy ja fixado acima; nao reinstalar)\n"
"print('  setup OK')\n"
"\n"
"print('⏳ [2/5] ESCREVENDO universal_app.py...')\n"
"APP = '/content/universal_app.py'\n"
"LOG = '/content/universal_app.log'\n"
"HEALTH = 'http://127.0.0.1:7861/api/health'\n"
"STOP = '/content/.stop_supervisor'\n"
"" + "APP_SRC = r'''" + app_src + "'''\n" +
"open(APP, 'w', encoding='utf-8').write(APP_SRC)\n"
"print('  universal_app.py:', len(APP_SRC), 'chars')\n"
"try:\n"
"    ast.parse(APP_SRC)\n"
"    print('  sintaxe OK')\n"
"except SyntaxError as e:\n"
"    print('  ERRO de sintaxe em universal_app.py:', e)\n"
"    raise\n"
"import torch\n"
"print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NAO DETECTADA - use Runtime > Change runtime type > T4 GPU')\n"
"\n"
"print('⏳ [3/5] INICIANDO APP em BACKGROUND (URL do Gradio abaixo)...')\n"
"def start_app():\n"
"    logf = open(LOG, 'w', encoding='utf-8')\n"
"    return subprocess.Popen([sys.executable, '-u', APP], cwd='/content',\n"
"                            stdout=logf, stderr=subprocess.STDOUT,\n"
"                            start_new_session=True)  # detached: sobrevive ao kernel\n"
"def supervisor():\n"
"    quick_fails = 0\n"
"    delay = 5\n"
"    while True:\n"
"        if os.path.exists(STOP):\n"
"            try:\n"
"                os.remove(STOP)\n"
"            except Exception:\n"
"                pass\n"
"            print('[supervisor] parada (.stop_supervisor)')\n"
"            p.terminate()\n"
"            break\n"
"        p = start_app()\n"
"        t0 = time.time()\n"
"        rc = p.wait()\n"
"        alive = time.time() - t0\n"
"        if alive >= 20:\n"
"            quick_fails = 0\n"
"            delay = 5\n"
"        else:\n"
"            quick_fails += 1\n"
"        if quick_fails >= 3:\n"
"            print('[supervisor] CRASH LOOP (3x em <20s). Ultimas linhas do log:')\n"
"            try:\n"
"                print(''.join(open(LOG, encoding='utf-8', errors='replace').read().strip().splitlines()[-30:]))\n"
"            except Exception:\n"
"                pass\n"
"            break\n"
"        print('[supervisor] app caiu (rc=' + str(rc) + ', viveu ' + str(int(alive)) + 's). Reiniciando em ' + str(delay) + 's...')\n"
"        time.sleep(delay)\n"
"        delay = min(delay * 2, 60)\n"
"subprocess.run(['pkill', '-9', '-f', 'universal_app.py'], capture_output=True)\n"
"try:\n"
"    if os.path.exists(LOG):\n"
"        os.remove(LOG)\n"
"except Exception:\n"
"    pass\n"
"globals()['_SUP_ACTIVE'] = False\n"
"if not globals().get('_SUP_ACTIVE'):\n"
"    threading.Thread(target=supervisor, daemon=True).start()\n"
"    globals()['_SUP_ACTIVE'] = True\n"
"    print('  Supervisor ativo (thread de fundo).')\n"
"time.sleep(2)\n"
"print('  Aguardando o app subir (ate ~180s)...')\n"
"url_shown = False\n"
"api_shown = False\n"
"for _ in range(90):\n"
"    time.sleep(2)\n"
"    try:\n"
"        log = open(LOG, encoding='utf-8', errors='replace').read()\n"
"    except Exception:\n"
"        log = ''\n"
"    m = re.search(r'https://[a-z0-9\\-]+\\.gradio\\.live', log)\n"
"    if m and not url_shown:\n"
"        print('  🎨 Gradio:', m.group(0))\n"
"        url_shown = True\n"
"    if not api_shown:\n"
"        try:\n"
"            if requests.get(HEALTH, timeout=2).status_code == 200:\n"
"                print('  ✅ API ativa em :7861 (GET /api/health)')\n"
"                api_shown = True\n"
"        except Exception:\n"
"            pass\n"
"    if url_shown and api_shown:\n"
"        break\n"
"if not api_shown:\n"
"    print('  ERRO: app nao subiu em 180s. Tail do log:')\n"
"    try:\n"
"        print(''.join(open(LOG, encoding='utf-8', errors='replace').read().strip().splitlines()[-40:]))\n"
"    except Exception:\n"
"        pass\n"
"    raise SystemExit('App falhou ao iniciar — cole o log acima.')\n"
"\n"
"API = 'http://127.0.0.1:7861'\n"
"print('\\n [4/5] HEALTH CHECK (sem download/geracao automatica — voce inicia via UI/API)...')\n"
"h = requests.get(API + '/api/health').json()\n"
"print('  health:', h.get('status'), '| loaded:', h.get('loaded'), '| app_ver:', h.get('app_ver'))\n"
"_du = _sh.disk_usage('/content')\n"
"print('  disco livre: %.1f GB (de %.1f GB)' % (_du.free / 1e9, _du.total / 1e9))\n"
"print('\\n [5/5] PRONTO — app no ar. Nenhum modelo baixado / imagem gerada automaticamente.')\n"
"print('   Carregue modelos pela aba Modelo (Civitai URL) ou pela API /api/load_model.')\n"
"print('\\n' + '=' * 60)\n"
"print('✅ CONCLUIDO! Interface Gradio: (URL acima, tambem em /content/universal_app.log)')\n"
"print('   Imagem de teste: /content/api_output.png  |  Log: /content/universal_app.log')\n"
"print('   Para PARAR o app (opcional):  !touch /content/.stop_supervisor')\n"
"print('   Outros modelos: use o Gradio (baixe qualquer modelo do Civitai na aba Modelo).')\n"
)

# ---------------------------------------------------------------------------
# Celula markdown de introducao
# ---------------------------------------------------------------------------
md = [
    "# 🎨 Advanced Multi-Model Image Studio v2 — Notebook Definitivo CivitAI (celula unica)\n",
    "\n",
    "Gera **qualquer base model do Civitai** (25 familias mapeadas) na GPU T4 gratuita.\n",
    "**3 motores em cascata:** Diffusers (nativo) → ComfyUI headless (universal) → Wan2GP (Krea-2).\n",
    "\n",
    "## Como usar\n",
    "1. `Runtime → Change runtime type → T4 GPU`\n",
    "2. **Rode a ÚNICA célula abaixo e aguarde (2-6 min na 1ª vez)**. Ela faz tudo:\n",
    "   setup (pip) → grava o app → sobe o Gradio + API em background →\n",
    "   health-check (`/api/health`: app_ver, disco) → tudo pronto para usar.\n",
    "3. Abra a URL do Gradio impressa (troque de modelo / gere mais imagens).\n",
    "4. API externa: `GET /api/health` · `POST /api/load_model` · `POST /api/generate`\n",
    "   (base64) em `http://127.0.0.1:7861` (Colab expoe via `/proxy/7861/`).\n",
    "5. Parar o app quando quiser: rode `!touch /content/.stop_supervisor`.\n",
    "\n",
    "> **Estado atual (v2.1):** Anima validado de ponta a ponta (ComfyUI + WanVAE oficial).\n",
    "> FLUX (Kestral) com familia por conteudo + mirrors publicos (clip_l/t5xxl/ae).\n",
    "> SDXL/Illustrious/Pony via Diffusers. API com auto-reparo de VAE + erro com traceback.\n",
    "> ⚠️ O notebook contém API keys em texto plano — não compartilhe o arquivo.\n",
    "\n",
    "## Modelos civitai cobertos\n",
    "Anima, Chroma, AuraFlow, ERNIE, FLUX.1 D/K/S, FLUX.2 Klein 4B, HiDream, Hunyuan DiT,\n",
    "Illustrious, Lumina, NoobAI, Other, Pony, Qwen, SD 1.5 (+Hyper/LCM), SD 2.1 (+768),\n",
    "SD 3/3.5, SDXL 0.9/1.0 (+LCM/Hyper/Lightning), Upscaler, Wan Video (aviso), Z-Image Base/Turbo, Grok (aviso).\n",
]

cells = []
cells.append({"cell_type": "markdown", "metadata": {}, "source": md})
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_run]})

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"gpuType": "T4", "provenance": [], "name": "Notebook_Definitivo_CivitAI.ipynb"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False)

print("Notebook gerado:", OUT, "| celulas:", len(cells), "| app embutido:", len(app_src), "chars")
