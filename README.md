# 🎨 Advanced Multi-Model Image Studio (CivitAI + Colab T4)

[![CI](https://github.com/sitariom/image-studio-civitai-colab/actions/workflows/ci.yml/badge.svg)](https://github.com/sitariom/image-studio-civitai-colab/actions/workflows/ci.yml)

Gerador de imagens **multi-modelo** para Google Colab (GPU T4), com suporte a **24+ famílias** de modelos do Civitai usando **3 motores em cascata**:

1. **Diffusers nativo** — SD 1.5, SDXL, Pony, Illustrious, NoobAI, Animagine XL...
2. **ComfyUI headless** — Anima, FLUX.1 (Kestral/dev/schnell), SD 3/3.5, Chroma, AuraFlow...
3. **Wan2GP (worker isolado)** — **Krea-2-Turbo** (modelo oficial INT8 e checkpoints customizados do Civitai, seleção automática de arquivo int8/fp8)

---

## 🚀 Como usar

1. Abra [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook**
2. Selecione **`Notebook_Definitivo_CivitAI.ipynb`**
3. **Runtime → Change runtime type → T4 GPU**
4. Rode a **única célula** e aguarde (~2-6 min na 1ª vez)
5. Abra a URL do Gradio impressa na saída
6. Na aba **Modelo**: cole uma URL do Civitai (ex.: `https://civitai.com/models/2700278`) ou clique em **"Krea-2-Turbo"** (modelo oficial INT8)
7. Na aba **Gerar**: prompt → Gerar

> **Tokens**: para baixar modelos gated, defina `CIVITAI_TOKEN` / `HF_TOKEN` como variáveis de ambiente no Colab (ou edite o `_gerar_notebook.py` e regenere o notebook).

---

## 🎯 Recursos

| Recurso | Detalhe |
|---|---|
| **Krea-2-Turbo (T4)** | Modelo oficial INT8 (DeepBeepMeep/krea-2) + custom do Civitai — **seleção inteligente de arquivo** (int8 > fp8; recusa bf16 de 24GB que causa OOM) |
| **Worker isolado** | O Krea-2 roda em processo Python limpo (`krea2_worker.py`) — se OOM matar, só o worker morre, a sessão sobrevive |
| **59 modelos de exemplo** | Dropdown com checkpoints populares (Arthemy, Kestral, A-Zovya, D&D...) |
| **Hires fix** | 2 passos (low-res → refine) |
| **CFG rescale + negativas** | Por família (Pony `score_9...`, Illustrious/NoobAI tags...) |
| **Inpainting / ControlNet** | Via ComfyUI universal |
| **PNGInfo** | Prompt/neg/steps/CFG/seed embutidos no PNG |
| **API HTTP externa** | `:7861` — `/api/health`, `/api/load_model`, `/api/generate` (base64) |
| **VRAM auto-budget** | Clamp de resolução por GPU (pynvml) |
| **Download paralelo** | aria2c (`-x16`) + resume + verificação de tamanho |
| **Biblioteca local** | Troca de modelo sem re-download |
| **Gestão LoRA/VAE/TI** | Aba dedicada com exemplos populares, consulta (versões + trigger words), baixar/ativar por versão, biblioteca por tipo, LoRAs ativos com remoção |

---

## 📁 Arquivos principais

| Arquivo | Função |
|---|---|
| `universal_app.py` | O app completo (v2.5.20260817) — fonte da verdade |
| `krea2_worker.py` | Worker isolado do Krea-2 (embutido no app via `KREA2_WORKER_SRC`) |
| `Notebook_Definitivo_CivitAI.ipynb` | Notebook de célula única para o Colab |
| `_gerar_notebook.py` | Regenera o notebook a partir do `universal_app.py` |
| `ANALISE_E_PLANO.md` | Documentação técnica completa (ADRs/hotfixes §8.1–§8.50) |

**Regra de ouro**: toda alteração no `universal_app.py` → rode `python3 _gerar_notebook.py` (o notebook é sempre regenerado, nunca editado à mão).

---

## 🧠 Arquitetura do Krea-2 (por que funciona na T4)

O Krea-2 causava desconexões (OOM memcg 12Gi) por 3 motivos que foram resolvidos:

1. **Seleção de arquivo por formato** — o picker antigo escolhia o primary **bf16 de 23.9GB** (impossível em 12GB RAM). Agora escolhe **int8/fp8** automaticamente.
2. **Versões exatas do stack** (validadas pelo notebook funcional):
   ```
   mmgp==3.7.12 · optimum-quanto==0.2.7 · gradio==5.29.0 · numpy==2.1.2
   transformers==4.54.0 · diffusers==0.36.0 · smplfitter==0.2.10
   ```
3. **Isolamento total** — o carregamento roda em `python -u krea2_worker.py` (processo limpo, TFBlocked, HTTP local :7862, progresso por arquivo).

Fluxo: **consulta Civitai → versão → arquivo int8/fp8 → download → worker isolado → proxy de geração**.

---

## 🔄 Manutenção / Versionamento

- **APP_VER** em `universal_app.py` — incrementar a cada correção (visível em `/api/health`).
- Alterações → `python3 _gerar_notebook.py` → conferir `roundtrip embedded == disk` → commit.
- Convenção de commit: `fix:`, `feat:`, `chore:`, `docs:` (Conventional Commits).

---

## ⚠️ Avisos

- **Segredos**: este repo é público — NÃO commite tokens. O `_local_secrets_backup/` e `backups/` são ignorados pelo `.gitignore`.
- Modelos customizados Krea-2 do Civitai em **bf16 (24GB)** não rodam na T4 (requer +30GB RAM).
- GGUF (Q8_0/Q4_0) não é suportado pelo Wan2GP.

---

## 📜 Licença

Uso educacional/pessoal. Modelos têm suas próprias licenças (respeite os termos dos criadores no Civitai/HF).
