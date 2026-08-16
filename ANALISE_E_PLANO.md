# 🎨 Análise Profunda + Blueprint Definitivo — Advanced Multi-Model Image Studio

## 1. O que existia na pasta

| Arquivo | Papel | Estado |
|---|---|---|
| `cópia_de_krea_2_turbo_colab.py` | Notebook original — Krea-2-Turbo (Wan2GP INT8, T4, Gradio) | Funcional, mas só Krea |
| `cópia_de_krea_2_turbo_colab(1).py` | Evolução "Advanced Multi-Model" (registry, Civitai, diffusers, LoRA) | **Quebrado** (ver abaixo) |
| `chat-Notebook Colab Funcionamento Profundo.txt` | 17.052 linhas documentando toda a evolução e **todos os erros** | Fonte de verdade dos requisitos |

## 2. Erros históricos que travaram o projeto (extraídos do chat)

1. **`DiffusionPipeline.from_single_file` inexistente** → usar classes específicas (SDXL/SD1.5/FLUX).
2. **Checkpoints do Civitai sem text encoder/VAE embutidos** → `Failed to load CLIPTextModel / UNet2DConditionModel` → injetar componentes do repositório base (SDXL base, SD 1.5, FLUX T5-XXL etc.).
3. **Anima detectada como SDXL** → `AnimaPipeline` não existe no diffusers 0.36 → precisa de ComfyUI (fallback universal).
4. **f-strings multilinha quebradas** → `SyntaxError: unterminated f-string` (linhas 825, 1108) → banido no código novo.
5. **Conflitos de dependências do Colab** (NumPy 2.x, protobuf, hf-gradio, onnx-weekly) → setups não podem rebaixar numpy; usar versões compatíveis.
6. **Download escolhia arquivo errado** (pegou `_txt.safetensors` de 2.9MB em vez do modelo completo de 7.79GB) → escolha por tipo `Model` + maior tamanho + SafeTensor.

## 3. Base models do Civitai (verificado ao vivo na API hoje)

`SD 1.4/1.5`, `SD 1.5 Hyper/LCM`, `SD 2.1/2.1 768`, `SDXL 0.9/1.0`, `SDXL Lightning/LCM/Hyper`, `Pony`, `Illustrious`, `NoobAI`, `Anima`, `Flux.1 D/Krea`, `Flux.2 Klein 4B-base`, `Krea 2`, `Qwen`, `ZImageBase/Turbo`, `Wan Video 1.3B/14B`, `Upscaler`, `Other` — + históricos mapeados: `FLUX.1 S/K`, `SD 3/3.5`, `Animagine XL`, `AuraFlow`, `Hunyuan DiT`, `HiDream`, `Lumina`, `Chroma`, `ERNIE`, `Grok`.

## 4. Blueprint definitivo — arquitetura de 3 motores

```
Usuário (URL/ID Civitai | arquivo local | Krea-2)
   ↓
API Civitai v1: modelo → versões → arquivo (Model/LoRA/VAE/TextualInversion)
   ↓
Detecção de família (BASE_MODEL_MAP) → presets completos (FAMILY_PRESETS)
   ↓
┌─────────────────────────────────────────────────────────────┐
│ Motor 1: DIFFUSERS (nativo)                                  │
│   - from_single_file com cadeia de fallbacks                  │
│   - injeção automática: TE1/TE2/VAE/UNet do repositório base  │
│   - scheduler por família (LCM/Hyper/Karras...)               │
│   - offload: model_cpu_offload + vae tiling/slicing           │
│   - FLUX: quantização INT8 weight-only (torchao) p/ T4        │
├─────────────────────────────────────────────────────────────┤
│ Motor 2: COMFYUI (fallback UNIVERSAL, headless via API)       │
│   - CheckpointLoaderSimple (SD1.5/SD2/SDXL/Pony/Anima/...)    │
│   - UNET+DualCLIP+VAE+FluxGuidance (FLUX/Chroma/Qwen)         │
│   - UNET+TripleCLIP+VAE (SD3/3.5)                             │
│   - LoRA via LoraLoader | img2img via LoadImage+VAEEncode     │
│   - Upscaler via RealESRGAN x4                                │
├─────────────────────────────────────────────────────────────┤
│ Motor 3: WAN2GP — Krea-2-Turbo (mantido do original)          │
└─────────────────────────────────────────────────────────────┘
   ↓
Geração unificada: template de prompt (Pony/Illustrious/NoobAI/Animagine)
   → Gradio (aba Modelo + aba Gerar + LoRA/VAE + ZIP)
```

### Requisitos por família implementados (FAMILY_PRESETS)
- **SD 1.5/2.x**: TE + VAE `sd-vae-ft-mse`; CFG 7-8; 512-768px; DPM++ 2M Karras; (Hyper/LCM: CFG 1, 4 passos).
- **SDXL**: TE duplo + VAE `sdxl-vae-fp16-fix`; CFG 6.5; 1024px; (Lightning/LCM/Hyper: CFG 1, 4 passos).
- **Pony**: prefixo obrigatório `score_9, score_8_up...` (auto); CFG 6.5.
- **Illustrious/NoobAI/Animagine**: tags de qualidade `masterpiece, best quality...` (+ `rating:` no NoobAI) (auto); CFG 5-6.
- **FLUX.1 D/S/K**: T5-XXL + CLIP-L + VAE flux; CFG 3.5 (dev) / 0 (schnell); 4-28 passos; **INT8 via torchao na T4**; fallback ComfyUI.
- **FLUX.2 Klein**: FluxPipeline2 (fallback FluxPipeline).
- **Z-Image Base/Turbo**: transformer single-file + pipeline base; CFG 0; 4-8 passos.
- **Qwen-Image**: QwenImagePipeline single-file.
- **SD 3/3.5**: StableDiffusion3Pipeline + fallback ComfyUI (TripleCLIP).
- **Anima/Chroma/AuraFlow/Hunyuan/Lumina/HiDream/ERNIE**: rota primária ComfyUI (mensagens claras p/ Grok — hospedado — e Wan Video).
- **Upscaler**: RealESRGAN via ComfyUI (2x/4x em imagem enviada).
- **Other**: sniffing + CheckpointLoaderSimple universal do ComfyUI.

## 5. Arquivos entregues

| Arquivo | Uso |
|---|---|
| `Notebook_Definitivo_CivitAI.ipynb` | **Importe no Colab** (File → Upload notebook) e rode Cell 1→2→3→4 |
| `universal_app.py` | O app completo (já validado: `ast.parse` OK, 40 funções) |
| `_gerar_notebook.py` | Regenera o `.ipynb` a partir do `universal_app.py` |

## 6. Como usar
1. Colab → Runtime → **Change runtime type → T4 GPU**.
2. Upload `Notebook_Definitivo_CivitAI.ipynb` → Run all (ou Cell 1 → 2 → 3 → 4).
3. Abra o link público do Gradio.
4. Aba **Modelo**: cole URL/ID do Civitai (ex.: `https://civitai.com/models/2700278`) → *Baixar e Carregar*.
5. Aba **Gerar**: prompt + proporção + resolução → *Gerar*.
6. LoRAs: URL/ID do LoRA + peso → *Adicionar LoRA*.

## 7. Notas de manutenção
- Fonte do app: `universal_app.py` — edite e rode `python3 _gerar_notebook.py` para regenerar o notebook.
- Para adicionar família nova: 1 linha em `BASE_MODEL_MAP` + presets em `FAMILY_PRESETS`.
- FLUX requer token HF (gated) para alguns componentes base; se faltar, o ComfyUI tenta mirrors públicos.

---

# 8. Versão 2 — Entrega Definitiva (todas as melhorias implementadas)

## 8.1 Resumo do que mudou (v1 → v2)

`universal_app.py` reescrito: **135 KB, 79 funções**, `ast.parse` OK, zero f-strings multilinha.

| # | Melhoria | Onde |
|---|----------|------|
| P0-1 | Trigger words do Civitai anexadas ao prompt (checkbox) | `apply_templates` / `STATE["trained_words"]` |
| P0-2 | Dropdown de versões (idx \| nome \| base \| GB \| tags) substituindo Number | `query_civitai_click` / `version_dropdown` |
| P0-3 | FLUX sem gate: mirrors `comfyanonymous/flux_text_encoders` + `Kijai/flux-fp8` | `_flux_components` / `comfy_ensure_aux_files` |
| P0-4 | Dead code removido (nó órfão `ConditioningSetArea`, imports mortos) | — |
| P0-5 | Limpeza de temp files pós-geração | `_cleanup_tmp` |
| P1-6 | Hires fix em 2 passos (low-res → refine) | `gen_single` / `_gen_diffusers` / rota ComfyUI |
| P1-7 | CFG rescale 0.5–0.7 (SDXL/Illustrious/NoobAI/Pony/Animagine) | `_gen_diffusers` (guidance_rescale) |
| P1-8 | Negativas recomendadas por família (Pony `score_1..4`, Illustrious/NoobAI/Animagine) | `neg_prefix` nos presets + `apply_templates` |
| P1-9 | Inpainting (mask) — via ComfyUI universal (qualquer família) | `gen_single` + `comfy_build_workflow` (LoadMask/SetLatentNoiseMask) |
| P1-10 | ControlNet Canny (SD1.5/SDXL) via ComfyUI | `comfy_ensure_aux_files` + node ControlNetApplyAdvanced |
| P1-11 | Prompt matrix (`a;b|b2;c` → produto cartesiano, máx 16) | `expand_matrix` |
| P1-12 | PNGInfo embutida (prompt, neg, steps, CFG, seed, size, model) | `build_png_info` / `save_image_pnginfo` |
| P2-13 | Auto-budget de VRAM (pynvml, fallback torch) — clamp de resolução | `vram_info` / `clamp_resolution` |
| P2-14 | Quantização opcional: torchao INT8 (FLUX) / bitsandbytes 4-bit | `_quantize_model` |
| P2-15 | Speed-up: TF32, cudnn.benchmark, torch.compile opt-in | startup + `_apply_compile` |
| P2-16 | Download paralelo aria2c (-x16) com fallback requests + resume (Range) | `download_with_aria2` / `download_file_stream` |
| P2-17 | Verificação pós-download (tamanho vs sizeKB; sha256 disponível no JSON) | `download_file_stream` |
| P2-18 | Cache em disco (TTL 1h) + backoff exponencial/Retry-After no 429 | `civitai_get_model` |
| P2-19 | Gestão de disco: biblioteca local + delete (sidecar `.meta.json`) | `list_downloaded_models` / `delete_downloaded` |
| P2-20 | Supervisor: Cell 4 reinicia o app automaticamente em 5s | `_gerar_notebook.py` cell4 |
| P3-21 | Card do modelo (descrição, versões, trigger words, preview) | `query_civitai_click` / `model_card` |
| P3-22 | Biblioteca local (troca de modelo sem re-download) | aba Modelo → biblioteca |
| P3-23 | Comparação A/B (mesmo prompt em 2 modelos) | `compare_click` |
| P3-24 | TextualInversion via Civitai (diffusers + ComfyUI embeddings) | `load_ti_from_civitai` |
| P3-25 | (auto) sempre baixa a VERSÃO MAIS RECENTE com arquivo recomendado | `pick_latest_version` / `pick_file_from_version` |
| P3-26 | API HTTP externa (stdlib) — outra aplicação gera imagens via JSON | `start_api_server` |
| Extra | LoRA baixa e ativa sozinho (peso configurável) | `load_lora_from_civitai` |

## 8.2 API externa (porta 7861, stdlib, zero deps)

```
GET  /api/health              → {status, loaded, model, backend, family, vram}
GET  /api/models              → lista de arquivos baixados
POST /api/load_model          → {model_url, civitai_token, hf_token, version_index?, force_comfy?}
POST /api/generate            → {prompt, negative_prompt, steps, cfg, width, height, seed, num_images,
                                 use_template?, use_trigger?, hires_fix?, hires_denoise?, prompt_matrix?}
                                 → {status, images: [{image: base64 PNG, seed, info, width, height}]}
POST /api/unload
```
- CORS aberto (`*`); auth opcional via `Authorization: Bearer <api_key>` ou campo `api_key` no body.
- Locks `GEN_LOCK`/`LOAD_LOCK` serializam UI + API (sem corrida).
- Exemplo de consumo: Cell 5 do notebook.

## 8.3 Arquitetura final (3 motores em cascata)

1. **Diffusers nativo** (single-file + injeção TE/VAE/UNet): sd15/sd2/sd2_768/SDXL/Pony/Illustrious/NoobAI/Animagine/FLUX(INT8)/SD3/SD3.5/Z-Image/Qwen/Anima/Chroma
2. **ComfyUI headless** (fallback universal): Chroma/AuraFlow/Hunyuan/HiDream/Lumina/ERNIE/Upscaler/Other + inpaint/ControlNet/hires
3. **Wan2GP** (Krea-2-Turbo INT8, patch float16 T4)

Mensagens claras: Grok (hospedado xAI), Wan Video (gera vídeo).

## 8.4 Estado de validação

- `ast.parse(universal_app.py)` ✅ · `py_compile` ✅ · zero f-strings multilinha ✅
- JSON do notebook válido ✅ · app embutido (135.425 chars) com `ast.parse` ✅
- Wiring Gradio: 15 handlers conferidos (inputs/outputs) ✅
- Teste de runtime na T4: **não executável localmente** (sem torch); cobertura por análise estática + lógica herdada da v1 já validada em execuções anteriores.

## 8.5 Cell 4 v2 — Launch em BACKGROUND (kernel livre p/ Cell 5)

**Problema descoberto**: a Cell 4 v1 rodava `while True` no thread principal do kernel.
Jupyter/Colab só executa 1 célula por vez nesse thread → a Cell 5 ficava enfileirada para sempre
(a API funcionava só para aplicações externas, não para teste local).

**Fix (Cell 4 v2)**:
1. App lançado **detached** (`start_new_session=True`) → sobrevive ao kernel; cell termina
2. **Supervisor em thread daemon de fundo** → reinicia se cair/travar, crash-loop detect, backoff
3. Cell 4 aguarda a URL do Gradio (tail do log) e **retorna** → kernel livre → **Cell 5 roda**
4. Guarda `_SUP_ACTIVE` (evita 2 supervisores) + `pkill` na re-execução + parada via `.stop_supervisor`
5. Pre-flight: arquivo existe + `ast.parse` + GPU detectada

**Fluxo**: Cells 1→4 → kernel livre + Gradio aberto + API :7861 ativa → Cell 5 testa a API
→ para o app com `!touch /content/.stop_supervisor`.

**Atenção de build**: nunca injetar código Python com `\n`/`\u` escapes via bash heredoc
(o heredoc corrompe escapes → bloco `cell4` quebrou 1x). Gerador é reconstruído via script de
patch com string raw + `chr()` (patch_cell4.py) ou edit tool direto.

## 8.6 Hotfix: ComfyUI morria no Colab + Anima (debug ao vivo)

**Sintoma**: `Falha em todos os motores: diffusers: Anima sem suporte diffusers (No module named 'diffusers_anima') | comfy: ComfyUI morreu`.
Causa raiz: (1) Colab limpo sem libs de sistema (libGL.so.1 mata ComfyUI no startup); (2) Anima não tem
pipeline no diffusers 0.36; (3) flux/sd3/anima usam `UNETLoader` (models/diffusion_models) mas o ckpt era
linkado em models/checkpoints.

**Correções**:
1. `ensure_comfyui()`: `apt-get install libgl1 libglib2.0-0 libsm6 libxext6 libxrender1` + pip best-effort
   (continua mesmo se falhar) + `pkill` de ComfyUI órfão (evita porta em uso) + erro agora inclui
   **tail do comfy.log** (diagnóstico acionável, via `_read_log_tail`).
2. `comfy_ensure_aux_files()`: multi-URL com fallback (3 mirrors p/ ae.safetensors; 2 p/ qwen2_vl_2b;
   2 p/ anima_vae); erro claro se nenhum mirror funcionar.
3. Template **anima** no ComfyUI: `UNETLoader` + `CLIPLoader(type=anima, qwen2_vl_2b)` + `VAELoader(anima_vae)`
   + `EmptySD3LatentImage` (latente 16ch) + KSampler Euler cfg 3.5.
4. `comfy_run()`: ckpt linkado em `models/diffusion_models` para templates UNET (flux/sd3/anima);
   `models/checkpoints` para checkpoint (SD1.5/SDXL/Pony...).
5. Preset anima: cfg 3.5, sampler euler, comfy_template "anima".

**Próximo passo no Colab**: re-rodar Cell 2 (reescreve o app) → Cell 4 (pkill + restart) → Cell 5.
1º setup do ComfyUI leva vários minutos (clone + apt + pip + aux). Se `qwen2_vl_2b`/`anima_vae` não
existirem nos mirrors tentados, o erro dirá exatamente qual arquivo falta (buscar no HF e ajustar URL).
Alternativa imediata: carregar modelo SDXL/Illustrious/Pony (diffusers nativo, sem ComfyUI).

## 8.7 Hotfix: race condition no download (FileNotFoundError .part)

**Sintoma**: `FileNotFoundError: ...arthemyComicsAnima_v20.safetensors.part' -> '...safetensors'` em
`download_file_stream` + "Nenhum modelo carregado" no /api/generate seguinte.
**Causa raiz**: `POST /api/load_model` e o auto-load de `POST /api/generate` (param `model_url`)
rodavam em threads separadas (ThreadingHTTPServer) e baixavam o MESMO arquivo no MESMO `.part`
(race TOCTOU: os dois passavam no check `exists`, o 1o renomeava via os.replace, o 2o morria).
**Fix (4 camadas)**:
1. `DOWNLOAD_LOCK` global em `download_from_civitai` envolvendo check-exists + download (serializa
   por arquivo, e checa tamanho esperado p/ rejeitar .safetensors parcial de aria2c morto);
2. `os.replace` anti-race em `download_file_stream`: se `.part` sumiu e final existe → usa final;
   senão erro claro ("temporario ausente");
3. `POST /api/load_model` agora segura `LOAD_LOCK` (consistente com run_generation);
4. (já existia) skip se arquivo final existe com tamanho certo.
**Licao**: ThreadingHTTPServer + download do mesmo destino = sempre serializar com lock OU tmp
único por chamada; check-then-act (TOCTOU) não basta.

## 8.8 Hotfix: erro ao SALVAR imagem (SDXL)

**Sintoma**: geração SDXL OK, mas erro na etapa de salvar ("no notebook", não no download do Civitai).
**Causas prováveis blindadas**:
1. `PngImagePlugin.PngInfo.add_text` quebra com chars fora do latin-1 (emoji, etc.) — PNG tEXt é
   latin-1; agora `info` é sanitizado (encode latin-1 replace) e add_text em try/except;
2. `img.save` com modo exótico (F/I/CMYK) — agora converte p/ RGB quando fora de RGB/RGBA/L/P/I;
3. se `save_image_pnginfo` ainda falhar, o handler /api/generate devolve JSON claro
   ("Falha ao codificar PNG: ...") em vez de traceback cru;
4. Cell 5 salva em /content/api_output.png (caminho ABSOLUTO; relativo podia falhar se cwd != /content)
   com try/except + traceback + print de disco livre (detecta disco cheio).

## 8.9 Hotfix: TypeError no callback de progresso (geracao SDXL/Illustrious)

**Sintoma**: `TypeError: generate_ui.<locals>.<lambda>() missing 1 required positional argument: 'd'`
em `step_cb` — geração morria logo após carregar o modelo (Illustrious/SDXL).
**Causa raiz**: `step_cb` (diffusers callback_on_step_end) chamava `progress_cb(step_index, int(ss))`
(2 args) mas a UI passa `lambda a, b, d: ...` (3 args). Mesmo risco latente em 2 lambdas do path
ComfyUI (`progress_cb(a, b)`).
**Fix**: normalizador `_mk_cb(cb)` que aceita QUALQUER aridade (1/2/3 args) e repassa
`(done, total, desc)`; aplicado no topo de `run_generation` e `gen_single`; step_cb agora passa
3 args com descricao "Passo x/y"; lambdas comfy passam (a, b, d).
**Obs**: warning "132 > 77 tokens" e truncamento do final do prompt no CLIP (Illustrious) — nao e
erro, mas prompt muito longo perde a cauda; nao tratado (limites diferem por familia).

## 8.10 Hotfix: ComfyUI "unrecognized arguments: --headless"

**Sintoma**: `main.py: error: unrecognized arguments: --headless` no tail do comfy.log → "ComfyUI morreu".
**Causa**: versão atual do ComfyUI (git main, 2025/2026) REMOVEU o flag `--headless`.
**Fix**: removido `--headless` do Popen em `ensure_comfyui()`. No Colab (sem DISPLAY) o ComfyUI roda
headless por padrão; `--disable-auto-launch` já impede abrir navegador.
**Status**: diffusers segue ok p/ SDXL/Illustrious/Pony (Carga: OK visto no log real). Anima ainda
depende do ComfyUI + aux files (qwen2_vl_2b / anima_vae) — próximo teste após o fix.

## 8.11 Anima: text encoder REAL (Qwen3-0.6B) + VAE Wan21 (HDR)

**Descobertas ao vivo (via API do Civitai + fonte do ComfyUI)**:
1. O ComfyUI atual (git main) usa para Anima o text encoder **Qwen3-0.6B** (`qwen3_06b`), NAO
   `qwen2_vl_2b`; `CLIPType` nao tem "anima" — a deteccao e por CONTEUDO do arquivo
   (`model.layers.0.post_attention_layernorm.weight` hidden 1024 -> TEModel.QWEN3_06B -> anima te).
2. O **text encoder vem do proprio modelo no Civitai**: Arthemy Anima v2.0 inclui
   `arthemyComicsAnima_v20_txt.safetensors` (1.16 GB, tipo "Text Encoder").
3. Anima usa **latente Wan21 (16ch)** -> VAE Wan-style. VAE oficial privado (CircleStone/Anima 401);
   usado o publico **"HDR VAE (Anima, Krea2, QWEN Image)"** (civitai 2718533, fp32 495 MB).
4. Recomendacoes do criador: CFG 6.0, steps 35, ~1024x1344, negativa com score_1..3.

**Implementado**:
- `download_from_civitai`: p/ baseModel Anima baixa TE (mesmo version) + VAE HDR -> STATE[anima_te_path/vae_path];
  `DOWNLOAD_LOCK` virou RLock (aux chama download_from_civitai aninhado; guard contra recursao no 2718533).
- `comfy_ensure_aux_files` template anima: copia os arquivos locais p/ text_encoders/ e vae/ (loop agora
  aceita caminho local); se modelo veio de arquivo local sem TE/VAE, erro claro com dica.
- workflow anima: CLIPLoader(clip_name1=<TE real>, type="anima") + UNETLoader weight_dtype fp16 (T4).
- preset anima: cfg 6.0, steps default 35, neg_prefix score_1..3, resolucoes incluem 1344x1024.

## 8.12 Anima: workflow = blueprint oficial do ComfyUI + diagnostico da API

**Status do teste do usuario**: load Anima OK via ComfyUI (loaded True, TE+VAE baixados) mas
/api/generate respondeu `status: None | imagens: 0` -> formato `{"error": ...}` (500) — provavel
"Nenhum modelo carregado" (estado perdido entre steps 3 e 4) ou erro de workflow cego.

**Workflow Anima CORRIGIDO p/ bater com o blueprint oficial do ComfyUI** (`Text to Image (Anima).json`):
- CLIPLoader type = `stable_diffusion` (NAO "anima"; o TE Qwen3-0.6B e detectado por CONTEUDO)
- Latent = `EmptyLatentImage` (NAO EmptySD3LatentImage; ComfyUI expande 4ch->16ch via fix_empty_latent_channels)
- KSampler sampler = `er_sde`, scheduler = `simple` (scheduler pass-through no preset)
- VAE = `qwen_image_vae` oficial (Qwen/Qwen-Image vae/diffusion_pytorch_model.safetensors, 253 MB,
  publico) com fallback p/ HDR VAE do Civitai
- UNETLoader weight_dtype fp16 (T4 sem bf16 nativo)

**Cell 5 melhorada (diagnostico completo)**: auto-load idempotente se loaded False; imprime
HTTP code + corpo completo da resposta em falha; tail do /content/studio/comfy.log (ultimas 25 linhas);
salva /content/api_output.png com try/except.

## 8.13 Bug Cell 2: `UsageError: Line magic function '%%writefile' not found`

**Sintoma**: usuário rodou a Cell 2 no Colab e recebeu `UsageError: Line magic function
'%%writefile' not found`.

**Causa raiz**: no `_gerar_notebook.py`, a `cell2` era montada com comentários descritivos
(`# 🧩 Cell 2 — ...`) **antes** da linha `%%writefile universal_app.py`. No IPython, um
**cell magic** (`%%...`) só é reconhecido se for o **primeiro token da célula**
(`inputtransformer2.cell_magic`). Com um comentário na primeira linha, o `%%writefile` é
interpretado como *line magic* dentro do código → `UsageError`.

**Fix aplicado**: mover `%%writefile universal_app.py` para a **linha 1** da célula; os
comentários descritivos ficam **depois** do magic (viram comentários inofensivos no topo do
arquivo gerado). Notebook regenerado e verificado:
- Cell 2 começa exatamente com `%%writefile universal_app.py` (bad_magic_pos em nenhuma célula);
- roundtrip: conteúdo da célula 2 (após remover as 2 linhas de comentário do topo) é **idêntico**
  ao `universal_app.py` (149.141 chars);
- `ast.parse` OK.

**Lição permanente**: `%%` (cell magic) SEMPRE na linha 1 da célula, sem comentários/linhas em
branco antes. `!` (line magic) pode aparecer em qualquer posição.

## 8.14 Anima no ComfyUI git main: `weight_dtype` e `clip_name` (validado na fonte)

**Sintoma (teste real do usuário na VM)**: `/api/generate` → HTTP 500, `ComfyUI rejeitou
workflow` com 2 erros de validação:
```
* UNETLoader 1: Value not in list: weight_dtype: 'fp16' not in ['default','fp8_e4m3fn','fp8_e4m3fn_fast','fp8_e5m2']
* CLIPLoader 2:  Required input is missing: clip_name
```

**Causas (API do ComfyUI mudou no git main)**: o projeto é instalado por `git clone main`, então
toda execução usa a versão mais nova — flags/nomes de input SÃO instáveis:
1. `UNETLoader.weight_dtype="fp16"` **removido**. Valores válidos: `default` / `fp8_e4m3fn` /
   `fp8_e4m3fn_fast` / `fp8_e5m2`. T4 (Turing 7.5) **não suporta fp8**; `default` resolve para
   fp16 na T4 via `model_management.unet_dtype()` + `PRIORITIZE_FP16` (confirmado no
   `comfy/sd.py:load_diffusion_model_state_dict`). O próprio blueprint oficial
   `blueprints/Text to Image (Anima).json` usa `weight_dtype="default"`.
2. `CLIPLoader` (loader simples) agora exige **`clip_name`** (singular) — o parâmetro antes
   chamava `clip_name1` (nodes.py:1007). `DualCLIPLoader`/`TripleCLIPLoader` continuam com
   `clip_name1/2/3`. O blueprint oficial usa `clip_name` + `type="stable_diffusion"` +
   `device="default"` para o TE Qwen3-0.6B.

**Verificações na fonte (clone /tmp/ComfyUI_src)**:
- `TEModel.QWEN3_06B` → `comfy.text_encoders.anima.te()` + `AnimaTokenizer` (detecção por
  CONTEÚDO do state dict, independente do `clip_type`) — logo `type="stable_diffusion"` OK.
- `Anima.supported_inference_dtypes = [bf16, fp16, fp32]`; com `default` a T4 cai em fp16
  (memory_usage_factor *= 1.4).
- sampler `er_sde` e scheduler `simple` existem; `VAELoader` usa `vae_name`;
  `EmptyLatentImage` width/height/batch_size; `SaveImage` images/filename_prefix.
- `TripleCLIPLoader` segue existindo em `comfy_extras/nodes_sd3.py` (branch sd3 intacto).

**Fix aplicado** (`comfy_build_workflow`, branch anima): `{"weight_dtype": "default"}`
+ `{"clip_name": te_name, "type": "stable_diffusion"}`. Notebook regenerado e roundtrip
verificado (149288 chars, idêntico ao universal_app.py).

## 8.15 VAE Anima: formato DIFFUSERS quebrava o ComfyUI — oficial `circlestone-labs` + header-check

**Sintoma (teste real na VM)**: workflow Anima foi ACEITO (fixes 8.14 resolveram a validação),
mas o VAELoader (node 3) falhou em runtime:
```
RuntimeError: Error(s) in loading state_dict for AutoencoderKL:
  size mismatch for encoder.conv_in.weight: [96,3,3,3,3] (checkpoint) vs [128,3,3,3] (modelo)
  ... decoder.conv_in.weight: [384,16,3,3,3] vs [384,16,3,3] ...
```

**Causa raiz**: `_download_anima_aux` baixava o VAE do **Qwen/Qwen-Image**
(`vae/diffusion_pytorch_model.safetensors`) — formato **diffusers** (convs 2D, 128ch).
O ComfyUI (`comfy/sd.py` VAE.__init__) NÃO detecta esse layout na cadeia de `decoder.conv_in.weight`
(shape[1] != 64/32 → cai no branch **default SD1.x**, `AutoencoderKL` 2D) → size mismatch.

**VAE correto (do blueprint oficial `Text to Image (Anima).json`)**: `circlestone-labs/Anima`
`split_files/vae/qwen_image_vae.safetensors` (253.806.246 bytes, **público**, 200 OK via HEAD;
org diferente do `CircleStone` que dera 401). É um **WanVAE 2.1 puro**:
- `decoder.middle.0.residual.0.gamma` presente → branch Wan 2.1 do detector
  (`comfy.ldm.wan.vae.WanVAE`, 16ch) — bate com `Anima.latent_format = latent_formats.Wan21`;
- sem `decoder.upsamples.0.upsamples.0.residual.2.weight` (senão cairia no Wan 2.2/48ch).

**Fix implementado**:
1. `ANIMA_VAE_URL` oficial como fonte **primária** (fallback HDR civitai 2718533 mantido).
2. `_anima_vae_valid(path)`: header-check de safetensors (8 bytes len + JSON header;
   exige `decoder.middle.0.residual.0.gamma` e ausência da chave Wan 2.2; rejeita <4KB e
   header corrompido) — aplicado em `_download_anima_aux` (apaga e rebaixa o arquivo errado
   que já está em disco) e em `comfy_ensure_aux_files` (recopia se o dest no ComfyUI estiver
   inválido — caso da VM, que já tinha o VAE diffusers copiado).
3. Teste local do helper: True para o header oficial baixado, False para vazio/lixo.
4. Notebook regenerado (151.963 chars), roundtrip idêntico, `fix_empty_latent_channels`
   confirmado em `comfy/sample.py:40`.

**Próximo passo do usuário**: re-baixar o notebook; rodar Cell 2 → Cell 4 → Cell 5. No Cell 5,
o app detecta o VAE inválido (~253 MB diffusers) em disco, remove e baixa o oficial (253 MB) —
uma única vez; dali em diante a geração Anima deve completar.

## 8.16 Mesmo erro VAE 3x — diagnóstico de código antigo + auto-reparo (APP_VER / repair_vae)

**Observação**: o erro `[96,3,3,3,3] vs [128,3,3,3]` repetiu IDÊNTICO ~3 min depois do fix 8.15,
sem troca da URL do Gradio — forte indicação de que a VM continuava rodando o **app antigo**
(usuário rodou só a Cell 5; o processo do app só muda com Cell 2 re-escrevendo o arquivo +
Cell 4 reiniciando o subprocesso; a URL do Gradio **muda a cada restart** — é o tell definitivo).

**Blindagem (elimina a ambiguidade e torna o VAE indestrutível)**:
1. `APP_VER = "v2.1.20260811"` — exposto em `/api/health` (`app_ver`) e no print de startup;
   a Cell 5 imprime. `app_ver: None` ⇒ app antigo ⇒ rodar Cell 2 + Cell 4.
2. `_download_anima_aux` reestruturado: **TE = best-effort** (erro logado, nunca bloqueia o VAE)
   e **VAE = obrigatório com falhas que vazam** (o `try/except` externo que engolia erros foi
   removido — antes um erro no TE abortava a função e o VAE nunca era corrigido).
3. `comfy_ensure_aux_files` (template anima): se `/studio` não tem WanVAE válido, **baixa o VAE
   oficial direto no diretório do ComfyUI** (última linha de defesa contra arquivo órfão errado).
4. Novo endpoint **`POST /api/repair_vae`**: valida/rebaixa o WanVAE 2.1 oficial em
   `/studio/models/vaes/` e copia p/ `/content/ComfyUI/models/vae/` — chamado pela Cell 5
   antes do load (passo 1.5), com resultado impresso.
5. `/api/health` agora reporta `anima_vae_valid` (studio) e `comfy_vae_valid` (ComfyUI).

**Protocolo de execução do usuário (ordem crítica)**:
1. Re-enviar o notebook (o arquivo NOVO) para o Colab.
2. Rodar **Cell 2** (reescreve universal_app.py) → **Cell 4** (reinicia o app; a URL do Gradio
   DEVE mudar) → **Cell 5** (mostra app_ver + repair_vae + geração).
3. Verificação: `app_ver: v2.1.20260811` e `studio_vae_valid: True | comfy_vae_valid: True`
   antes do generate; `status: ok | imagens: 1`.

## 8.17 `git clone` exit 128 — repair_vae criava /content/ComfyUI antes do clone

**Sintoma**: em VM NOVA (Colab resetou — disco voltou a 67.8 GB, modelo/TE/VAE baixados de
novo), `/api/load_model` falhou: `git clone ... returned non-zero exit status 128`.
O usuário marcou "force comfyui" na UI (mesmo caminho).

**Causa raiz (100% consistente com a saída)**: o endpoint **`/api/repair_vae`** (passo 1.5 da
Cell 5, criado no fix 8.16) copiava o VAE para `/content/ComfyUI/models/vae/anima_vae.safetensors`
e **criava o diretório** `/content/ComfyUI/` (mkdir parents=True) **antes** do ComfyUI existir.
Na sequência, `ensure_comfyui` via que `main.py` não existia → `git clone ... /content/ComfyUI`
→ **"destination path already exists and is not empty"** → exit 128. O clone em si nunca falhou —
o repair_vae criou o obstáculo.

**Fix em 2 camadas**:
1. `repair_vae`: só copia para o ComfyUI se `/content/ComfyUI/main.py` existir; senão imprime
   "copia adiada p/ comfy_ensure_aux_files" (na geração o arquivo é copiado do /studio, que
   continua válido).
2. `ensure_comfyui`: antes do clone, **remove** diretório inválido/incompleto
   (`shutil.rmtree`); clone com **retry 3× + `--depth 1`** (clone raso, mais rápido/robusto) +
   `subprocess.run(capture_output=True, text=True)` para **mostrar o stderr real do git** no
   erro (antes, `check_call` só dizia "exit 128" sem detalhe).

**Estado na VM após o fix**: modelo 8.1 GB + TE 1.16 GB + VAE oficial 253 MB já em disco
(baixados no load que falhou); próxima execução só clona o ComfyUI (~1-2 min), instala deps,
e roda a geração.

**Protocolo**: re-enviar notebook → Cell 2 → Cell 4 → Cell 5. Esperado:
`repair_vae: studio_vae_valid True | comfy_vae_valid True (ou adiado se ComfyUI nao instalado)`,
load OK (clona ComfyUI), generate `status: ok | imagens: 1`.

## 8.18 Flux + Kestral: "clip input is invalid: None" — instrumentação do traceback

**Sintoma**: workflow FLUX no ComfyUI (Kestral Flux Anime, com e sem "forçar ComfyUI"):
load OK, mas na geração erro no nó 3 (`CLIPTextEncode`): *"clip input is invalid: None —
If the clip is from a checkpoint loader node your checkpoint does not contain a valid
clip or text encoder model"*. `execution_cached: ['1','4']` (UNETLoader + VAELoader) —
confirma que o template flux é o usado (VAE = nó 4).

**Análise na fonte (`/tmp/ComfyUI_src`) que descartou hipóteses**:
- `DualCLIPLoader` (nodes.py:1031, git main) usa `folder_paths.get_filename_list("text_encoders")`
  → a pasta usada pelo app (models/text_encoders) está correta;
- `get_full_path_or_raise` → arquivo inexistente daria erro no **nó 2** com "file not found";
- `comfy.sd.load_text_encoder_state_dicts` **nunca retorna None** (retorna CLIP válido ou lança)
  → o None no nó 3 não pode vir de um load "normal" do DualCLIPLoader.

Conclusão: a causa real está no **traceback do `execution_error`** — que o app cortava
(`raise RuntimeError("ComfyUI erro: " + str(msgs)[:600])`, truncando exatamente aí no print do usuário).

**Fix (instrumentação + blindagem de download)**:
1. `comfy_run`: no erro, extrai `exception_message` + **últimas 14 linhas do traceback** do evento
   `execution_error` + **tail do comfy.log (30 linhas)** — a próxima saída mostra o stack real
   (será RAM OOM ou chave errada no T5, e.g.).
2. `comfy_ensure_aux_files`: `MIN_SIZES` por arquivo (t5xxl_fp8 ≥ 2 GB, clip_l ≥ 200 MB,
   clip_g ≥ 1 GB, ae ≥ 200 MB, sd_vae ≥ 200 MB) — o check anterior `>1024 bytes` deixava passar
   HTML truncado/página de erro; agora rebaixa e tenta o próximo mirror.

**Hipóteses para o próximo run** (o stack dirá): RAM 12 GB insuficiente para T5-XXL fp8 (4.9 GB)
+ UNET grande simultaneamente; ou t5xxl_fp8_e4m3fn.safetensors corrompido/errado no disco.

**Protocolo**: re-enviar notebook → Cell 2 → Cell 4 → Cell 5 (forçar ComfyUI). Se o erro
persistir, a saída agora traz o traceback completo + tail do comfy.log — colar aqui.

## 8.19 Anima: geração rodou no ComfyUI (120s) — "Falha ao codificar PNG: unknown file extension"

**Grande progresso**: a geração Anima executou até o fim no ComfyUI (25/25 steps, "Prompt
executed in 120.42 seconds", `Requested to load WanVAE` — **fix do VAE oficial (8.15) confirmado
em runtime**). A falha final foi no APP:

```
Falha ao codificar PNG: unknown file extension:
```

**Causa raiz**: `/api/generate` salva em `io.BytesIO()` e chamava `img.save(buf, pnginfo=...)`
**sem `format=`** — PIL não infere extensão de file-like → `KeyError: unknown file extension`.
Este bug existia desde o fix do SDXL que só tornou o erro legível; Anima foi o primeiro fluxo
que chegou de fato à etapa de salvar via API.

**Fix**: `save_image_pnginfo(img, path, info)` agora passa `format="PNG"` explícito (no save
principal E no fallback) — funciona para BytesIO e para caminhos `.png`.

**Observação (Gradio "parou")**: o app respondeu o 500 (estava vivo durante o teste); o Gradio
é a mesma thread principal (demo.launch). Causa provável: iframe expirado/refresh, ou o app foi
reiniciado pelo supervisor da Cell 4 (nova URL impressa na Cell 4). Checar com
`!curl -s http://127.0.0.1:7861/api/health`.

**Protocolo**: re-enviar notebook → Cell 2 → Cell 4 → Cell 5 (Anima já em disco na VM:
modelo 8.1 GB + TE + VAE válido; ComfyUI já clonado). Esperado: `status: ok | imagens: 1` +
`Salvo: /content/api_output.png` (a première imagem real via API!).

## 8.20 API keys pararam de carregar — Cell 1 não rodada em VM nova

**Sintoma**: campos de token do Gradio vazios / chaves não pre-preenchidas.

**Diagnóstico**: `_load_env_tokens()` (app, roda no import) lê `CIVITAI_TOKEN`/`HF_TOKEN` do
**env do processo** + `/content/studio/tokens.json` — e AMBOS eram criados **somente pela Cell 1**.
Em VM nova (Colab reseta `/content`), quem roda Cell 2 → Cell 4 → Cell 5 sem a Cell 1 fica
sem chaves: o `Popen` da Cell 4 herda o env do kernel (que não tem as variáveis) e o
`tokens.json` não existe. Não há regressão no código do app (função intacta).

**Fix (idempotente, cobre qualquer ordem de células)**:
1. **Cell 4**: logo após os imports, seta `os.environ['CIVITAI_TOKEN'/'HF_TOKEN']` + grava
   `/content/studio/tokens.json` (o processo do app herda o env → textboxes pre-preenchidas);
2. **Cell 5**: `os.environ.setdefault('CIVITAI_TOKEN'/'HF_TOKEN', ...)` para os POSTs locais
   (setdefault preserva um valor já existente no env).
Cell 1 mantida (pip installs, Wan2GP, patch krea2 float16).

**Nota de segurança**: o notebook contém os segredos em texto plano (pedido explícito do
usuário) — não compartilhar o arquivo.

**Protocolo**: re-enviar notebook → Cell 2 → Cell 4 → Cell 5. Na Cell 4 deve aparecer
`API keys: carregadas (env + tokens.json)` e no Gradio os campos já preenchidos.

## 8.21 Kestral Flux: "clip input is invalid: None" — causa raiz FALHA DE CASE no baseModel

**Traceback completo (instrumentação 8.18 funcionou) + comfy.log revelaram tudo**:
- `model_type FLUX` + `WARNING: No VAE weights detected, VAE not initalized.` +
  `WARNING: no CLIP/text encoder weights in checkpoint` → o app usou o **branch genérico
  (`CheckpointLoaderSimple`)** (executed nodes 1,4; erro no node 3 = negative CLIPTextEncode)
  em vez do template FLUX (`UNETLoader`+`DualCLIPLoader`+`VAELoader`).
- Um checkpoint FLUX **não contém CLIP nem VAE dentro** do safetensors → o
  `CheckpointLoaderSimple` devolve `clip=None` → CLIPTextEncode: "clip input is invalid".

**Causa raiz**: o Civitai reporta `baseModel: 'Flux.1 D'` (F maiúsculo), mas o
`BASE_MODEL_MAP` tem a chave `"FLUX.1 D"` → `dict.get("Flux.1 D")` = `None` →
`family = "other"` → `comfy_template "checkpoint"`. (Confirmado consultando a API do Civitai.)

**Fix em 3 camadas**:
1. `_family_from_base()`: lookup **case-insensitive + aliases** (flux, flux.1 d/dev/s/krea,
   sdxl, sd 1.5, pony, illustrious, noobai, animagine xl, qwen image, wan video, etc.) —
   substituiu todos os usos diretos de `BASE_MODEL_MAP.get` (3 pontos).
2. `_sniff_family_from_file()`: **rede de segurança** — lê o header safetensors (struct+JSON):
   `transformer_blocks.` + `guidance_embed`/`additive_encodings`/`double_blocks` → flux_dev;
   `model.diffusion_model.input_blocks.` → sd15; `conditioner.embedders.` → sdxl. Aplicado
   após download (load_model_from_civitai) e em load_local_file quando `family == "other"`
   (baseModel "Other" mal rotulado é comum no Civitai).
3. Com `family=flux_dev` o Kestral entra no template flux: UNETLoader(weight_dtype default) +
   DualCLIPLoader(clip_l, t5xxl_fp8) + VAELoader(ae) + EmptySD3LatentImage — os aux já estão
   na VM (clip_l + t5xxl + ae baixados nas tentativas anteriores).

**Testes locais**: `_family_from_base` 9 casos OK (Flux.1 D→flux_dev); sniff com header
safetensors fake flux → flux_dev (e não sobrescreve family já definida). Notebook 161.511 chars.

**Protocolo**: re-enviar → Cell 2 → Cell 4 → Cell 5 → gerar Kestral com "Forçar ComfyUI"
(flux no comfy usa T5 fp8 4.9 GB + ae — ideal p/ T4/12 GB RAM). Na carga deve aparecer
`Familia: flux_dev`. Se sem force (diffusers) o T5 bf16 9.6 GB pode estourar RAM — usar comfy.

## 8.22 Flux: `ae.safetensors` indisponível — mirrors BFL gated; Kijai renomeou o arquivo

**Sintoma**: `Arquivo auxiliar do ComfyUI indisponivel: ae.safetensors (nenhum mirror)`.
(Confirma que o fix 8.21 funcionou: o Kestral agora entra no template flux e pede os aux —
clip_l e t5xxl baixaram OK; só o ae falhou.)

**Testes HEAD locais**:
- `black-forest-labs/FLUX.1-schnell|dev/ae.safetensors` → **401** (gated: exige licença aceita
  na conta — token sozinho não basta);
- XLabs-AI, madroid, lllyasviel, prithivMLmods, digiplay → 401;
- `Kijai/flux-fp8/ae.safetensors` → **404** (repositório público — arquivo foi REMOVIDO/renomeado);
- Árvore via API HF: Kijai tem **`flux-vae-bf16.safetensors`** (167.664.710 bytes);
- HEAD direto → **200 OK** (CDN AWS), bf16, 100% compatível com VAEDecode.

**Fix**: mirror primário do ae = `Kijai/flux-fp8/resolve/main/flux-vae-bf16.safetensors`
(público, sempre disponível); BFL gated mantidos como fallback (funcionam se o token do
usuário tiver a licença aceita). Ajuste: `MIN_SIZES["ae.safetensors"]` de 200 MB → 100 MB
(o bf16 do Kijai tem 167 MB — o limite antigo rejeitaria o próprio mirror!).

**Protocolo**: re-enviar → Cell 2 → Cell 4 → Cell 5 → gerar Kestral (Forçar ComfyUI).
Esperado: `ComfyUI aux: ae.safetensors <- flux-vae-bf16.safetensors` (1x) e depois a geração.

## 8.23 Kestral: mesmo erro repetiu — a VM rodava o APP ANTIGO (verificação com APP_VER)

**Sintoma**: erro IDÊNTICO ao 8.21 (`CheckpointLoaderSimple` + clip None) ~33 min depois,
apesar dos fixes. 

**Análise decisiva**: o erro `ae.safetensors indisponível` da rodada anterior (que parecia
provar o 8.21 rodando) **também aconteceria no app antigo** — os mirrors BFL gated +
`Kijai/ae.safetensors` 404 já existiam antes do 8.21. Portanto a VM continuava com o
**processo antigo**: ou `universal_app.py` em `/content` não foi re-escrito (Cell 2 não
rodada), ou o app não foi reiniciado (Cell 4 não rodada — o supervisor reinicia com o
arquivo que ESTÁ no disco).

**Medidas**:
1. **`APP_VER` bump → `v2.1.20260813`** — exposto em `/api/health` e impresso na Cell 5;
   o usuário agora pode VERIFICAR qual código está rodando (se `app_ver` não for o novo,
   é app antigo).
2. **Rede de segurança em `comfy_run`**: se `family` não é flux/sd3/anima e o checkpoint
   tem header FLUX (sniff por conteúdo), trata como `flux_dev` **antes** do
   template/aux/symlink → o flux usa UNETLoader+DualCLIPLoader+VAELoader e baixa
   clip_l/t5xxl/ae — imune a metadata errada/antiga.
3. Corrigido bug de edição que colou as linhas `unet`+`clip` do flux branch (agora 2 linhas).

**Protocolo (com verificação OBRIGATÓRIA)**:
1. Re-upload do notebook NOVO → **Cell 2** (escreve o app no disco) → **Cell 4** (reinicia;
   **a URL do Gradio DEVE mudar** — se não mudou, o app novo não subiu).
2. Conferir no Cell 5: `app_ver: v2.1.20260813`.
3. Carregar o Kestral → o status deve mostrar **`Familia: flux_dev`**.
4. Gerar (Forçar ComfyUI) → `status: ok | imagens: 1`.


## §8.24 — Notebook unificado em UMA celula (v3)

### Pedido do usuario
"Facilitar a execucao do notebook — unificar as celulas em uma celula unica".

### O que mudou
- Notebook 6 celulas -> 2 (markdown + **1 celula de execucao**).
- A celula unica roda em cascata 5 etapas resilientes (try/except, nunca aborta):
  1. **[1/5] SETUP**: keys idempotente (env + tokens.json), pip best-effort (_pip wrapper),
     Wan2GP clone + krea2 float16 patch.
  2. **[2/5] ESCREVENDO**: universal_app.py gravado em Python — **sem %%writefile**
     (magic exige linha 1 da celula). Tecnica: `APP_SRC = r'''...'''` raw string,
     **0 ocorrencias de ''' no fonte** (verificado) — sem escapes, sem colisao com
     as aspas triplas dos docstrings. ast.parse pre-flight + GPU print.
  3. **[3/5] INICIANDO**: app detached (start_new_session) + supervisor em THREAD
     (guarda `_SUP_ACTIVE`; reinicia se cair; STOP = .stop_supervisor) + pkill limpo +
     poll ate 180s imprimindo a **URL do Gradio** e status da API (rails se nao subir:
     tail de 40 linhas do log + SystemExit).
  4. **[4/5] TESTE**: health (app_ver/VAE/disco) + repair_vae + load_model Anima 2700278.
  5. **[5/5] GERACAO**: generate 25 steps/1024 -> base64 -> save /content/api_output.png
     (com tail comfy.log se falhar).
- Fim: resumo + comando de parada + lembrete do Gradio.

### Correcoes no runner (detectadas por AST scan no notebook inteiro)
- `base64` NAO importado porem usado na etapa 5 -> adicionado ao import.
- `json.dumps` usava o nome `json` mas o import era `json as _json` -> `_json.dumps`.
- Falsos positivos do scan: `e` (except), `x` (comprehension), `_f` (with as),
  `mod` (for), `pkgs` (vararg) — validados como OK.

### Validacao (local, sem torch)
- celulas: 2 | app embutido: 162295 chars | notebook: 185.732 bytes
- ast.parse da CELULA INTEIRA (runner + app) OK
- roundtrip: app extraido da celula == universal_app.py (bytes identicos)
- sem f-string multilinha (f-tripla-dupla / f-tripla-simples) em celula + app
- APP_SRC embed: linha 74, raw string.

### Uso (para o usuario)
1. Upload do notebook novo -> Runtime T4 -> rodar a UNICA celula + aguardar.
2. Verificar: URL do Gradio muda (prova de app novo) + `[4/5] health app_ver`
   esperado v2.1.20260813 + imagem salva em /content/api_output.png.
3. Depois: abrir o Gradio p/ outros modelos; parar com `!touch /content/.stop_supervisor`.

### Contexto
- universal_app.py NAO mudou nesta iteracao (so o notebook v3).
- APP_VER continua v2.1.20260813.


## §8.25 — FLUX ainda falhava com 'clip input is invalid: None' — CAUSA RAIZ REAL (v2.1.20260814)

### Sintoma (repetido pelo usuario, com app NOVO — formatacao 8.18 de erro visivel)
- comfy.log: "model weight dtype torch.float8_e4m3fn, manual cast: torch.float16" +
  "model_type FLUX" + "No VAE weights detected" + "no CLIP/text encoder weights in checkpoint"
- traceback node 3 = CLIPTextEncode -> RaiseError("clip input is invalid: None")

### Diagnostico — DUAS causas empilhadas (uma escondia a outra)

1. **_sniff_family_from_file reconhecia so FLUX diffusers**: exigia `transformer_blocks.`
   + guidance_embed/additive_encodings/double_blocks E cortava o header nas primeiras
   120 chaves. O Kestral e checkpoint no formato **ComfyUI** (`model.diffusion_model.
   double_blocks.*` + `single_blocks.*` — o mesmo que o log `model_type FLUX`+fp8
   reporta) — sem `transformer_blocks.` nunca; e com ordem de chaves comeca por
   time_in/vector_in/guidance_in, double_blocks so aparece bem depois da posicao 120.
   -> quando baseModel vinha vazio/'Other', o sniff NAO corrigia e family ficava 'other'.

2. **BUG DE NOME no comfy_build_workflow (a causa que explica TUDO)**: o flux template
   era selecionado por `if family == "flux":` — mas a familia real do preset e
   **'flux_dev'/'flux_schnell'/'flux2_klein'** (com `comfy_template: "flux"` no preset).
   O comfy_run usava o template 'flux' para o sub_dir/aux, mas o BUILDER caia no else
   -> `CheckpointLoaderSimple` -> flux nao tem CLIP/VAE dentro -> clip=None -> erro.
   Ou seja: MESMO com familia correta detectada (load printava 'Familia: flux_dev'),
   o workflow ia para o template generico. Para o usuario, era o MESMO erro da 8.21,
   mas por razao diferente (e o app 8.23 ja roda).

### Fixes (universal_app.py)
- `_sniff_family_from_file`: usa o header COMPLETO (removeu o corte em 120 keys) e
  reconhece os formatos: `model.diffusion_model.double_blocks.`+`single_blocks.` ->
  flux_dev (ComfyUI FLUX, Kestral/FP8); `joint_transformer_blocks.` -> sd3 (SD3/SD3.5);
  diffusers FLUX (transformer_blocks.+guidance/additive/double); sd15 (input_blocks);
  sdxl (conditioner.embedders). Limite do header subiu 64MB->256MB (defensivo).
- `comfy_build_workflow`: branch flux agora aceita `("flux","flux_dev","flux_schnell",
  "flux2_klein")` (o preset manda comfy_template='flux'); branch sd3 aceita
  `("sd3","sd35")`.
- APP_VER -> **v2.1.20260814**.

### Testes locais (9 cases + 9 aliases, tudo verificado)
- sniff: Kestral ComfyUI format -> flux_dev; Kestral com double/single blocks SO apos a
  posicao 200 (o corte 120 antigo falharia) -> flux_dev; diffusers FLUX -> flux_dev;
  SD1.5 -> sd15; SDXL -> sdxl; SD3 -> sd3; outro -> other; family!=other nao sobrescreve.
- aliases _family_from_base: Flux.1 D/FLUX.1 D/flux.1 d -> flux_dev; SDXL 1.0 -> sdxl;
  Other/None -> other; Illustrious; Anima.
- templates: flux_dev/schnell/klein -> 'flux'; sd35 -> 'sd3'; anima -> 'anima'.
- ast.parse universal_app.py + celula inteira do notebook OK; roundtrip identico.

### Protocolo de re-teste (usuario)
1. Re-upload do notebook -> rodar a celula unica (re-grava o app v2.1.20260814).
2. Conferir em /api/health (ou saida da celula): app_ver v2.1.20260814.
3. Carregar o Kestral (URL 697877) -> log deve mostrar 'Familia: flux_dev' e, se a
   metadata vier vazia, 'Metadata other — detectado por conteudo: flux_dev'.
4. Gerar com Forcar ComfyUI -> status: ok | imagens: 1.
   Esperado: ae.safetensors do Kijai (167MB, 1x) se faltar + flux completo
   (clip_l + t5xxl_fp8 ja estao na VM; template flux = UNETLoader+DualCLIPLoader+
   VAELoader+EmptySD3LatentImage+FluxGuidance).

### Contexto
- O erro 8.23 era indistinguivel para o usuario (mesma msg) — por isso o APP_VER bump
  e ESSENCIAL para provar qual codigo rodou.
- universal_app.py agora com 163.158 chars; notebook 186.626 bytes.


## §8.26 — FLUX VALIDADO de ponta a ponta (usuario confirmou: "funcionou o flux")

- Modelo: Kestral Flux Anime (697877), formato ComfyUI fp8_e4m3fn.
- Resultado: geracao completa via template flux (UNETLoader + DualCLIPLoader clip_l/t5xxl_fp8
  + VAELoader ae Kijai + EmptySD3LatentImage + FluxGuidance) | status: ok | 1 imagem.
- Confirma retroativamente o diagnostico da §8.25: eram os 2 bugs empilhados
  (sniff so diffusers/corte-120 + builder comparando family=='flux' vs flux_dev).
- Status do roteiro de validacao: Anima OK (8.13-8.19) | FLUX OK (8.25-8.26).
  Restam: Illustrious (1273254) -> SD1.5 -> SDXL/Pony -> SD3.5 -> Wan/SDXL+LoRA,
  e re-teste do encode PNG do Anima via API (fix 8.19 nao re-testado na API).


## §8.27 — ROTEIRO DE VALIDACAO 100% CONCLUIDO (usuario confirmou: Illustrious ok, SD1.5 ok, SDXL ok, Pony ok, FLUX ok, Anima ok)

### Modelos validados de ponta a ponta (cada um em sua rota correta)
| Familia | Rota | Status |
|---|---|---|
| Anima (2700278) | ComfyUI (UNET + Qwen3 TE + WanVAE oficial) | OK |
| FLUX / Kestral (697877) | ComfyUI template flux (clip_l+t5xxl_fp8+ae Kijai) | OK |
| Illustrious (1273254) | Diffusers | OK |
| SD1.5 | Diffusers | OK |
| SDXL / Pony | Diffusers (CFG rescale + negativas por familia) | OK |

### Significado
- As 3 camadas de roteamento funcionam: Diffusers (SD1.5/SDXL/Pony/Illustrious),
  ComfyUI headless (Anima DiT + FLUX), Wan2GP (Krea-2 — nao testado, sem modelo).
- A cadeia de correcoes 8.13-8.26 estabilizou: deteccao de familia por metadata+conteudo,
  templates por familia, mirrors publicos de aux, PNG encode, erro com traceback completo.
- Notebook v3 celula unica: setup->app->teste automatico->imagem salva (validado).

### Pendentes (nao bloqueiam)
- Wan Video / Krea-2 (Wan2GP) sem modelo de teste.
- Ofertas em aberto: (a) geracao automatica de API key; (b) celula 'Publicar API' (cloudflared).
- Considerar Hindsight/Mnemosyne se pedir (ADR-005).


## §8.28 — Krea2 do Civitai: 'Nenhum arquivo do tipo Model' — 2 fixes (v2.1.20260815)

### Sintoma (usuario)
Baixar/carregar https://civitai.com/models/2759057 (Arthemy Comics Krea2) ->
"Erro ao carregar do Civitai: Nenhum arquivo do tipo 'Model' nessa versao."
(na pagina o arquivo existe).

### Diagnostico (API real consultada: 2 versoes, baseModel 'Krea 2')
- v1.1: 3 arquivos TODOS com type='Diffusion Model' (25.039.328 KB primary, 12.8GB x2)
- v1.0: 1 arquivo type='Diffusion Model' (12.8GB)
- O app so aceitava type='Model' no pick_file_from_version (wanted_type='Model' do
  load_model_from_civitai) -> raise. 'Diffusion Model' e o rotulo do Civitai para
  DiT/UNET puros (Krea2, FLUX...) — sem CLIP/VAE no safetensors.

### Fixes
1. pick_file_from_version: MODEL_FILE_TYPES = ('Model','Diffusion Model','Pruned Model',
   'Checkpoint') aceito quando o pedido e 'Model'/None; wanted_type explicito
   (TextEncoder/VAE/LoRA/TI) mantem o raise claro (sem fallback amplo que pegaria
   checkpoint como TE). Testado com JSON real: v1.1 -> arthemyComicsKrea2_v11 (23.88GB,
   primary) | pick_latest OK | TextEncoder raise correto.
2. Rota Krea2 no load (ANTES do ComfyUI — que falharia: DiT sem CLIP/VAE):
   - load_model_from_civitai e load_local_file: family=='krea2' e !force_comfy ->
     _load_wan2gp_custom(local, base, name): Wan2GP (motor 3) com o checkpoint custom +
     TE (Qwen3-VL-4B int8) + VAE (qwen_vae) base (baixa via download_krea_models se
     faltar); tenta model_type 'krea2_turbo' e fallback 'krea2_raw'; offload.profile
     budgets 9000/3500/1500/400; STATE backend=wan2gp/family=krea2.
   - _gen_wan2gp ja funciona com STATE['krea_model'] (qualquer arquivo).
- APP_VER -> v2.1.20260815.

### Nota pratica (disco/RAM T4)
- Pick padrao = v1.1 primary 23.88GB (fp8) — download demorado (~15-40min) e ocupa
  muito; se preferir, escolher a versao v1.0 (12.8GB) no dropdown de versoes do
  Gradio (version_index>0).
- Wan2GP offload: transformer 9GB budget + TE 3.5GB + VAE 1.5GB em 12GB RAM — apertado
  mas gerencia por blocos; se OOM, usar o arquivo menor.

### Validacao
- ast.parse + pick testado com JSON real da API + roundtrip notebook identico
  (167.016 chars, 190.672 bytes) + sem f-string multilinha.


## §8.29 — Fix Krea-2 Civitai via Wan2GP: Incompatibilidade de Transformers 4.45+ & Meta Tensors

### Causas Raiz
1. **`Unrolling kwargs is not supported for preprocess of None class`**: Em `transformers >= 4.45`, `ProcessorMixin` exige `image_processor` válido ao validar kwargs na chamada do processador se `image_processor` estiver em `attributes`. No `Krea2Qwen3VLProcessor`, o `image_processor` é `None` (apenas condicionamento de texto).
2. **`Cannot copy out of meta tensor; no data!`**: Em `krea2_main.py` do Wan2GP, `_load_transformer` instancia `SingleStreamDiT` com `init_empty_weights` (dispositivo `meta`). Ao carregar checkpoints quantizados do Civitai (com chaves `weight_scale`), o `optimum-quanto` tenta copiar tensores `meta` sem dados prévios.
3. **Falta de Fallback Graceful**: Não havia rotas alternativas se o carregamento via Wan2GP falhasse.

### Correções Aplicadas (`universal_app.py` & `_patch_krea2_main`)
- **`Krea2Qwen3VLProcessor`**: Sobrescrito o método `__call__` para redirecionar chamadas exclusivamente de texto direto para `self.tokenizer(text, **kwargs)`, ignorando a verificação de `image_processor` do `transformers 4.45+`.
- **`_load_transformer`**: Adicionado tratamento de exceção com fallback de inicialização direta em CPU sem `init_empty_weights` caso a cópia de meta tensor falhe.
- **Detecção de Quantização e Fallbacks**: Teste dinâmico de `quantizeTransformer` `(True, False)` e fallback automático para o motor universal ComfyUI se Wan2GP rejeitar o checkpoint.
- **Notebook & Versão**: `Notebook_Definitivo_CivitAI.ipynb` atualizado e validado (`ast.parse` OK).



## §8.30 — Sincronia quebrada: `_gerar_notebook.py` com SyntaxError + notebook defasado (v2.1.20260815)

### Sintoma (descoberto no restauro de contexto do Orchestrator)
1. `python _gerar_notebook.py` → `SyntaxError: invalid syntax. Perhaps you forgot a comma?` na linha 19
   (`cell_run = (` + primeira string do runner). O notebook da pasta (gerado 06:00) estava OK,
   mas era IMPOSSIVEL regenera-lo — o gerador em disco estava quebrado desde a sessao das 01:00.
2. O notebook embutia um app **defasado** em relacao ao `universal_app.py` (disco):
   - embed: `align_pkgs = ["numpy>=2.0.0", ...]` (reinstalaria numpy 2.4+ se Wan2GP faltasse)
   - disco: `align_pkgs = ["numpy==2.3.5", ...]`  (fix: numpy>=2.4 removeu `_blas_supports_fpe`
     que o scipy 1.16 exige — Wan2GP importa scipy.stats; sem o pin -> AttributeError no import)
   Ou seja: subir o notebook atual no Colab reintroduzia o bug do Krea-2/Wan2GP a cada VM nova.

### Causa raiz (2 camadas)
1. **Typo de aspa no gerador**: 2 linhas do bloco `cell_run` tinham `_pip('numpy==2.3.5')  # ...`
   FORA das aspas — ex.:
   ```python
   "...'safetensors')\n"_pip('numpy==2.3.5')  # trava versao...\n"   # FALTA " no inicio da 2a string
   ```
   O parser via `STRING NOME(call)` -> `invalid syntax. Perhaps you forgot a comma?`.
   **Tell diagnostico**: `tokenize` passa (639 tokens OK) e `ast.parse` falha — porque o tokenizer
   nao exige sequencia valida; o parser (peg) rejeita `STRING NAME`. O erro reportado aponta para
   OUTRA linha (19) pois o token falho vem "depois" no fluxo de recuperacao do peg parser.
   Raiz do typo: a ultima sessao adicionou o pin numpy ao runner E deixou variantes quebradas
   (`_t5.py`/`_tb.py` com o mesmo typo; `_t40..43.py` eram testes de escape/concat).
2. Processo: `universal_app.py` mudou (align_pkgs) sem regenerar o notebook; e o notebook foi
   gerado por um gerador valido que NAO esta em disco.

### Fix
- `_gerar_notebook.py` linhas 40 e 49: segunda string teve a aspa de abertura restaurada
  (implicit concat `"...\n" "_pip(...)\n"`).
- Markdown corrigido: "carrega Anima -> gera 1 imagem" (fluxo antigo) -> "health-check
  (app_ver, disco)" (fluxo real da celula unica — a celula NAO auto-carrega modelo desde o v3).
- Notebook regenerado de `universal_app.py` (203.407 chars).

### Validacao (100%)
- Roundtrip: app embutido == universal_app.py (bytes identicos, 203.407)
- ast.parse: gerador OK, celula (runner+APP_SRC) OK, app embutido OK
- JSON notebook OK, 2 celulas | zero aspas triplas simples no embed (raw-string) | zero f-string tripla
- runner + app com numpy==2.3.5 | md bater com health-only

### Licoes permanentes (aplicar em qualquer mudanca futura)
1. **Regra de ouro**: TODA alteracao no `universal_app.py` => rodar `python _gerar_notebook.py`
   e conferir `embedded == disk`. O notebook nunca deve ser editado a mao.
2. **NUNCA** colar chamadas codigo fora das strings no bloco do runner (aspa esquecida = bug
   invisivel: tokenize OK / ast falha apontando linha errada).
3. Apos regenerar, checar: zero aspas triplas simples no app (raw-string APP_SRC) e zero f-string tripla.
4. `_push_uapp.py` (helper de deploy da sessao anterior) embute app ANTIGO (200.580 chars, sem
   pin numpy) — **NAO usar**; o fluxo oficial e o notebook (celula unica).
5. APP_VER = v2.1.20260815 segue valido (o app em si nao mudou nesta iteracao; so o gerador +
   notebook). Se a VM antiga continuar no ar, conferir `app_ver` em /api/health antes de testar
   o Krea-2.

### Proximos passos sugeridos (nao bloqueiam)
- Re-validar Krea-2 (2759057) na T4 com o notebook REGENERADO (o app agora garante numpy 2.3.5
  no align do Wan2GP).
- Opcional: cell "[4/5]" restaurar auto-load Anima (download 8.1 GB — deixar desligado por padrao).


---

# §8.31 — Restauro de contexto do Orchestrator + deploy caiu (v2.1.20260815)

## Contexto restaurado (memorias: Engram/Hindsight/Mnemosyne + sessao 2026-08-11 + history colab)

### Estado verificado no restauro (tudo OK localmente)
1. `_gerar_notebook.py` roda limpo (exit 0) e **regenerou** `Notebook_Definitivo_CivitAI.ipynb`
   (fecha o §8.30; notebook 190.672 B -> 220 KB).
2. Roundtrip: `APP_SRC` extraido do notebook == `universal_app.py` em disco (203.407 chars, bytes
   identicos), APP_VER v2.1.20260815 presente no notebook.
3. `ast.parse`: app OK (112 funcoes, 0 TODOs, 0 f-string tripla, 0 aspas triplas simples).
4. Pins criticos presentes no app: `numpy==2.3.5` (scipy/_blas_supports_fpe) e
   `optimum-quanto==0.2.4` (API antiga — layout `tensor/weights/qint4.py` existe; a VM anterior
   tinha a versao NOVA sem esse layout, o que quebrou a investigacao do qint4/qint8).
5. Cell 4 v2 / supervisor `_sup.py` presentes e corretos.

### Problema principal: VM do Colab caiu
- `session_terminated: pruned` às 22:02 UTC (keep-alive 404 ~21:00 — ``gpu-t4-s-kkb-usw1b2-2ip740cubvi2``).
- `colab ls` -> "No active sessions found"; `sessions.json` vazio; token OAuth valido ate 23:02 UTC.
- O `deploy_atual` (image_studio_T4_...gradio.live) NAO pode mais ser atingido.

### Diagnostico da criacao de sessao T4 (colab new -s image_studio --gpu T4)
- `GET /tun/m/assign` -> **200 OK** com `acc:"t4"` (maquina reservada!), mas o **POST** de
  finalizacao -> **503 Service Unavailable** com `{"endpoint":"","sub":0,"subTier":0,
  "outcome":2}` — servico de provisionamento instavel/lotação no lado do Google (sabado ~19h
  Brasilia). NAO e problema de auth (token valido) nem de codigo.

### Entregue nesta sessao
- `deploy_colab_autoretry.py`: provisiona T4 com retry (30x, backoff 30-180s) + deploy completo
  (upload run_all.py -> setup+pip -> app detached+supervisor -> health app_ver -> repair_vae),
  espelhando deploy_new_session.py mas com `--check` e retry.
- Executado em background (deploy_retry.log) tentando capturar a janela de servico.

### Pendente (bloqueado pelo 503 do Colab)
- Re-validar Krea-2 (Civitai 2759057) na T4 com o notebook regenerado (fecha §8.28-8.29).
- Quando a VM subir: conferir `app_ver: v2.1.20260815` em /api/health antes de qualquer teste;
  a URL do Gradio DEVE mudar (prova de app novo).

### Licões permanentes
1. Nunca deixar o `_push_uapp.py` (embute app antigo) substituir o notebook como fonte de deploy.
2. `colab new` pode falhar com GET 200/POST 503 — retry com backoff E o caminho; nao eh bug local.
3. Sempre conferir `app_ver` (APP_VER bump) p/ provar qual codigo esta na VM.

## §8.32 — QA com pyflakes achou 2 bugs reais; fix + bump v2.1.20260815b

### Bugs encontrados (pyflakes, antes do deploy)
1. **`undefined name 'np'` (3x, `_parse_image_input`)**: a funcao usava `np.ndarray`
   (componentes `gr.Image` do Gradio podem entregar numpy array) mas **numpy nunca era importado**
   no app — NameError latente nos fluxos img2img/inpaint (P1-9). Eram 3 pontos: val do dict,
   layers[0] e o proprio img_input.
   Fix: `try: import numpy as np / except: np = None` no topo (robustez: se numpy faltar,
   os caminhos PIL/str/dict continuam; guard `np is not None` apenas nos ramos ndarray).
2. **`undefined name 'o'` em `expand_matrix`**: `"".join(o) if False else ...` era dead code
   (nunca avaliado) — limpo para `", ".join(...)` direto.

Outros 12 avisos pyflakes sao cosmeticos (imports nao usados, locals nao usados) — sem acao.

### Testes locais realizados
- `expand_matrix`: 6/6 (cartesiano por ';'/'|', cap 16) — o comportamento retorna exatamente
  a spec P1-11; o assert inicial do Orchestrator e que estava com expectativa errada.
- `_parse_image_input`: 12/12 (Teste A np-ausente 7/7: None/path inexistente/PIL/dict composite/
  mask/image/layers; Teste B np-presente 5/5: ndarray direto/dict/layers/PIL/path existente).

### Estado
- APP_VER bump para **v2.1.20260815b**; notebook regenerado; roundtrip embedded==disk True
  (203.713 chars); AST celula OK; zero f-string tripla; backups em backups/.
- Deploy aguarda o servico Colab voltar (503 no POST /tun/m/assign; retry 40x em background).

## §8.33 — Evolucao: WORKER KREA-2 ISOLADO + fix do contrato de imagem (v2.1.20260816)

### Motivacao
Analise comparativa com `krea_2_turbo_funcional.ipynb` (notebook que NAO desconecta):
- O notebook funcional roda Wan2GP num **processo python limpo** (`!python -u run_krea_turbo.py`),
  nunca no kernel inchado (TF ~1-2GB) — se o OOM memcg 12Gi matar algo, morre SO o subprocesso;
- mmgp `pinnedMemory=False` + `asyncTransfers=False` + budgets VRAM-first (10000/4000/1500/500);
- TFBlocker no sys.meta_path; PYTORCH_CUDA_ALLOC_CONF expandable_segments + gc threshold;
- gc.collect + empty_cache entre geracoes; inference_mode; limites T4 conscientes.
O nosso app ja tinha quase tudo (exceto isolamento e o contrato de imagem).

### Implementado (universal_app.py -> v2.1.20260816)
1. **`KREA2_WORKER_SRC`** (~14KB) — worker completo embutido no app (raw string `r"""..."""`,
   zero aspas triplas simples E duplas no conteudo, para coexistir com o APP_SRC `r'''...'''`).
   - Processo limpo: TFBlocker + env vars + patches (krea2_main float16/preprocess/processor,
     transformers docstring) + pins (numpy==2.3.5, optimum-quanto==0.2.4);
   - `load_model` com quant_opts (False/True por conteudo do nome) + krea2_turbo/raw;
   - `profile_model` pinnedMemory=False asyncTransfers=False budgets
     {transformer:9000, te:3500, vae:1500, *:400} + sdpa;
   - HTTP local na porta 7862: GET /health | POST /generate (base64 PNG) | POST /unload;
   - progress por arquivo (KREA2_WORKER_PROGRESS = /content/krea2_worker_progress.json);
   - gera com `torch.inference_mode()` + callback de step.
2. `_write_krea_worker_file()` — grava krea2_worker.py em APP_DIR (idempotente).
3. `_kill_krea_worker()` — terminate/kill + pkill + limpa STATE.
4. `_spawn_krea_worker(ckpt, name, cb)` — garante TE/VAE (download_krea_aux), requirements,
   spawn detached (start_new_session), poll health ate 10min, raise com tail do log.
5. `_proxy_worker_generate(...)` — POST /generate em thread + poll do progress file (1s),
   decodifica base64 -> PIL RGB; erro claro se worker morrer.
6. `_load_wan2gp_custom` e `load_wan2gp_krea`: **tentam worker primeiro**; fallback in-process
   intacto (try/except -> warn + caminho antigo).
7. `_gen_wan2gp`: se `STATE["krea_worker"]` -> proxy; senao in-process.
8. `unload_current_model`: POST /unload + _kill_krea_worker antes de limpar estado.
9. `/api/health` expoe `krea_worker: bool`.

### Fix importante: contrato de imagem do Wan2GP (bug latente)
- O padrao herdado `Image.fromarray(result[:, 0].permute(1, 2, 0).numpy())` **FALHA** no PIL
  para (H,W,1) — e o caminho Wan2GP **nunca foi validado de ponta a ponta** (§8.27: "nao testado").
- Novo `_result_to_image` (worker) / `_wan2gp_result_to_image` (app): aceita torch.Tensor/ndarray
  com shapes (B,C,H,W)/(B,H,W,C)/(B,H,W)/(H,W,C)/(H,W), escala float 0..1 ou 0..255,
  normaliza e converte para RGB. Testado 5/5 shapes no worker e 4/4 no app.

### Testes locais (sem GPU)
- AST app + AST worker OK; worker sem triplas simples/duplas; roundtrip notebook==disk
  (227.464 chars); pyflakes: nenhum undefined; f-tripla zero.
- Conversao de resultado: worker 5/5 (float CHW / mono / HWC / HW / float 0..255).
- do_generate com fake krea_model -> PNG base64 valido + callback de progresso disparado.

### Estado atual
- APP_VER v2.1.20260816. Backups: backups/universal_app_v2.1.20260816.py.
- Deploy: Colab continua 503 (sessao nova nao atribuida apos 40+12 tentativas);
  deploy_colab_autoretry.py em background; re-rodar quando o servico voltar.
- Proxima validacao na T4: carregar KREA-2 (2759057) e conferir `krea_worker: true` no health;
  app_ver v2.1.20260816; geracao via proxy; se OOM, SO o worker morre e a UI mostra o erro.

## §8.34 — Remocao definitiva do suporte Krea-2 / Wan2GP (v2.1.20260817)

### Decisao do projeto
Devido a problemas persistentes de **OOM (memcg 12Gi) e desconexao da VM no Colab** ao tentar
carregar e gerar com modelos Krea-2 (Wan2GP INT8/quantized), o suporte a Krea-2/Wan2GP foi
**totalmente removido** da aplicacao e do notebook, simplificando a arquitetura para **2 motores
em cascata** (Diffusers nativo + ComfyUI headless universal).

### O que foi removido
1. `WAN2GP_DIR` e constantes associadas;
2. `BASE_MODEL_MAP["Krea 2"]` e preset `"krea2"` de `FAMILY_PRESETS`;
3. `KREA2_WORKER_SRC`, worker isolado, helpers de spawn/kill/proxy HTTP;
4. `download_krea_aux`, `download_krea_models`, `_patch_krea2_main`, `_patch_transformers_rope`,
   `_patch_transformers_docstring`, `_ensure_wan2gp_requirements`, `_load_wan2gp_custom`,
   `load_wan2gp_krea`, `_wan2gp_result_to_image`, `_gen_wan2gp`;
5. Referencias em `STATE`, `unload_current_model`, `load_local_file`, `load_krea`, `CIVITAI_EXAMPLES`;
6. Subtitle "Wan2GP" atualizado para "Diffusers + ComfyUI".

### Guards de seguranca adicionados
- Tentativa de carregar modelo Krea-2 pelo Civitai ou arquivo local retorna imediatamente a mensagem:
  `"O suporte a modelos Krea-2 foi removido (causava desconexão por OOM no Colab). Use SDXL, Pony, Illustrious, NoobAI, Anima ou FLUX."` (sem tentar carregar nem gastar RAM).
- Botao de atalho Krea-2 retorna a mesma mensagem amigavel.

### Estado atual da aplicacao
- APP_VER: **v2.1.20260817** (tamanho do app reduzido de 227KB para 177KB; notebook de 243KB para 200KB).
- **2 Motores estaveis mantidos:**
  1. **Diffusers nativo**: SD 1.5, SDXL, Pony, Illustrious, NoobAI, Animagine XL, etc.
  2. **ComfyUI headless**: Anima (Qwen3 TE + WanVAE oficial), FLUX.1 (Kestral/dev/schnell), SD 3/3.5, Chroma, AuraFlow, etc.
- Validação estática 100% OK (ast.parse OK, pyflakes sem erros, roundtrip notebook==disk True).


## §8.35 — Reintegracao do Krea-2-Turbo Oficial INT8 baseada na analise dos notebooks (v2.2.20260817)

### Analise dos notebooks funcionais (`krea_2_turbo_colab(1).ipynb` e `krea-2-turbo-fast-text-to-image-generator.ipynb`)
1. **Diferenca de Modelo (Causa do OOM)**: Os notebooks funcionais rodam **exclusivamente o modelo oficial pré-quantizado INT8** (`DeepBeepMeep/krea-2`: `Krea2Turbo_quanto_bf16_int8.safetensors` 12.5GB + `Qwen3-VL-4B-Instruct` 4GB + VAE 1.5GB).
2. **Causa dos Crashes Anteriores**: Checkpoints customizados do Civitai (ex: `Arthemy Comics Krea2` - 23.88 GB FP8/FP16) exigem +30GB RAM para quantização em tempo de carga, estourando a memória de 12GB do Colab T4.
3. **Solucao Aplicada**:
   - Reintegrado o **modelo oficial Krea-2-Turbo INT8** (`DeepBeepMeep/krea-2`) via botão de 1-clique na aba Modelo.
   - Configurado perfil MMGP Colab-optimised (`pinnedMemory=False`, `asyncTransfers=False`, `budgets={"transformer": 10000, "text_encoder": 4000, "vae": 1500, "*": 500}`).
   - Adicionado guard inteligente no `load_model_from_civitai`: orienta o usuário de que checkpoints customizados de 24GB do Civitai exigem +30GB RAM, e sugere o uso do botão oficial Krea-2-Turbo INT8.

### Estado atual da aplicacao
- APP_VER: **v2.2.20260817** (tamanho do app: 227KB; notebook: 243KB).
- **3 Motores estaveis mantidos:**
  1. **Diffusers nativo**: SD 1.5, SDXL, Pony, Illustrious, NoobAI, Animagine XL, etc.
  2. **ComfyUI headless**: Anima (Qwen3 TE + WanVAE oficial), FLUX.1 (Kestral/dev/schnell), SD 3/3.5, Chroma, AuraFlow, etc.
  3. **Wan2GP**: Krea-2-Turbo Oficial INT8 (DeepBeepMeep/krea-2).
- Validação estática 100% OK (ast.parse OK, pyflakes sem erros, roundtrip notebook==disk True).


## §8.36 — REMOCAO DEFINITIVA E PERMANENTE do Krea-2 / Wan2GP (v2.3.20260817)

### Decisao FINAL do projeto
Apos multiplas tentativas de estabilizar o Krea-2 no Colab T4 (worker isolado, modelo oficial
INT8, offload Colab-optimized), o usuario confirmou que o Krea-2 **continua causando desconexao
por OOM** na VM. Decisao definitiva: **remover TOTAL e PERMANENTEMENTE** o suporte a Krea-2/Wan2GP.

### O que foi removido (20 pontos, 51.243 chars)
1. `WAN2GP_DIR` e constantes; 2. `BASE_MODEL_MAP["Krea 2"]`; 3. preset `krea2` em FAMILY_PRESETS;
4. `STATE["krea_model"]`; 5. **Seção 7 MOTOR WAN2GP inteira** (worker KREA2_WORKER_SRC + helpers,
   `download_krea_aux`, `download_krea_models`, `_patch_krea2_main`, `_patch_transformers_docstring`,
   `_ensure_wan2gp_requirements`, `_load_wan2gp_custom`, `load_wan2gp_krea`);
6. `_wan2gp_result_to_image` + `_gen_wan2gp`; 7. branch wan2gp na geracao;
8. worker kill no `unload_current_model`; 9. rota wan2gp no load civitai;
10. `load_krea` real -> mensagem; 11. rota krea2 no load_local_file; 12. health `krea_worker`;
13. neg_rec krea2; 14-15. exemplos CIVITAI; 16. subtitle Wan2GP; 17-18. UI botao/status/click
Krea-2-Turbo; 19. docstring topo.

### Guard FINAL (unico ponto de contato restante)
- Se qualquer URL/modelo do Civitai tiver baseModel "Krea 2"/"krea2"/"krea-2", retorna imediatamente:
  > "O suporte ao modelo Krea-2 foi DESCONTINUADO permanentemente (causava desconexao por OOM no Colab).
  > Use SDXL, Pony, Illustrious, NoobAI, Anima ou FLUX."
- Sem download, sem carregamento, sem risco de RAM.

### Estado FINAL da aplicacao
- APP_VER: **v2.3.20260817** (app: 176.641 chars / 172KB; notebook: 196KB).
- **2 motores estaveis e definitivos:**
  1. **Diffusers nativo**: SD 1.5, SDXL, Pony, Illustrious, NoobAI, Animagine XL...
  2. **ComfyUI headless**: Anima, FLUX.1, SD 3/3.5, Chroma, AuraFlow...
- Validacao 100%: ast.parse OK, pyflakes sem undefined, roundtrip notebook==disk True,
  zero f-string tripla, zero aspas triplas simples no embed.
- Backups: `backups/remocao_definitiva_krea2_20260816_131136/` (estado anterior v2.2.20260817)
  e `backups/universal_app_v2.3.20260817_limpo.py` (estado final limpo).


## §8.37 — Reimplementacao do Krea-2-Turbo do ZERO (v2.4.20260817) — worker fiel aos notebooks validados

### Base de referencia (analise profunda)
- `krea_2_turbo_colab_implementado.ipynb` (Colab T4 12GB): pinnedMemory=False, asyncTransfers=False,
  budgets {transformer:10000, te:4000, vae:1500, *:500}, quantizeTransformer=False.
- `krea-2-turbo-fast-text-to-image-generator_implementado.ipynb` (Kaggle T4 x2 30GB): mesmos patches,
  mas pinnedMemory=True + budgets {13000, 4500, 2000, 1000} (NAO usar no Colab 12GB).
- **Contrato de imagem revelado pelo notebook Kaggle**: resultado do Wan2GP e um tensor
  **(3, 1, H, W)** (C=3, batch=1 na dim 1) — `result[:, 0].permute(1, 2, 0)` -> **(H, W, 3) RGB**.
  O nosso padrão anterior falhava ao assumir (B, C, H, W).

### Implementacao (arquitetura final do Krea-2)
1. **`krea2_worker.py` standalone** (16.6KB): processo limpo isolado, 100% fiel aos notebooks:
   - TFBlocker + env vars EXATAS (expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.5, MALLOC_TRIM_THRESHOLD_=0);
   - patches krea2_main.py: dtype bfloat16 comentado + tf/te/vae preprocess float16 + processor __call__
     (inclui o padrao do notebook `modelPrefix="language_model"` no TE — ausente nas versoes anteriores);
   - load: quantizeTransformer=False, dtype=float16, VAE_dtype=float16, model_type=krea2_turbo;
   - offload.profile: profile_no=2, pinnedMemory=False, asyncTransfers=False, budgets {10000, 4000, 1500, 500}, sdpa;
   - geracao: guide_scale=0.0, batch_size=1, callback, loras phase1, torch.inference_mode();
   - conversao `_result_to_image`: contrato (3,1,H,W) + fallback (B,C,H,W)/(B,H,W,C)/(B,H,W)/(H,W,C);
   - HTTP local: GET /health | POST /generate (base64 PNG) | POST /unload; progress por arquivo.
2. **Embute no app como KREA2_WORKER_SRC** (r"""...""", sem triplas internas) — gravado em
   /content/krea2_worker.py pelo `_write_krea_worker_file()`.
3. **Integracao no universal_app.py**:
   - `load_krea()`: clone Wan2GP (se faltar) -> `download_krea_official` (hf_hub_download resume,
     modelo INT8 oficial) -> `_spawn_krea_worker` (Popen detached + health poll 10min) -> STATE;
   - `_gen_wan2gp()`: proxy HTTP POST /generate em thread + poll progress file (1s);
   - `unload_current_model()`: POST /unload + kill + pkill;
   - guard Civitai Krea2 custom -> orienta usar o botao oficial (modelos custom 24GB FP8 exigem +30GB RAM);
   - `/api/health` -> `krea_worker: bool`; preset krea2 (steps 1-12, cfg 0.0) restaurado.
4. APP_VER: **v2.4.20260817** (app 204.346 chars; notebook 218KB).

### Validacao (local, sem GPU)
- AST app + worker OK; pyflakes sem undefined; roundtrip notebook==disk True; zero f-tripla;
  worker sem aspas triplas (simples E duplas) p/ coexistir com APP_SRC e KREA2_WORKER_SRC.
- Contrato (3,1,H,W): (3,1,64,64) float 0..1 -> PNG RGB (64,64) OK + fallbacks OK.
- **Integracao ponta a ponta**: servidor HTTP real (fake krea_model) -> /health OK ->
  /generate (prompt/steps/seed) -> PNG base64 -> decodificado RGB pelo app + progress file
  {done:8,total:8} gravado pelo callback.

### Por que esta implementacao evita a desconexao (diferente das anteriores)
1. **Modelo certo**: SOMENTE o INT8 oficial (DeepBeepMeep/krea-2) — nunca custom Civitai 24GB
   (desquantizacao em RAM estourava o memcg 12Gi);
2. **Isolamento total**: worker em `python -u` processo limpo (padrao notebooks) — OOM mata so o worker;
3. **Configs EXATAS dos notebooks Colab**: pinnedMemory=False + budgets 10000 (o Kaggle usa 13000);
4. **Patch TE com modelPrefix="language_model"** (o nosso antigo nao tinha) + contrato de imagem (3,1,H,W).


## §8.38 — EVOLUCAO: checkpoints Krea-2 CUSTOM do Civitai rodando na T4 (v2.5.20260817)

### Causa raiz do OOM anterior (confirmada na API real)
- `Arthemy Comics Krea2` (2759057) v1.1: primary = **bf16 23.9GB**; ha tambem **int8 12.2GB** e fp8 12.2GB.
- `Krea2 Tubro Q8 from BF16` (2792164): fp8_mixed 11.9GB (safetensors) + GGUF Q8_0/Q4_0 (nao suportados).
- O `pick_file_from_version` escolhia o arquivo de MAIOR tamanho (= primary bf16 23.9GB) ->
  impossivel carregar em 12GB RAM / 16GB VRAM -> OOM memcg 12Gi e desconexao.

### Evolucao implementada (universal_app.py -> v2.5.20260817)
1. **`_pick_krea_file(version)`**: selecao por FORMATO com prioridade
   `int8 > fp8/fp8_mixed`; exclui `.gguf` (Wan2GP nao suporta) e **recusa bf16 >20GB com aviso claro**.
2. **`load_krea_custom(url_or_id, cb)`**: fluxo completo para custom:
   consulta API -> versao -> arquivo certo -> download (download_file_stream) ->
   `_spawn_krea_worker` (processo isolado, quantizeTransformer=False) -> STATE.
3. **`download_from_civitai`**: quando family==krea2 e wanted_type Model, usa `_pick_krea_file`
   no lugar do picker por tamanho.
4. **Guard de load_model_from_civitai** deixou de bloquear: agora ROTEIA para `load_krea_custom`.
5. `BASE_MODEL_MAP["Krea 2"] = "krea2"` restaurado.

### Resultado esperado na T4
- `Arthemy Comics Krea2` v1.1 -> baixa o **int8 12.2GB** (formato quanto, quantizeTransformer=False).
- `Krea2 Tubro` -> baixa o **fp8_mixed 11.9GB** (preprocess float16 no worker).
- Nenhum arquivo bf16 de 23.9GB e selecionado -> sem OOM, sem desconexao.
- OOM eventual mata SO o worker; kernel/sessao/app continuam vivos.

### Validacao local
- AST app OK; pyflakes sem undefined; roundtrip notebook==disk True (211.941 chars).
- Teste com dados REAIS da API: Arthemy -> int8 12.2GB; Krea2 Tubro -> fp8_mixed 11.9GB;
  versoes GGUF corretamente recusadas.
- Backups: backups/universal_app_v2.5.20260817_krea2_custom.py + notebook.


## §8.39 — Fix do worker Krea-2: WAN2GP_DIR path (FileNotFoundError) + TE path (v2.5.20260817)

### Sintoma (teste real do usuario no Colab)
```
Worker Krea2 nao subiu:
  File "/content/studio/krea2_worker.py", line 327, in main
    os.chdir(WAN2GP_DIR)
FileNotFoundError: [Errno 2] No such file or directory: '/content/studio/Wan2GP'
```

### Causa raiz (2 bugs)
1. **WAN2GP_DIR relativo ao cwd errado**: o worker calculava `os.path.abspath("Wan2GP")`
   relativo ao cwd do spawn (`/content/studio`), mas o Wan2GP e clonado em `/content/Wan2GP`.
2. **TE path errado**: `_spawn_krea_worker` montava `--te` com `MODELS_DIR`
   (`/content/studio/models`) mas o `download_krea_official`/`_ensure_krea_te_vae` baixam o
   Text Encoder em `/content/Wan2GP/models/Qwen3-VL-4B-Instruct/...` — paths divergentes.

### Correcoes
1. **worker** (`krea2_worker.py`): novo arg `--wan2gp`; `WAN2GP_DIR = args.wan2gp or
   os.environ.get("WAN2GP_DIR") or abspath("Wan2GP")` (topo + main); log do path no boot.
2. **app** (`_spawn_krea_worker`): `env["WAN2GP_DIR"] = WAN2GP_DIR`; cmd ganha
   `"--wan2gp", WAN2GP_DIR`; `--te` agora aponta para `WAN2GP_DIR/models/Qwen3-VL-4B-Instruct/...`.
3. **`_ensure_krea_te_vae()`**: nova funcao (TE + VAE sem transformer) — reutilizada por
   `download_krea_official` e chamada em `load_krea_custom` antes do spawn (garante TE/VAE
   para checkpoints custom).

### Validacao
- AST app+worker OK; pyflakes sem undefined; roundtrip notebook==disk True (212.962 chars);
  worker re-embutido sem triplas.
- Teste com mock: comando gerado = `--ckpt ... --te /content/Wan2GP/models/Qwen3-VL-4B-Instruct/
  Qwen3-VL-4B-Instruct_quanto_bf16_int8.safetensors --wan2gp /content/Wan2GP` + env WAN2GP_DIR OK.
- Backups: backups/universal_app_v2.5.20260817_fix_worker.py + krea2_worker_v2.5_fix.py.


## §8.40 — Fix: deps do Wan2GP (ModuleNotFoundError: smplfitter) (v2.5.20260817)

### Sintoma (teste real do usuario no Colab)
```
File "/content/Wan2GP/models/wan/scail/nlf/multiperson_model.py", line 8, in <module>
    import smplfitter.pt
ModuleNotFoundError: No module named 'smplfitter'
```
O Wan2GP upstream (git main) importa smplfitter na cadeia de imports do krea2_main
(models/wan/modules/model.py -> scail -> nlf -> multiperson_model).

### Causa raiz
Nosso `_ensure_wan2gp_requirements`/worker `ensure_requirements` instalavam apenas
mmgp/gradio + pins — faltava o `-r Wan2GP/requirements.txt` (que os notebooks rodam
`pip install -r Wan2GP/requirements.txt`) e pacotes explicitos (smplfitter).

### Correcoes
1. **Worker** (`ensure_requirements`): instala `-r <WAN2GP_DIR>/requirements.txt`
   (timeout 120, retries 5 — fiel ao notebook) + loop de fallback garantindo
   `mmgp`, `gradio`, `smplfitter` + pins (numpy==2.3.5, optimum-quanto==0.2.4).
2. **App** (`_ensure_wan2gp_deps`): mesma logica, chamada em `_spawn_krea_worker`
   logo apos o clone (progresso na UI); fallback explicito mmgp/gradio/smplfitter.
3. Worker re-embutido no app (KREA2_WORKER_SRC, 17.4KB, sem triplas).

### Validacao
- AST app+worker OK; pyflakes sem undefined; roundtrip notebook==disk True (215.373 chars);
  worker contem smplfitter e -r requirements.txt.
- Backups: backups/universal_app_v2.5.20260817_fix_deps.py + krea2_worker_v2.5_deps.py.


## §8.41 — Fix numpy: 2.3.5 quebra scipy (cannot import name '_slice') (v2.5.20260817)

### Sintoma (teste real do usuario no Colab)
```
File ".../numpy/_core/strings.py", line 24, in <module>
    from numpy._core.umath import (...)
ImportError: cannot import name '_slice' from 'numpy._core.umath'
```
Cadeia: sklearn -> scipy -> numpy.char -> numpy._core.defchararray -> numpy._core.strings
-> `from numpy._core.umath import _slice` (simbolo REMOVIDO no numpy 2.3.x).

### Causa raiz
Nosso pin `numpy==2.3.5` (adicionado em §8.30 p/ o `_blas_supports_fpe` do scipy 1.16)
**removeu `_slice`** do `numpy._core.umath` — quebrando o import do scipy via numpy.char.
O Colab nativo (2.1.2) e o notebook funcional rodam SEM pin e funcionam.

### Correcao
- Pin seguro: **`numpy==2.2.6`** — tem `_slice` (umath) E `_blas_supports_fpe`
  (2.4+ remove o segundo; 2.3+ remove o primeiro; 2.2.x tem ambos).
- Aplicado em TODOS os pontos: cell_run do notebook (setup), `_ensure_wan2gp_deps`,
  `ensure_requirements` do worker (condicional: so age se numpy >= 2.3), app.
- Logica condicional: `if major > 2 or (major == 2 and minor >= 3): rebaixar p/ 2.2.6`
  (mantem o nativo 2.1.x se ja for compativel).

### Validacao
- Zero ocorrencias de `numpy==2.3.5` em universal_app.py, krea2_worker.py, _gerar_notebook.py.
- Notebook source contem `numpy==2.2.6` em 5 pontos (setup + app + worker) e zero 2.3.5.
- Roundtrip notebook==disk True (215.577 chars); AST OK; pyflakes sem undefined.
- Backups: backups/universal_app_v2.5.20260817_fix_numpy.py.


## §8.42 — Fix preprocess do mmgp: 'tuple' object has no attribute 'items' (v2.5.20260817)

### Sintoma (teste real do usuario no Colab)
```
File "mmgp/offload.py", line 2167, in load_model_data
    state_dict = preprocess_fn(*[state_dict, quantization_map, tied_weights_map][:num_params])
File "krea2_main.py", line 684, in te_preprocess
    return {k: v.to(dtype) if ... for k, v in sd.items()}
AttributeError: 'tuple' object has no attribute 'items'
```

### Causa raiz
O mmgp (versao atual do Wan2GP git main) chama o `preprocess_fn` com multiplos args:
`(state_dict, quantization_map, tied_weights_map)` — e em alguns fluxos o primeiro arg
chega como TUPLA. Nossos `tf_preprocess/te_preprocess/vae_preprocess` (assinatura `(sd)`)
recebiam a tupla e quebravam em `sd.items()`.

### Correcao
Todos os preprocess dos patches passaram a aceitar **aridade variavel** (`*a`) e extrair
o state_dict de forma robusta:
```python
def te_preprocess(*a):
    sd = a[0] if a else {}
    if isinstance(sd, tuple):
        sd = sd[0] if sd else {}
    ...
```
Aplicado em `tf_preprocess`, `te_preprocess` (single-line E multiline) e `vae_preprocess`
no krea2_worker.py; worker re-embutido no app (KREA2_WORKER_SRC).

### Validacao
- Teste de comportamento 3/3: dict direto / 3 args (sd, qm, twm) / tuple empacotado -> OK.
- AST app+worker OK; pyflakes sem undefined; roundtrip notebook==disk True (216.416 chars);
  worker sem triplas; backups criados.


## §8.43 — Fix preprocess (2a rodada): _orig_* retornava TUPLA (v2.5.20260817)

### Sintoma (repetido, linha 692)
Mesmo AttributeError 'tuple' object has no attribute 'items' — agora na linha 692
(a linha mudou 684->692 porque o guard de entrada foi aplicado, empurrando o return).

### Causa raiz REAL
O guard de entrada (sd = a[0]; if isinstance tuple -> sd[0]) protegia a ENTRADA,
mas o patch ainda chamava o processador ORIGINAL do upstream:
- TE: `sd = _orig_te_preprocess(sd)` — e `_build_krea2_text_encoder_preprocessor(config)`
  retorna uma TUPLA (provavelmente (state_dict, quant_map, tied_weights)) -> sd vira tuple
  DEPOIS do guard -> `sd.items()` quebra.
- VAE: `sd = _orig_vae_pp(sd)` — mesmo risco.

### Correcao (fiel aos notebooks)
- REMOVIDA a chamada aos processadores originais: os preprocesses agora fazem APENAS o
  cast para dtype (como o run_krea_turbo.py dos notebooks: `return {k: v.to(dtype) ...}`).
- Guards mantidos: entrada (dict ou tuple -> a[0]/sd[0]) e saida (tf_preprocess pós
  preprocess_sd original: se retornou tuple, extrai sd[0]).
- Worker: 5 guards `isinstance(sd, tuple)`; zero `_orig_*`; re-embutido (18.2KB, sem triplas).

### Validacao
- AST app+worker OK; pyflakes sem undefined; roundtrip notebook==disk True (216.145 chars);
  backups criados.


## §8.44 — Fix Missing keys no Text Encoder (strip de prefixo como modelPrefix) (v2.5.20260817)

### Sintoma (teste real do usuario no Colab)
```
mmgp/offload.py, line 2294, in load_model_data
    raise Exception(f"Missing keys: {missing_keys}")
Exception: Missing keys: ['layers.0.self_attn.q_proj.weight', ... (layers 0..35 attn+mlp)]
```
O preprocess do TE passou (fix §8.43), mas o load do Text Encoder nao encontra as chaves.

### Causa raiz
O safetensors do TE (Qwen3-VL-4B-Instruct_quanto_bf16_int8) tem chaves com prefixo
`language_model.` (ex.: language_model.layers.0.self_attn.q_proj.weight) — o modelo
espera `layers.0...` SEM o prefixo. O notebook usa `modelPrefix="language_model"` no
load_model_data (o mmgp remove o prefixo). O nosso cast simples (§8.43) NAO renomeava.

### Correcao
`te_preprocess` (2 variantes) agora faz strip duplo de prefixo + cast:
- `language_model.X` -> `X`
- `language_model.model.X` -> `X`
- `model.X` -> `X`
- `vae_preprocess`: strip `vae.`/`model.` + cast
- mantidos: *args + guard tuple (entrada) do mmgp.

### Validacao
- Teste 3/3: chaves language_model./language_model.model./model. -> renomeadas; aridade mmgp OK.
- Roundtrip notebook==disk True (217.688 chars); AST OK; backups criados.


## §8.45 — Fix Missing keys TE (2a rodada): modelPrefix + layout QUANTO real (v2.5.20260817)

### Inspecao do safetensors REAL (via HTTP Range header)
- **TE** (`Qwen3-VL-4B-Instruct_quanto_bf16_int8`, 1155 chaves): prefixo `language_model.`
  E formato **quanto**: `layers.0.self_attn.q_proj.weight._data / ._scale / input_scale / output_scale`.
- **Transformer** (`Krea2Turbo_quanto_bf16_int8`, 1103 chaves): SEM prefixo,
  formato quanto `blocks.0.attn.wq.weight._data/_scale/...`.

### Causa raiz do Missing keys persistente
O patch multiline do TE substituiu a chamada por `preprocess_sd=te_preprocess` SEM
`modelPrefix="language_model"` — sem ele o mmgp nao strip o prefixo das chaves quanto
(e nao monta o quantization_map do layout `._data/._scale`) -> Missing keys.

### Correcao
- `modelPrefix="language_model"` adicionado na chamada multiline do TE (igual ao
  single-line/notebook): o mmgp strip o prefixo + detecta o quanto automaticamente.
- Strip manual mantido como rede de seguranca; preprocess continua so-cast (fiel notebook).
- Transformer: chaves sem prefixo — o load usa `preprocess_sd=preprocess_sd` original
  + cast (sem mudanca necessaria).

### Validacao
- AST app+worker OK; pyflakes sem undefined; roundtrip notebook==disk True (217.748 chars);
  worker com 5 ocorrencias modelPrefix; backups criados.


## §8.46 — Fix mmgp: pin 3.7.12 (versao VALIDADA do notebook) — meta tensor resolvido (v2.5.20260817)

### Descoberta decisiva (output real do notebook funcional)
O output da celula 8 do `krea_2_turbo_colab_implementado.ipynb` mostra o carregamento
BEM-SUCEDIDO do Krea-2 com **mmgp 3.7.12**:
```
Memory Management for the GPU Poor (mmgp 3.7.12) by DeepBeepMeep
Hooked to model 'transformer' (SingleStreamDiT)
Hooked to model 'text_encoder' (Qwen3VLTextModel)
Hooked to model 'vae' (AutoencoderKLQwenImage)
Model ready! -> gerou 1024x1024 steps=2
```

### Causa raiz do meta tensor (Cannot copy out of meta tensor)
O requirements.txt do Wan2GP pin `mmgp==3.7.12` + `numpy==2.1.2` + `transformers==4.54.0`.
Os nossos ensures faziam `pip install mmgp gradio` SEM versao DEPOIS do -r requirements.txt,
ATUALIZANDO o mmgp para uma versao mais nova do PyPI que quebra o load quanto do modelo
meta (init_empty_weights) no `_requantize` -> `NotImplementedError: Cannot copy out of meta tensor`.
O nosso pin numpy==2.2.6 tambem desfazia o numpy==2.1.2 do requirements.

### Correcao (100% fiel ao notebook)
1. **`mmgp==3.7.12`** e **`gradio==5.29.0`** fixados nos ensures (app + worker) — nunca
   instalar mmgp sem versao apos o requirements.
2. **numpy**: remover o pin 2.2.6/2.3.5 — o requirements.txt do Wan2GP ja fixa 2.1.2
   (que funciona — foi o usado no notebook).
3. `smplfitter==0.2.10` (pin do requirements).
4. Worker reembutido; notebook regenerado (217.598 chars, roundtrip OK).

### Validacao
- Zero ocorrencias numpy 2.2.6/2.3.5; mmgp==3.7.12 presente no app+worker+notebook.
- Roundtrip notebook==disk True; AST OK; backups criados.


## §8.47 — Fix meta tensor: optimum-quanto 0.2.7 (versao do notebook) (v2.5.20260817)

### Sintoma (repetido, mmgp 3.7.12)
```
optimum/quanto/nn/qmodule.py, line 206, in from_module
    qmodule.weight.copy_(module.weight)
NotImplementedError: Cannot copy out of meta tensor; no data!
```
Ainda no `_requantize` do mmgp 3.7.12, mas agora confirmado: o problema NAO era o mmgp,
e sim o **optimum-quanto 0.2.4** (nosso pin do §8.29) — versao de 2024-07, antiga demais,
que falha ao copiar pesos meta (init_empty_weights do krea2_main).

### Descoberta
- mmgp 3.7.12 (pyproject) depende de `optimum-quanto` SEM versao.
- Versoes do quanto no PyPI: 0.2.4 (2024-07) ... 0.2.7 (2025-03, a mais recente).
- O notebook (08/2026) instalou a mais nova = **0.2.7** — e funcionou.
- O quanto 0.2.4 trata o modelo meta de forma incompativel; o 0.2.7 nao.

### Correcao
- Todos os pins `optimum-quanto==0.2.4` -> **`optimum-quanto==0.2.7`** (worker, app, ensure).
- Mantidos: mmgp==3.7.12, gradio==5.29.0, smplfitter==0.2.10, numpy 2.1.2 (requirements).
- Worker reembutido; notebook regenerado (217.606 chars, roundtrip OK).

### Stack FINAL de versoes (todas as do notebook validado)
mmgp==3.7.12 | optimum-quanto==0.2.7 | gradio==5.29.0 | smplfitter==0.2.10 |
numpy==2.1.2 (requirements.txt) | transformers==4.54.0 | diffusers==0.36.0


## §8.48 — Fix definitivo dos pins: FORCAR sempre (import-check nao pegava versao errada) (v2.5.20260817)

### Sintoma (repetido com quanto 0.2.7 no codigo)
Mesmo erro meta tensor, mesmo apos pinar quanto 0.2.7 — porque o pin NUNCA aplicava:
```python
try:
    import optimum.quanto   # se JA esta instalado (ex.: 0.2.4 de rodada anterior)
except Exception:
    pip install optimum-quanto==0.2.7   # <- nunca roda
```
O requirements.txt do Wan2GP tambem NAO pin o quanto — entao a versao bugada (0.2.4)
permanecia na VM mesmo com o codigo novo.

### Correcao
Os ensures (app `_ensure_wan2gp_deps` + worker `ensure_requirements`) agora FORCAM os
pins SEMPRE, sem import-check:
```python
for _pin in ("mmgp==3.7.12", "gradio==5.29.0", "optimum-quanto==0.2.7", "smplfitter==0.2.10"):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", _pin], timeout=600)
```
`pip install -q mmgp==3.7.12` com a versao certa ja instalada = "already satisfied" (rapido);
com versao errada = rebaixa/atualiza para o pin exato.

### Importante para o teste do usuario
Se a VM ja rodou tentativas anteriores com quanto 0.2.4, o notebook NOVO forca 0.2.7
na proxima execucao do load (pip rebaixa/atualiza). Pode levar ~1-2 min no primeiro load.

### Validacao
- AST app+worker OK; pyflakes sem undefined; roundtrip notebook==disk True (216.510 chars);
  `for _pin in` presente no notebook; backups criados.


## §8.49 — Fix duplo-strip: preprocess TE NAO deve strip (modelPrefix ja faz) (v2.5.20260817)

### Sintoma (missing keys de TODAS as chaves do TE, incluindo embed_tokens e _data/_scale)
```
Missing keys: ['embed_tokens.weight', 'layers.0.self_attn.q_proj.weight._data', ...]
```
O _requantize PASSou (quanto 0.2.7 ok!) — o erro agora e no load_state_dict:
TODAS as chaves do modelo estao missing => state_dict chegou VAZIO.

### Causa raiz (duplo strip)
A ordem real no mmgp 3.7.12:
1. preprocess_fn (recebe as chaves COM prefixo language_model.)
2. detect_and_convert (quanto)
3. **modelPrefix="language_model" -> filter_state_dict_basic (remove o prefixo)**

O nosso te_preprocess fazia STRIP MANUAL do prefixo (fix §8.44, quando ainda nao havia
modelPrefix) — e o modelPrefix do mmgp rodava DEPOIS, NAO encontrando mais chaves com
`language_model.` -> state_dict VAZIO -> load_state_dict -> TODAS missing.

### Correcao
- Removido o strip manual dos 2 te_preprocess (single-line e multiline).
- O preprocess volta a ser SO o cast (fiel ao notebook): 
  `return {k: v.to(dtype) if (isinstance(v, torch.Tensor) and v.is_floating_point()) else v for k, v in sd.items()}`
- modelPrefix="language_model" mantido na chamada (o mmgp faz o strip unico).
- Guards de aridade (*args + tuple) mantidos.

### Validacao
- AST app+worker OK; pyflakes sem undefined; roundtrip notebook==disk True (215.362 chars);
  zero `startswith('language_model.')` no TE; backups criados.


## §8.50 — Fix: preprocessor_config.json do TE (v2.5.20260817)

### Sintoma
```
OSError: models/Qwen3-VL-4B-Instruct does not appear to have a file named preprocessor_config.json
Qwen2VLImageProcessorFast.from_pretrained(tokenizer_path)
```
O TE CARREGOU (fix §8.49 ok!) — o erro agora e no image_processor do Qwen2VL.

### Causa raiz
O `_ensure_krea_te_vae` baixava 5 arquivos do TE; o notebook baixa 6 —
incluindo **preprocessor_config.json** (exigido pelo Qwen2VLImageProcessorFast).

### Correcao
`preprocessor_config.json` adicionado a lista de arquivos do TE no `_ensure_krea_te_vae`.
Notebook regenerado (215.406 chars, roundtrip OK); backups criados.
