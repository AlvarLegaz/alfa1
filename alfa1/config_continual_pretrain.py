import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Sancho - CONTINUAL PRETRAINING BPE  |  MODELO v2 (candidata L, ~520M)
# ============================================================
#
# Arquitectura v2 tipo Llama/Qwen: RoPE + RMSNorm + SwiGLU + GQA.
#   E1536 / L18 / H24 / KV6, weight tying activado.
#   Exporta a LlamaForCausalLM (GGUF directo con llama.cpp).
#
# Esta NO es la corrida original de pretraining desde cero. Es una fase de
# continual pretraining que parte de un checkpoint YA entrenado
# (INIT_FROM_CHECKPOINT mas abajo) y sigue entrenando sobre la mezcla
# nueva es/eng/prog generada con mezclar_shards.py (~42.6B tokens en
# train.bin, dominio-ancla 'es' al 100%, eng/prog casi al 100% sin
# repetirse). Presupuesto: ~1 pasada completa por volumen sobre ese
# train.bin (MAX_ITERS x tokens/iter, ver mas abajo).
#
# CAMBIO DE HARDWARE: esta corrida NO va al H100 de Hyperstack (esa fue la
# corrida original hasta iter 52000). Va LOCAL, en la RTX 5090 de Falcata-ws
# (32GB VRAM, frente a los 80GB del H100). BATCH_SIZE y GRAD_ACCUM_STEPS
# estan recalibrados para esa VRAM: BATCH_SIZE=16 es zona segura confirmada
# en el full SFT sobre este mismo modelo (BATCH_SIZE=8 funciono sin
# problema; 24 y 48 desbordaron VRAM a RAM del sistema). Si nvidia-smi
# muestra margen de sobra en las primeras iteraciones, se puede subir.
#
# DIFERENCIA CLAVE CON v1:
# v1 era SanchoGPT clasico (wpe + LayerNorm + GELU). v2 NO es continuable
# desde un checkpoint v1: es un pretraining desde cero con model_v2.py.
# El conteo de parametros cambia (sin wpe se ahorra, SwiGLU anade un
# proyector); el numero exacto lo imprime train_continual_pretrain_bpe.py
# al arrancar.
#
# NOTA SOBRE ESCALAR:
# La arquitectura NO cambia respecto al checkpoint de partida (misma
# N_EMBD, N_LAYER, N_HEAD, N_KV_HEAD, FFN_HIDDEN): si se tocaran, el
# load_state_dict al cargar INIT_FROM_CHECKPOINT fallaria.
# ============================================================

# El corpus se construye fuera y se tokeniza a train.bin / val.bin con
# preparar_datos.py. La mezcla de repeticiones por fuente (Wikipedia xN,
# CulturaX 1x, etc.) se decide al construir el corpus, no aqui.
# Estas rutas son referencia; el entrenamiento lee de los .bin via memmap.
PRETRAIN_FILES = [
    f"/mnt/data/pretrain/salida_20B/corpus_25B_{i:06d}.txt"
    for i in range(1, 173)
]

TOKENIZER_TRAIN_FILES = [
    # El tokenizer se entrena sobre un SUBCONJUNTO representativo del corpus
    # (espanol + ingles + algo de codigo Python/Arduino), no sobre los 20B.
    "/mnt/data/pretrain/muestra_tokens/muestra_tokenizer.txt",
]

# ============================================================
# Tokenizador  (CONGELADO - identico para 05B 1B y 4B)
# ============================================================

MODO_LECTURA_CORPUS = "lineas"   # o "parrafos" segun tu corpus
LIMPIAR_AGRESIVO = False          # True para v1 (solo prosa); False cuando metas codigo en v2


TOKENIZER_TYPE = "bpe"
BPE_VOCAB_SIZE = 48000
BPE_MIN_PAIR_FREQ = 2

# ============================================================
# Split del dataset
# ============================================================

# A esta escala basta un val pequeno; lo importante es que sea estable.
TRAIN_RATIO = 0.999
MAX_CHARS_BLOQUE = 4000

# Rutas de los binarios: la mezcla nueva generada con mezclar_shards.py
# (dominios es/eng/prog, patron proporcional ~40:27:16, dominio-ancla es).
DATA_BIN_DIR = r"C:\Users\alegaz\Desktop\PROYECTO SANCHO\Generar_corpus\Corpus_continucacion_bi\mezcla_continual_pretraining"
TRAIN_BIN_PATH = DATA_BIN_DIR + r"\train.bin"
VAL_BIN_PATH = DATA_BIN_DIR + r"\val.bin"

# OJO: mezclar_shards.py NO genera tokenizer_state.json (solo produce
# train.bin/val.bin). El tokenizer 48k esta congelado desde el pretraining
# original, asi que apunta al tokenizer_state.json ya existente del corpus
# original. Cambia esta ruta si lo tienes en otro sitio, o cópialo dentro
# de DATA_BIN_DIR si prefieres tenerlo todo junto.
TOKENIZER_STATE_PATH = r"C:\Users\alegaz\Desktop\PROYECTO SANCHO\MODELOS\Alfa1_05B\checkpoints\tokenizer_state.json"

# ============================================================
# Arquitectura — candidata L (v2 Llama/Qwen)
# ============================================================
# E1536 / L18 / H24. n/cabeza = 1536/24 = 64. GQA y SwiGLU mas abajo.


BLOCK_SIZE = 512

N_EMBD = 1536
N_HEAD = 24          # 64 dim/cabeza
N_LAYER = 18
DROPOUT = 0.0        # a 1B con corpus grande, dropout 0 es lo habitual
                     # (la regularizacion la da el volumen de datos)

# ============================================================
# Arquitectura v2 — tipo Llama/Qwen (RoPE + RMSNorm + SwiGLU + GQA)
# ============================================================
#
# Estos parametros los lee train_pretrain_bpe.py via globals().get(), y los
# consume model_v2.SanchoGPT. NO existen en la v1 (wpe + LayerNorm + GELU).
#
# IMPORTANTE: v2 NO es continuable desde un checkpoint v1. La arquitectura
# cambia (no hay position_embedding_table, las normas son RMSNorm, el FFN es
# SwiGLU), asi que el pretraining arranca DESDE CERO. El tokenizer 48k sigue
# congelado, los .bin se reutilizan o regeneran con el mismo vocab.
# ============================================================

# GQA (Grouped-Query Attention): N_HEAD cabezas Q comparten N_KV_HEAD cabezas
# K/V. Reduce VRAM del KV-cache (~4x aqui) sin apenas perder calidad.
# N_HEAD debe ser divisible por N_KV_HEAD -> 24 / 6 = 4 grupos.
# Si se pusiera None, model_v2 cae a MHA plano (N_KV_HEAD = N_HEAD = 24).
N_KV_HEAD = 6

# Weight tying: lm_head comparte matriz con la tabla de embeddings (V x E).
# Ahorra ~135M params a vocab 48k y exporta como tie_word_embeddings=True.
TIE_WEIGHTS = True

# RoPE: base de las frecuencias rotatorias. 10000 es el estandar Llama/Qwen
# y es de sobra para BLOCK_SIZE=512. Solo se sube (p.ej. 1e6) si se extiende
# el contexto a miles de tokens; aqui no hace falta.
ROPE_THETA = 10000.0

# SwiGLU: dimension oculta del FFN. Llama usa ~8/3*N_EMBD redondeado a 256
# para igualar parametros frente al FFN GELU 4x (SwiGLU tiene 3 matrices en
# vez de 2). Para N_EMBD=1536 -> 8/3*1536 = 4096 exacto.
# Se fija explicito (en vez de dejar que model_v2 lo calcule) para que las
# tablas de arquitectura del paper SEPLN cuadren con el modelo real.
FFN_HIDDEN = 4096

# RMSNorm epsilon. 1e-5 es el valor de Llama. 1e-6 es el de Qwen2; ambos
# valen, la diferencia es despreciable. Lo dejamos en el de Llama por ser
# el target de export.
RMS_EPS = 1e-5

# ============================================================
# Entrenamiento  
# ============================================================

# Tokens / iteraciones (presupuesto = 1 pasada completa por volumen sobre
# el train.bin de la mezcla continual, 42,612,559,872 tokens).
#
# BATCH_SIZE baja de 56 (H100 80GB) a 20 (RTX 5090 32GB). Medido en la 5090:
#   batch 16 / accum 63 -> ~47,325 tok/s, 21.13/31.84 GB VRAM
#   batch 20 / accum 50 -> ~50,210 tok/s, 23.62/31.84 GB VRAM  <-- elegido
# No se sube mas: batch 24 ya desbordo VRAM a RAM del sistema en el full SFT
# sobre este mismo modelo/block_size.
BATCH_SIZE = 20
GRAD_ACCUM_STEPS = 50

MAX_ITERS = 83228      # 83228 x 512,000 tok/iter = 42,612,736,000 tokens
                        # vistos = 1 pasada completa por volumen sobre el
                        # train.bin de la mezcla continual (42,612,559,872).
                        # Con muestreo aleatorio con reemplazo (ver get_batch
                        # en train_continual_pretrain_bpe.py), esto da
                        # cobertura esperada de posiciones unicas segun
                        # coupon collector, no un recorrido exhaustivo.

# Modo de torch.compile. "max-autotune-no-cudagraphs" hace benchmark de multiples
# configuraciones de kernel y elige la mas rapida: la primera compilacion
# tarda VARIOS MINUTOS (normal, no esta colgado), pero luego cada iteracion
# va mas rapida. En una corrida de ~10 dias se amortiza de sobra.
# Poner "default" si diera errores de compilacion o faltara Triton.
COMPILE_MODE = "max-autotune-no-cudagraphs"

EVAL_INTERVAL = 500
EVAL_ITERS = 100

GRAD_CLIP = 1.0

# ============================================================
# Learning rate + scheduler (warmup + cosine decay)
# ============================================================
#
# A 1B, el warmup NO es opcional: sin el, los primeros updates con LR alto
# desestabilizan el entrenamiento. El cosine decay lleva el LR hasta un
# minimo al final del presupuesto de tokens.
#
# El scheduler lo aplica train_pretrain_bpe.py leyendo estos valores.
# ============================================================

# LR pico MAS BAJO que en el pretraining original (3e-4): partimos de un
# checkpoint ya entrenado (iter 52000), no de pesos aleatorios, asi que un
# LR pico alto arriesga a desestabilizar/olvidar lo ya aprendido. 1e-4 es
# un punto de partida razonable (~1/3 del original); ajustalo si el train
# loss se dispara al arrancar o si baja demasiado lento.
LEARNING_RATE = 1e-4          # LR pico (tras el warmup)
MIN_LR = 1e-5                 # LR minimo al final del cosine (10% del pico)
WARMUP_ITERS = 1665            # ~2% de MAX_ITERS
LR_DECAY_ITERS = 74905        # ~90% de MAX_ITERS; deja ~10% de cola en MIN_LR
WEIGHT_DECAY = 0.1            # AdamW; estandar a esta escala
BETA1 = 0.9
BETA2 = 0.95                 # 0.95 (no 0.999) es lo habitual en LLMs

# El "recorte de coseno" (RECORTE_LR_ACTIVO) de train_continual_pretrain_bpe.py
# fue una decision puntual de la corrida ORIGINAL (activado en iter 27500 por
# 3 subidas seguidas de val loss). Esta es una corrida nueva desde iter 0:
# lo desactivamos para que use el coseno normal definido arriba.
RECORTE_LR_ACTIVO = False

# ============================================================
# Early stopping
# ============================================================
#
# A esta escala el early stopping casi nunca dispara (no hay overfitting
# con 20B tokens), pero lo dejamos como red de seguridad con patience alta.
# ============================================================

EARLY_STOPPING = True
PATIENCE = 20
MIN_DELTA = 0.001

# ============================================================
# Checkpoints
# ============================================================

CHECKPOINT_PATH = r"C:\Users\alegaz\Desktop\PROYECTO SANCHO\MODELOS\Alfa1_05B\checkpoints\alpha_1_continual_pretrain_40bt.pt"

# Checkpoint de PARTIDA para el continual pretraining: se cargan solo los
# PESOS del modelo (no optimizer/scaler/iteracion/RNG) al arrancar en
# iteracion 0, con LR/scheduler nuevos definidos en este fichero. Esto es
# distinto de RESUME (que retoma una corrida cortada de ESTA MISMA fase,
# con su propio optimizer). Ver train_continual_pretrain_bpe.py.
INIT_FROM_CHECKPOINT = r"C:\Users\alegaz\Desktop\PROYECTO SANCHO\MODELOS\Alfa1_05B\checkpoints\alpha_1_pretrain_05B_best_iter52000.pt"

SEED = 1337

# ============================================================
# Generación de prueba
# ============================================================

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.7
TOP_K = 40
