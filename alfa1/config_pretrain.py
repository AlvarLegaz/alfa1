import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Sancho - Pretraining BPE  |  MODELO v2 (candidata L, ~480-500M)
# ============================================================
#
# Arquitectura v2 tipo Llama/Qwen: RoPE + RMSNorm + SwiGLU + GQA.
#   E1536 / L18 / H24 / KV6, weight tying activado.
#   Exporta a LlamaForCausalLM (GGUF directo con llama.cpp).
# Presupuesto: ~39.16B tokens VISTOS (MAX_ITERS x tokens/iter) = 1 pasada
# completa por volumen sobre train.bin (39,158,243,328 tokens).
#
# Este pretraining corre en CLOUD (H100 80GB), no en la 3070 local.
# El batching y el LR estan calibrados para esa GPU.
#
# DIFERENCIA CLAVE CON v1:
# v1 era SanchoGPT clasico (wpe + LayerNorm + GELU). v2 NO es continuable
# desde un checkpoint v1: es un pretraining desde cero con model_v2.py.
# El conteo de parametros cambia (sin wpe se ahorra, SwiGLU anade un
# proyector); el numero exacto lo imprime train_pretrain_bpe.py al arrancar.
#
# NOTA SOBRE ESCALAR:
# Para cambiar de tamano solo se tocan aqui: N_EMBD, N_LAYER, N_HEAD,
# N_KV_HEAD, FFN_HIDDEN, MAX_ITERS (y revisar WARMUP_ITERS). model_v2.py y
# tokenizer_bpe.py NO cambian. El scheduler lee sus parametros de aqui.
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
BPE_MIN_PAIR_FREQ = 5   # valor REAL usado al entrenar tokenizer_alfa1.json
                        # (subido de 2 a 5 para evitar el OOM del entrenador BPE)

# ============================================================
# Split del dataset
# ============================================================

# A esta escala basta un val pequeno; lo importante es que sea estable.
TRAIN_RATIO = 0.999
MAX_CHARS_BLOQUE = 4000

# Rutas de los binarios (deben coincidir con binarizar_corpus.py, NO con
# preparar_datos.py -- ese pipeline ya no se usa para este corpus)
DATA_BIN_DIR = "C:/Users/alegaz/Desktop/PROYECTO SANCHO/Generar_corpus/corpus_80Bt/bin_alfa1"
TRAIN_BIN_PATH = "C:/Users/alegaz/Desktop/PROYECTO SANCHO/Generar_corpus/corpus_80Bt/bin_alfa1/train.bin"
VAL_BIN_PATH = "D:/Minaya_Corpus_General_v1/val_meseta/val_meseta.bin"  # CAMBIO 2026-08-25: held-out representativo de train_meseta.bin (6.5% codigo vs ~36% del val.bin antiguo)
TOKENIZER_STATE_PATH = "C:/Users/alegaz/Desktop/PROYECTO SANCHO/Generar_corpus/corpus_80Bt/bin_alfa1/tokenizer_state.json"

# ============================================================
# CAMBIO 2026-08-23 (solo usado por train_pretrain_v2.py): corpus
# rebalanceado en dos ficheros (meseta + final, ver
# Generar_corpus/construir_train_dos_fases.py). TRAIN_BIN_PATH (arriba)
# NO se toca -- sigue siendo el train.bin original, para no romper
# train_pretrain_bpe.py (v1) si hiciera falta usarlo. v2 lee estas rutas
# aparte y hace el cambio meseta->final EL SOLO, en caliente, al cruzar
# ITER_CAMBIO_A_FINAL -- no hace falta editar config a mano ni parar el
# entrenamiento para hacer el swap manualmente.
# ============================================================
TRAIN_BIN_PATH_MESETA = "D:/Minaya_Corpus_General_v1/train_rebalanceado/train_meseta.bin"
TRAIN_BIN_PATH_FINAL = "D:/Minaya_Corpus_General_v1/train_rebalanceado/train_final.bin"

# Iteracion en la que train_pretrain_v2.py cambia de train_meseta.bin a
# train_final.bin. Derivado de: MAX_ITERS - (TOKENS_FINAL_TOTAL /
# tokens_por_iteracion) = 78.125 - (5.100.000.000 / 1.024.000) = 78.125 -
# 4.981 = 73.144 -> se activa en la iteracion 73.145 (la siguiente vez que
# el bucle evalua la condicion). Si MAX_ITERS o el presupuesto de la fase
# FINAL en construir_train_dos_fases.py cambian, este numero HAY que
# recalcularlo a mano -- no se deriva automaticamente de otro fichero.
ITER_CAMBIO_A_FINAL = 73_145

# ============================================================
# Arquitectura — candidata L (v2 Llama/Qwen)
# ============================================================
# E1536 / L18 / H24. n/cabeza = 1536/24 = 64. GQA y SwiGLU mas abajo.


BLOCK_SIZE = 4096 #1024 primeras 33K Iter, 2048 hasta iter 49600, luego 4096

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
# train.bin, 39.16B tokens; ya NO coincide con el presupuesto del 1B/20B)

# Tokens / iteraciones. BATCH_SIZE/GRAD_ACCUM_STEPS son los YA PROBADOS en
# la RTX 5090 con esta misma arquitectura (E1536/L18/H24/KV6, block_size
# 512) durante el continual pretraining: ~50.200 tok/s, 23.6/31.8 GB VRAM.

BATCH_SIZE = 2 #10 primeras 33K Iter, 5 hasta iter 49600, luego 2 (BLOCK_SIZE 4096)
GRAD_ACCUM_STEPS = 125 # 100 hasta iter 49600 -- 2*4096*125 = 1,024,000 tok/iter, igual que antes (5*2048*100)

# MAX_ITERS = 1 epoca = una pasada completa sobre train.bin.
#
#   tokens_por_iteracion = BATCH_SIZE * BLOCK_SIZE * GRAD_ACCUM_STEPS
#                         = 20 * 1024 * 50 = 1,024,000 tokens/iter
#   MAX_ITERS = len(train_data) // tokens_por_iteracion
#
# Recalculado tras subir BLOCK_SIZE 512->1024: tokens_por_iteracion se
# DUPLICO (1,024,000 vs 512,000), asi que MAX_ITERS se ha vuelto a fijar
# a la MITAD (156,250 -> 78,125) para mantener el mismo presupuesto total
# de ~80B tokens (y por tanto el mismo tiempo total aprox, +6% por el
# coste extra de atencion a 1024).
# PENDIENTE: sustituir por el numero real en cuanto termine la binarizacion.
MAX_ITERS = 78_125   # <-- AJUSTAR con el total real de train.bin

EVAL_INTERVAL = 100

# Validacion v2: examen FIJO y reproducible.
# EVAL_ITERS se conserva como fallback por compatibilidad con codigo antiguo.
EVAL_ITERS = 100
EVAL_ITERS_TRAIN = 100
EVAL_ITERS_VAL = 1500   # 1000 x 10 x 1024 = 10.24M tokens fijos de validacion
VALIDATION_PROTOCOL = "fixed_val_v1"

# El log nuevo se separa del historico aleatorio para no mezclar protocolos.
LOSS_LOG_PATH = "C:/Users/alegaz/Desktop/PROYECTO SANCHO/Generar_corpus/corpus_80Bt/bin_alfa1/loss_log_fixedval.csv"

GRAD_CLIP = 1.0

# ============================================================
# Learning rate + scheduler (warmup + cosine decay)
# ============================================================
#
# Coseno NORMAL de una sola pasada: decay sobre TODO el presupuesto de
# iteraciones, sin cola plana en MIN_LR y sin el mecanismo de "recorte"
# (ver RECORTE_LR_ACTIVO mas abajo -- IMPORTANTE dejarlo en False, o
# train_pretrain_bpe.py activaria por defecto un recorte con el horizonte
# de una corrida anterior).
# ============================================================

LEARNING_RATE = 3e-4          # LR pico (tras el warmup)
MIN_LR = 3e-5                 # LR minimo al final del cosine (10% del pico)
WARMUP_ITERS = MAX_ITERS // 50   # ~2% de MAX_ITERS, misma proporcion que antes (recalculado: 1,562)
LR_DECAY_ITERS = MAX_ITERS       # coseno normal: decay termina justo al final,
                                  # sin cola de MIN_LR (a diferencia de la
                                  # corrida anterior, que dejaba ~15.6% de cola)
WEIGHT_DECAY = 0.1            # AdamW; estandar a esta escala
BETA1 = 0.9
BETA2 = 0.95                 # 0.95 (no 0.999) es lo habitual en LLMs

# Recorte de coseno: DESACTIVADO para esta corrida. Es un mecanismo aparte
# (para cortar un coseno a mitad de camino si hace falta parar antes de
# tiempo -- ver notas en train_pretrain_bpe.py), no forma parte de un
# coseno normal de una sola pasada.
RECORTE_LR_ACTIVO = False

# ============================================================
# WSD (Warmup-Stable-Decay) -- CAMBIO 2026-08-24
# ============================================================
# Activado tras el segundo rollback del dia (iter 44.900): en vez de
# seguir con el coseno normal (LR bajando de forma continua hasta
# LR_DECAY_ITERS=78.125), se congela el LR en el valor exacto que tenia
# el coseno en el momento de activar esto (iter ~44.910, en marcha), y se
# mantiene CONSTANTE indefinidamente. El annealing final (decay real a
# MIN_LR) se activa mas adelante, a mano, cuando se decida parar el
# pretraining -- sin tener que haber acertado de antemano cuantas
# iteraciones exactas iba a durar.
#
# Motivo: con tantos reinicios/rollbacks este dia, tiene mas sentido no
# depender de un LR_DECAY_ITERS fijo -- un LR constante es ademas
# trivialmente reanudable (es solo un numero, no hay estado de progreso
# de scheduler que pueda desincronizarse en un resume, a diferencia del
# coseno con recorte).
#
# IMPORTANTE: si WSD_ACTIVO=True, tiene prioridad sobre RECORTE_LR_ACTIVO
# (se ignora el recorte mientras WSD este activo).
WSD_ACTIVO = True
WSD_LR = 1.37136228e-04   # LR exacto del coseno en el momento de activar (iter ~44.910) -- NO TOCAR a mano, es el punto de continuidad

# Annealing final del WSD: DESACTIVADO por ahora. Cuando se decida parar
# el pretraining, poner WSD_ANNEALING_ACTIVO=True (y ajustar
# WSD_ANNEALING_ITERS si se quiere una cola mas larga/corta) y relanzar --
# el decay de WSD_LR a MIN_LR arranca en la iteracion en que esto se
# active por primera vez (se guarda en el checkpoint, resume-safe).
WSD_ANNEALING_ACTIVO = False
WSD_ANNEALING_ITERS = 3000   # duracion del decay final una vez activado

# torch.compile: "max-autotune" tarda mas en compilar al arrancar (varios
# minutos) pero suele dar mas rendimiento en entrenamientos largos, donde
# ese coste inicial se amortiza rapido. Alternativas: "default" (compila
# mas rapido, menos ganancia) o "reduce-overhead" (CUDA graphs).
TORCH_COMPILE_MODE = "default"

# ============================================================
# Early stopping
# ============================================================
#
# En esta corrida de ~80B tokens NO usamos early stopping: queremos completar
# la pasada planificada y seleccionar despues entre best_fixedval y final.
# ============================================================

# Para esta corrida se completa la pasada de ~80B tokens. El best.pt se sigue
# actualizando aunque EARLY_STOPPING=False. MIN_DELTA=0.0 hace que cualquier
# mejora real sobre el MISMO examen fijo pueda convertirse en nuevo best.
EARLY_STOPPING = False
PATIENCE = 20
MIN_DELTA = 0.0

# ============================================================
# Checkpoints
# ============================================================

CHECKPOINT_PATH = "./checkpoints/alpha_1_pretrain_05B.pt"

# Se usa un nombre nuevo para no sobrescribir el best antiguo, calculado con
# batches de validacion aleatorios distintos en cada evaluacion.
BEST_CHECKPOINT_PATH = "./checkpoints/alpha_1_pretrain_05B_best_fixedval.pt"

# Guardar checkpoint de reanudacion cada 100 iteraciones (antes: default 500).
CKPT_RESUME_INTERVAL = 100

# ============================================================
# Bateria de evaluacion automatica al guardar checkpoint
# ============================================================
#
# Se lanza justo cuando se guarda best_checkpoint_path (mejora la val loss)
# o resume_checkpoint_path (cada CKPT_RESUME_INTERVAL iters) -- no hay
# snapshots nuevos ni ficheros extra, usa los checkpoints que ya guardabas.
#
# Solo se registran los VALORES (ultima linea de salida de cada comando),
# no el log completo -- una fila por prueba en BATERIA_DOC (CSV).
# ============================================================

BATERIA_ACTIVA = False
BATERIA_DOC = "./resultados_bateria.csv"
BATERIA_TIMEOUT_S = 3600   # corta una prueba colgada en vez de bloquear

# Cada entrada: (nombre, plantilla_de_comando). {checkpoint} se sustituye
# por la ruta del .pt, {iter}/{train_loss}/{val_loss} por los valores en
# el momento del guardado (el .pt no lleva train_loss, por eso se pasa).
# benchmark_bcc_es.py con --iter genera solo el informe compacto
# (test_pretrain_it<iter>.txt, el formato "solo valores" pedido) ademas
# del informe completo de siempre -- no hace falta parsear su salida.
BATERIA_COMANDOS = [
    ("BCC-ES", "python benchmark_bcc_es.py --checkpoints {checkpoint} "
              "--iter {iter} --train-loss {train_loss} --val-loss {val_loss}"),
]

SEED = 1337

# ============================================================
# Generación de prueba
# ============================================================

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.7
TOP_K = 40
