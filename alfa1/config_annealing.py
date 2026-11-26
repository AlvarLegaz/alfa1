import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Alfa1 - MIDTRAIN (fase de annealing/decay final)  |  script propio
# ============================================================
#
# Fase separada de train_pretrain_v2.py, con checkpoints y log de losses
# PROPIOS (no toca nada del pretrain). Arranca con WARM START desde un
# checkpoint de pretrain -- SOLO pesos, optimizer/iteracion/RNG se
# reinician a cero, ver INIT_FROM_CHECKPOINT mas abajo -- y aplica un
# UNICO decay 1-sqrt del LR congelado hasta MIN_LR~0 a lo largo de TODO
# el corpus de midtrain (12BT: conocimiento 6BT / razonamiento 3BT /
# codigo 3BT). No hay warmup: el LR arranca directo en LR_PICO en la
# iteracion 0 de ESTE script (el modelo ya viene "caliente" del
# pretrain).
#
# MAX_ITERS NO se fija a mano: train_midtrain.py lo calcula solo a partir
# de la longitud real de TRAIN_BIN_PATH (una pasada completa) en cuanto
# el .bin este generado -- la duracion de esta fase la determina el
# tamano del corpus nuevo, no un numero de iteraciones elegido de
# antemano. Deja MAX_ITERS=None mas abajo salvo que quieras forzar un
# valor concreto (p.ej. para un smoke test corto).
# ============================================================

# ------------------------------------------------------------
# Warm start (SOLO pesos) desde el pretrain
# ------------------------------------------------------------
# Se usa UNA UNICA VEZ, en el arranque en frio de midtrain (si ya existe
# un checkpoint de REANUDACION propio de midtrain, ese tiene prioridad --
# ver logica en train_midtrain.py, cargar_para_reanudar() vs
# cargar_warm_start()). Optimizer, iteracion y RNG de entrenamiento
# arrancan siempre desde cero con este init, nunca se heredan del
# pretrain.
#
# AJUSTAR: confirma que este es el checkpoint de pretrain que quieres
# usar como punto de partida (por defecto, el mejor checkpoint actual).
INIT_FROM_CHECKPOINT = (
    "C:/Users/alegaz/Desktop/PROYECTO SANCHO/MODELOS/Alfa1_05B_v2/"
    "checkpoints/alpha_1_pretrain_05B_estable_finalizada_58K.pt"
)

# ============================================================
# Tokenizador (CONGELADO, identico al pretrain -- NO tocar)
# ============================================================
TOKENIZER_TYPE = "bpe"
BPE_VOCAB_SIZE = 48000
BPE_MIN_PAIR_FREQ = 5

# ============================================================
# Datos: corpus de midtrain (12BT: conocimiento 6 / razonamiento 3 / codigo 3)
# ============================================================
# PLACEHOLDER -- ajustar en cuanto el pipeline de mezcla genere estos
# ficheros. El script falla al arrancar (con mensaje claro) si no existen
# todavia, igual que hace train_pretrain_v2.py con train.bin/val.bin.
DATA_BIN_DIR = "C:/Users/alegaz/Desktop/PROYECTO SANCHO/Generar_corpus/corpus_80Bt/bin_alfa1"
TRAIN_BIN_PATH = "C:/Users/alegaz/Desktop/PROYECTO SANCHO/Generar_corpus/corpus_80Bt/bin_alfa1/train_annealing.bin"
VAL_BIN_PATH = "C:/Users/alegaz/Desktop/PROYECTO SANCHO/Generar_corpus/corpus_80Bt/bin_alfa1/val_annealing.bin"
# Mismo tokenizer 48k congelado de siempre -- reutiliza el JSON existente.
TOKENIZER_STATE_PATH = (
    "C:/Users/alegaz/Desktop/PROYECTO SANCHO/Generar_corpus/corpus_80Bt/"
    "bin_alfa1/tokenizer_state.json"
)

LOSS_LOG_PATH = "D:/Minaya_Corpus_General_v1/midtrain/loss_log_midtrain.csv"

# ============================================================
# Arquitectura -- identica al pretrain EXCEPTO BLOCK_SIZE/ROPE_THETA
# ============================================================
N_EMBD = 1536
N_HEAD = 24
N_LAYER = 18
N_KV_HEAD = 6
TIE_WEIGHTS = True
FFN_HIDDEN = 4096
RMS_EPS = 1e-5
DROPOUT = 0.0

# De vuelta a 8K: la iteracion 1 lenta a block=4096 era solo la
# compilacion de torch.compile, no un problema real de rendimiento (iter
# 2 ya iba a 25.3s/iter, mas rapido que el pretrain). Con block=8192 no
# se truncan las trazas largas de razonamiento (OpenMathInstruct2 CoT,
# OmniThought, GSM8K-OpenMath-MathReason).
BLOCK_SIZE = 8192

# CAMBIO 2026-08-29: ROPE_THETA revertido de 500000.0 a 10000.0 (el mismo
# que uso el pretrain) tras 4 auditorias RoPE consecutivas empeorando
# (+480.9% -> +483.6% -> +494.7% -> +499.4% de loss al final del contexto
# de 8192 tokens). El escalado NTK a 500000 para extrapolar de 2048/4096 a
# 8192 no estaba funcionando -- se prueba sin reescalar, dejando que el
# modelo interpole con el mismo theta de pretrain en vez de extrapolar.
ROPE_THETA = 10000.0

# ============================================================
# Batch / tokens por iteracion
# ============================================================
# 1 * 8192 * 125 = 1,024,000 tok/iter, LR_PICO comparable sin reescalar.
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 125

EVAL_INTERVAL = 100
EVAL_ITERS_TRAIN = 100
EVAL_ITERS_VAL = 1500
VALIDATION_PROTOCOL = "fixed_val_v1"

GRAD_CLIP = 1.0

# ============================================================
# Learning rate: SOLO decay 1-sqrt, sin warmup
# ============================================================
# LR_PICO es el LR con el que el checkpoint de partida estaba entrenando
# (el WSD_LR congelado del pretrain) -- el decay arranca ahi mismo en la
# iteracion 0 de ESTE script.
LR_PICO = 1.37136228e-04
MIN_LR = 1e-6   # objetivo ~0; valor simbolico solo por estabilidad numerica del optimizer

WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95

# None -> train_midtrain.py lo calcula solo como
# len(train_data) // tokens_por_iteracion (una pasada completa sobre el
# corpus de 12BT). Fijar aqui a mano SOLO para forzar un numero distinto.
MAX_ITERS = None

TORCH_COMPILE_MODE = "default"

# ============================================================
# Early stopping -- DESACTIVADO (igual que el pretrain: se completa la
# pasada planificada sobre el corpus de midtrain)
# ============================================================
EARLY_STOPPING = False
PATIENCE = 20
MIN_DELTA = 0.0

# ============================================================
# Checkpoints propios de midtrain (NO tocan los del pretrain)
# ============================================================
CHECKPOINT_PATH = "./checkpoints_midtrain/alpha_1_annealing_05B.pt"
BEST_CHECKPOINT_PATH = "./checkpoints_midtrain/alpha_1_annealing_05B_best.pt"
CKPT_RESUME_INTERVAL = 100
RESUME = True

# ============================================================
# Bateria de evaluacion automatica (reutiliza benchmark_bcc_es.py)
# ============================================================
BATERIA_ACTIVA = False
BATERIA_DOC = "./resultados_bateria_midtrain.csv"
BATERIA_TIMEOUT_S = 3600
BATERIA_COMANDOS = [
    ("BCC-ES", "python benchmark_bcc_es.py --checkpoints {checkpoint} "
              "--iter {iter} --train-loss {train_loss} --val-loss {val_loss}"),
]

SEED = 1337

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.7
TOP_K = 40
