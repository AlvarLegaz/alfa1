import os
import torch

# Misma fuente de arquitectura que config_sft.py: el punto de partida de
# GRPO es un checkpoint SFT (arquitectura BLOCK_SIZE=8192, annealed), no el
# de pretraining crudo.
from alfa1.config_annealing import (
    N_EMBD, N_HEAD, N_LAYER, N_KV_HEAD, FFN_HIDDEN, TIE_WEIGHTS, ROPE_THETA,
    BLOCK_SIZE, BPE_VOCAB_SIZE, DEVICE,
)

# ============================================================
# Sancho/Alfa1 - Fase GRPO (RL con recompensa verificable)
# ============================================================
#
# EXPERIMENTO 1 (2026-09-05): primer intento de GRPO tras 9 fases de SFT
# estancadas en varias categorias (email, procedimiento, agente_tool_use,
# json, formato/restricciones). Arranca desde el checkpoint SFT del
# Exp.10 (fix de indentacion en train_sft_bpe.py), NO desde el Exp.9 --
# comparar "antes/despues de GRPO" contra el Exp.9 confundiria el efecto
# del fix de indentacion con el del GRPO.
#
# Auditoria previa (ver D:/Generar_SFT/grpo/README.md) confirmo que:
#   - los verificadores literales anulaban el 70% de los aciertos reales
#     (corregido con extraccion de respuesta + credito parcial)
#   - el dataset v1 (math/logic/code) no cubria los agujeros reales del
#     SFT -> se anadieron 100 items de categoria "agente" (tool_call,
#     json_enum, restricciones, estructura_texto)
#   - "code" media 0.000 de recompensa / 100% de prompts sin gradiente
#     sobre el checkpoint v9 (el bug de indentacion le impedia generar
#     Python valido) -- se relanza con code incluido para comprobar si,
#     arreglado el bug de origen, ahora si aporta senal.

# --- Punto de partida: el mejor checkpoint del Exp.10 SFT (fix de indentacion) ---
INIT_FROM = r"C:\Users\alegaz\Desktop\PROYECTO SANCHO\MODELOS\Alfa1_05B_v2\checkpoints\alpha_1_sft_alpaca_plus_v10_best.pt"
TOKENIZER_STATE_PATH = r"C:\Users\alegaz\Desktop\PROYECTO SANCHO\Generar_corpus\corpus_80Bt\bin_alfa1\tokenizer_state.json"

# --- Dataset GRPO ---
# v2 = v1 (1000, math/logic/code) + 100 de categoria "agente". Se incluye
# "code" completo (300 items) para comprobar si el fix de indentacion basta.
DATASET_PATH = r"D:\Generar_SFT\grpo\dataset_grpo_v2.json"

USER_TAG = "<USER>"
ASSISTANT_TAG = "<ASSISTANT>"
FIN_TAG = "<FIN>"
PAD_TAG = "<PAD>"

# --- Rollouts (muestreo de completions por prompt) ---
# GROUP_SIZE alto ayuda a estimar mejor la media/varianza del grupo (menos
# ruido en la ventaja), pero cuesta VRAM y tiempo linealmente. 8 es el
# minimo razonable para que la desviacion tipica del grupo no sea puro ruido.
GROUP_SIZE = 8
# Prompts distintos por actualizacion de gradiente (grad accum implicito,
# igual que GRAD_ACCUM_STEPS en train_sft_bpe.py: cada prompt se procesa y
# se hace backward por separado, un solo optimizer.step() al final).
PROMPTS_PER_STEP = 4

MAX_NEW_TOKENS = 220          # cubre respuestas cortas y explicaciones/codigo
TEMPERATURE = 0.9             # mas alto que en benchmark (0.30): GRPO necesita
                              # diversidad real dentro del grupo, si no toda
                              # la ventaja es 0 (ver auditoria de baseline)
TOP_K = 50

# --- Optimizacion ---
# LR muy por debajo del SFT (2e-5): en RL sobre un modelo ya afinado, un LR
# alto destruye el checkpoint de partida en pocos pasos (la señal de
# recompensa es mucho mas ruidosa que la de cross-entropy supervisado).
LEARNING_RATE = 2e-6
WEIGHT_DECAY = 0.0
BETA1 = 0.9
BETA2 = 0.95
GRAD_CLIP = 1.0

# Coeficiente de la penalizacion KL contra el modelo de referencia (copia
# congelada del checkpoint de partida). Sin este termino el modelo puede
# alejarse arbitrariamente del SFT persiguiendo recompensa y degradar todo
# lo que el verificador no mide (fluidez, idioma, formato no verificado).
KL_COEF = 0.02

MAX_ITERS = 1500
SAVE_INTERVAL = 50            # checkpoint numerado cada N iteraciones
LOG_INTERVAL = 5              # imprime recompensa media cada N iteraciones
SEED = 1337

CHECKPOINT_DIR = r"C:\Users\alegaz\Desktop\PROYECTO SANCHO\MODELOS\Alfa1_05B_v2\checkpoints_grpo"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "alpha_1_grpo_exp1.pt")

# Reanudacion: la corrida original murio en un CUDA OOM en iter ~851 (causado
# por un smoke-test de vLLM lanzado en la misma GPU, 2026-09-06 -- error
# ajeno al propio GRPO). El ultimo checkpoint bueno es iter850. RESUME_FROM
# carga politica+optimizador desde ahi y continua hasta MAX_ITERS; el
# modelo de referencia (KL) sigue cargando de INIT_FROM como siempre, no de
# aqui. Poner a None para una corrida nueva desde cero.
RESUME_FROM = os.path.join(CHECKPOINT_DIR, "alpha_1_grpo_exp1_iter850.pt")
