import os

# Misma fuente de arquitectura que config_sft/config_grpo: el punto de
# partida de DPO es el checkpoint SFT v16 (arquitectura annealed, BLOCK 8192).
from alfa1.config_annealing import (
    N_EMBD, N_HEAD, N_LAYER, N_KV_HEAD, FFN_HIDDEN, TIE_WEIGHTS, ROPE_THETA,
    BLOCK_SIZE, DEVICE,
)

# ============================================================
# Alfa1 - Fase de preferencias (DPO). Ver PLAN_preferencias_alfa1.md
# ============================================================
#
# PASO 1 (2026-09-08): validar que DPO mueve la identidad fabricada (F1) y el
# "no reconoce el error" (F3) SIN romper nada (anclaje anti-olvido). Si sube
# control_identidad a >=38/40 con seguridad 20/20 intacta -> ampliar a F2/F4/F5.

# El release real de v16 es el _best, que se movio/renombro a D:\alpha_1_05B_it_v16.pt
# (todo el scoring v16 se hizo sobre ese). El checkpoints\..._v16.pt es el
# "final" de la misma corrida (2 epocas sin early-stop -> casi identico).
INIT_FROM = r"D:\alpha_1_05B_it_v16.pt"
TOKENIZER_STATE_PATH = r"C:\Users\alegaz\Desktop\PROYECTO SANCHO\Generar_corpus\corpus_80Bt\bin_alfa1\tokenizer_state.json"

DATASET_PATH = r"D:\Generar_SFT\dpo\pares_dpo_v2.jsonl"
HELDOUT_PATH = r"D:\Generar_SFT\dpo\pares_dpo_v2_heldout.jsonl"

USER_TAG = "<USER>"
ASSISTANT_TAG = "<ASSISTANT>"
FIN_TAG = "<FIN>"
PAD_TAG = "<PAD>"

# --- DPO ---
BETA = 0.1                 # temperatura de la preferencia.
# DPO v1 (LR 2e-6, 328 steps) desestabilizo la identidad de v16 (35->30/40,
# "creada por OpenAI"). v2: LR muy suave + termino NLL sobre el chosen (RPO)
# para anclar la politica al texto bueno, no solo a la preferencia relativa.
LEARNING_RATE = 4e-7
NLL_COEF = 0.25          # peso del -logp(chosen) sumado a la loss DPO (RPO/DPO+NLL)
WEIGHT_DECAY = 0.0
BETA1, BETA2 = 0.9, 0.95
GRAD_CLIP = 1.0
WARMUP_STEPS = 15
MIN_LR_FRAC = 0.2         # LR final = LEARNING_RATE * MIN_LR_FRAC (coseno)

BATCH_PAIRS = 8
GRAD_ACCUM = 1
EPOCHS = 2               # v1 con 4 sobre-ajusto; 2 basta y arriesga menos deriva
MAX_LEN = 1024

EVAL_INTERVAL = 25
LOG_INTERVAL = 10
SEED = 1337

CHECKPOINT_DIR = r"C:\Users\alegaz\Desktop\PROYECTO SANCHO\MODELOS\Alfa1_05B_v2\checkpoints_dpo"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "alpha_1_dpo_v2.pt")
BEST_PATH = os.path.join(CHECKPOINT_DIR, "alpha_1_dpo_v2_best.pt")
