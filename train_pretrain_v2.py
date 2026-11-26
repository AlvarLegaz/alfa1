# train_pretrain_bpe.py
"""
Fase 2 del pipeline de pretraining.

REQUISITO: ejecutar antes 'python -m sancho.preparar_datos', que genera
train.bin, val.bin y tokenizer_state.json.

Este script ya NO carga ni tokeniza texto: lee los tokens desde los .bin
con np.memmap (RAM plana) y entrena el modelo. Asi soporta corpus de
400 MB+ sin saturar la RAM.

--------------------------------------------------------------------
CAMBIO 13/07/2026: RECORTE DE COSENO EN EL SCHEDULER DE LR
--------------------------------------------------------------------
Activado en iter 27500, tras 3 subidas consecutivas de val loss
(26000->27500: +0.0023, +0.0025, +0.0037, magnitud creciente) -- la
primera racha de 3 subidas seguidas de toda la serie de Alpha 1 (54
puntos revisados, maximo previo: 2 subidas seguidas).

El criterio formal acordado (4a subida consecutiva o confirmacion por
caida de BCC-ES en el informe de iter ~29000-30000) NO se habia
cumplido todavia en el momento de activar; se activa por decision del
usuario, no por cumplimiento estricto del criterio.

Nuevo horizonte de decay: iter 50000 (antes: LR_DECAY_ITERS=63070).
La transicion es continua (sin salto de LR): se parte del LR que daria
el coseno ORIGINAL en la iteracion exacta de reanudacion, no del
peak_lr. Verificado: continuidad exacta (diferencia 0.0) en iter 27500.

Para desactivar el recorte y volver al coseno original, poner
RECORTE_LR_ACTIVO = False mas abajo (o en config_pretrain.py).
--------------------------------------------------------------------
"""

import os
import re
import math
import random
import time
import csv
from tqdm import tqdm

# Referencia global a la barra de progreso activa (se asigna mas abajo,
# justo antes del bucle principal). _log() la usa para no corromper la
# barra: mientras esta activa, tqdm.write() imprime la linea POR ENCIMA
# del hueco reservado a la barra y la vuelve a redibujar debajo, en vez
# de que un print() normal la parta en dos o la deje duplicada en
# pantalla. Antes de que exista (arranque del script) o si tqdm no esta
# disponible por cualquier motivo, cae a print() normal sin romper nada.
_barra = None


def _log(*args, sep=" ", end="\n"):
    msg = sep.join(str(a) for a in args)
    if _barra is not None:
        try:
            _barra.write(msg)
            return
        except Exception:
            pass
    print(msg, end=end)

import numpy as np
import torch

from alfa1.config_pretrain import *
from alfa1.model import SanchoGPT
from alfa1.tokenizer_bpe import BPETokenizer


# ============================================================
# Ajustes por defecto si no existen en config_pretrain.py
# ============================================================

TRAIN_RATIO = globals().get("TRAIN_RATIO", 0.90)
MAX_CHARS_BLOQUE = globals().get("MAX_CHARS_BLOQUE", 2000)

EARLY_STOPPING = globals().get("EARLY_STOPPING", True)
PATIENCE = globals().get("PATIENCE", 6)
MIN_DELTA = globals().get("MIN_DELTA", 0.001)

GRAD_CLIP = globals().get("GRAD_CLIP", 1.0)

SEED = globals().get("SEED", 1337)

# Validacion fija/reproducible. Los seeds son independientes para que evaluar
# nunca consuma ni altere la secuencia de batches usada para entrenar.
EVAL_ITERS_TRAIN = globals().get("EVAL_ITERS_TRAIN", globals().get("EVAL_ITERS", 100))
EVAL_ITERS_VAL = globals().get("EVAL_ITERS_VAL", globals().get("EVAL_ITERS", 100))
VALIDATION_PROTOCOL = globals().get("VALIDATION_PROTOCOL", "fixed_val_v1")
TRAIN_RNG_SEED = globals().get("TRAIN_RNG_SEED", SEED + 100)
TRAIN_EVAL_SEED = globals().get("TRAIN_EVAL_SEED", SEED + 200)
VALIDATION_SEED = globals().get("VALIDATION_SEED", SEED + 300)

GRAD_ACCUM_STEPS = globals().get("GRAD_ACCUM_STEPS", 1)

# --- Reanudacion automatica tras caida ---
# Cada CKPT_RESUME_INTERVAL iteraciones se guarda un checkpoint COMPLETO
# (pesos + optimizer + scaler + iteracion + best_val_loss) que permite
# continuar exactamente donde se corto. Distinto del best: este se sobrescribe
# siempre, sea cual sea la val loss, porque su proposito es reanudar, no
# guardar el mejor.
RESUME = globals().get("RESUME", True)
CKPT_RESUME_INTERVAL = globals().get("CKPT_RESUME_INTERVAL", 500)

# --- Scheduler + optimizer (defaults si el config no los define) ---
MIN_LR = globals().get("MIN_LR", LEARNING_RATE / 10)
WARMUP_ITERS = globals().get("WARMUP_ITERS", max(1, MAX_ITERS // 50))
LR_DECAY_ITERS = globals().get("LR_DECAY_ITERS", MAX_ITERS)
WEIGHT_DECAY = globals().get("WEIGHT_DECAY", 0.1)
BETA1 = globals().get("BETA1", 0.9)
BETA2 = globals().get("BETA2", 0.95)

# --- Recorte de coseno (LR) ---
# Ver cabecera del fichero para el porque y la fecha de activacion.
RECORTE_LR_ACTIVO = globals().get("RECORTE_LR_ACTIVO", True)
NUEVO_LR_DECAY_ITERS = globals().get("NUEVO_LR_DECAY_ITERS", 50000)

_dir_ckpt = os.path.dirname(CHECKPOINT_PATH) or "."

# Rutas de los binarios y estado del tokenizador (deben coincidir con
# las de preparar_datos.py).
_dir_datos = globals().get("DATA_BIN_DIR",
                           os.path.dirname(CHECKPOINT_PATH) or ".")
TRAIN_BIN_PATH = globals().get("TRAIN_BIN_PATH",
                               os.path.join(_dir_datos, "train.bin"))
VAL_BIN_PATH = globals().get("VAL_BIN_PATH",
                             os.path.join(_dir_datos, "val.bin"))
TOKENIZER_STATE_PATH = globals().get("TOKENIZER_STATE_PATH",
                                     os.path.join(_dir_datos, "tokenizer_state.json"))
# CSV con train/val loss en cada iteracion evaluada (EVAL_INTERVAL). Se
# abre en modo append: si el fichero ya existe (p.ej. tras un RESUME),
# NO se sobrescribe, se sigue anadiendo -- asi el historial completo
# sobrevive a caidas y reanudaciones.
LOSS_LOG_PATH = globals().get("LOSS_LOG_PATH",
                              os.path.join(_dir_datos, "loss_log.csv"))

# Debe coincidir con el dtype usado en preparar_datos.py
DTYPE_TOKENS = np.uint16


# ============================================================
# Ajustes de rendimiento
# ============================================================

torch.set_float32_matmul_precision("high")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)  # compatibilidad con cualquier uso NumPy legado

# RNG exclusivo del entrenamiento. Su estado se guarda/restaura en resume.pt.
rng_train = np.random.default_rng(TRAIN_RNG_SEED)

USE_AMP = DEVICE == "cuda"
AMP_DEVICE_TYPE = "cuda" if DEVICE == "cuda" else "cpu"

# bf16 en GPUs modernas (H100/A100): mas estable que fp16 y NO necesita
# GradScaler. Si la GPU no soporta bf16, caemos a fp16 con scaler.
if DEVICE == "cuda" and torch.cuda.is_bf16_supported():
    AMP_DTYPE = torch.bfloat16
    USE_SCALER = False
elif DEVICE == "cuda":
    AMP_DTYPE = torch.float16
    USE_SCALER = True
else:
    AMP_DTYPE = torch.float32
    USE_SCALER = False


# ============================================================
# Comprobaciones previas
# ============================================================

_TRAIN_BIN_PATH_MESETA_CHECK = globals().get("TRAIN_BIN_PATH_MESETA", TRAIN_BIN_PATH)
for ruta in (_TRAIN_BIN_PATH_MESETA_CHECK, VAL_BIN_PATH, TOKENIZER_STATE_PATH):
    if not os.path.isfile(ruta):
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta primero:\n"
            f"    python -m sancho.preparar_datos"
        )


# ============================================================
# Cargar tokenizador (solo para get_state en el checkpoint y vocab real)
# ============================================================

import json

tokenizer = BPETokenizer()
with open(TOKENIZER_STATE_PATH, "r", encoding="utf-8") as f:
    tokenizer.load_state(json.load(f))

vocab_size = tokenizer.vocab_size()

# ID del separador canonico entre documentos. Se usa en get_batch() para
# no entrenar la transicion artificial <|endofstory|> -> inicio del siguiente
# documento, manteniendo intacto el aprendizaje de ultimo_token -> endofstory.
STORY_ID = tokenizer.story_id
if STORY_ID is None:
    raise RuntimeError("No se encontro <|endofstory|> en el tokenizador.")


# ============================================================
# Datos: memmap a los .bin (RAM plana)
# ============================================================
#
# CAMBIO 2026-08-23 (train_pretrain_v2.py, cambio automatico meseta->final):
# v2 arranca leyendo TRAIN_BIN_PATH_MESETA (no TRAIN_BIN_PATH, que sigue
# siendo el train.bin original para no tocar v1). Si TRAIN_BIN_PATH_MESETA
# no esta definido en config_pretrain.py (config antiguo, o alguien lanza
# v2 sin haberlo anadido), cae a TRAIN_BIN_PATH para no romper -- pero en
# ese caso NO habra swap automatico a train_final.bin (ver mas abajo).
TRAIN_BIN_PATH_MESETA = globals().get("TRAIN_BIN_PATH_MESETA", TRAIN_BIN_PATH)
TRAIN_BIN_PATH_FINAL = globals().get("TRAIN_BIN_PATH_FINAL", None)
ITER_CAMBIO_A_FINAL = globals().get("ITER_CAMBIO_A_FINAL", None)

# Fichero activo ahora mismo. Persistido en el checkpoint (ver guardar_modelo/
# cargar_para_reanudar mas abajo) para que reanudar sepa cual tocaba sin
# tener que re-derivarlo de la iteracion (mas robusto si ITER_CAMBIO_A_FINAL
# cambiara de valor entre una ejecucion y otra).
_corpus_activo = "meseta"


def _cargar_train_data(ruta):
    """(Re)abre train_data desde `ruta` y recalcula TODO lo que depende de
    su longitud: FIXED_TRAIN_EVAL_POSITIONS (posiciones fijas de examen de
    train loss) y el muestreador secuencial. Se usa tanto en el arranque
    como en el swap meseta->final en caliente -- extraido a funcion para
    no duplicar esta logica en los dos sitios."""
    global train_data, FIXED_TRAIN_EVAL_POSITIONS, muestreador_train
    train_data = np.memmap(ruta, dtype=DTYPE_TOKENS, mode="r")

    if len(train_data) <= BLOCK_SIZE:
        raise ValueError(
            f"Train data demasiado pequeno para BLOCK_SIZE={BLOCK_SIZE}. "
            f"Ruta: {ruta}  Train tokens: {len(train_data)}"
        )

    # Mismo seed que en el arranque -- las posiciones concretas cambian
    # porque el fichero es distinto (no tiene sentido "continuar" offsets
    # de un fichero a otro), pero el criterio de generacion es identico,
    # asi que el examen sigue siendo igual de reproducible dentro de cada
    # fichero.
    _rng_train_eval = np.random.default_rng(TRAIN_EVAL_SEED)
    FIXED_TRAIN_EVAL_POSITIONS = _rng_train_eval.integers(
        0, len(train_data) - BLOCK_SIZE,
        size=(EVAL_ITERS_TRAIN, BATCH_SIZE), dtype=np.int64,
    )

    muestreador_train = MuestreadorSecuencial(
        len(train_data), BLOCK_SIZE, BATCH_SIZE, TRAIN_RNG_SEED
    )
    return train_data


val_data = np.memmap(VAL_BIN_PATH, dtype=DTYPE_TOKENS, mode="r")

if len(val_data) <= BLOCK_SIZE:
    raise ValueError(
        f"Val data demasiado pequeno para BLOCK_SIZE={BLOCK_SIZE}. "
        f"Val tokens: {len(val_data)}"
    )

tokens_por_microbatch = BATCH_SIZE * BLOCK_SIZE
tokens_por_iteracion = BATCH_SIZE * BLOCK_SIZE * GRAD_ACCUM_STEPS


# ============================================================
# Batches + validacion fija/reproducible
# ============================================================

# val_data NO cambia nunca (viene del corpus_80Bt original, no de la
# reconstruccion meseta/final) -- val_loss sigue siendo comparable entre
# TODA la vida del entrenamiento, swap incluido.
_rng_val_setup = np.random.default_rng(VALIDATION_SEED)
FIXED_VAL_EVAL_POSITIONS = _rng_val_setup.integers(
    0, len(val_data) - BLOCK_SIZE,
    size=(EVAL_ITERS_VAL, BATCH_SIZE), dtype=np.int64,
)

_SEQ_OFFSETS = np.arange(BLOCK_SIZE + 1, dtype=np.int64)[None, :]

class MuestreadorSecuencial:
    """Recorre train_data en bloques NO solapados de BLOCK_SIZE tokens, en
    un orden barajado (permutacion determinista a partir de una semilla),
    con posicion PERSISTENTE (se guarda/restaura en el checkpoint).

    CAMBIO 2026-08-23: sustituye al muestreo aleatorio CON REEMPLAZO
    (rng.integers(0, len(source)-BLOCK_SIZE, ...)) que habia antes. Aquel
    esquema, con ~39M ventanas de 2048 tokens muestreadas a lo largo de
    todo el entrenamiento sobre un corpus de ~79.5B tokens, da cobertura
    tipo Poisson (lambda~1.0): ~36.5% de los tokens del corpus NUNCA se
    llegan a ver, y otro ~26.7% se ve 2 o mas veces -- pese a que el
    computo total de tokens "vistos" cuadre con ~1 pasada. Este muestreador
    en cambio genera una permutacion de los ~N/BLOCK_SIZE bloques
    disponibles y los consume en ese orden sin reemplazo: cobertura EXACTA
    de todo el corpus en una epoca, sin huecos ni repeticion. Al agotar la
    permutacion (fin de epoca) se genera una nueva con una semilla
    derivada y se continua sin cortes.

    Solo se aplica al MUESTREO DE ENTRENAMIENTO -- los examenes fijos
    (FIXED_TRAIN_EVAL_POSITIONS / FIXED_VAL_EVAL_POSITIONS) no cambian,
    siguen siendo un conjunto pequeno y fijo, donde el muestreo aleatorio
    con reemplazo no supone ningun problema de cobertura.
    """

    def __init__(self, n_tokens, block_size, batch_size, seed_base, epoca=0, posicion=0):
        self.block_size = block_size
        self.batch_size = batch_size
        self.seed_base = seed_base
        self.n_bloques = (n_tokens - 1) // block_size
        self.epoca = epoca
        self.posicion = posicion
        self._generar_permutacion()

    def _generar_permutacion(self):
        rng = np.random.default_rng(self.seed_base + self.epoca)
        self.permutacion = rng.permutation(self.n_bloques)

    def siguiente_offsets(self):
        if self.posicion + self.batch_size > self.n_bloques:
            # Fin de epoca: nueva permutacion (semilla derivada) y se
            # continua desde el principio, sin parar el entrenamiento.
            self.epoca += 1
            self.posicion = 0
            self._generar_permutacion()
        idxs_bloque = self.permutacion[self.posicion:self.posicion + self.batch_size]
        self.posicion += self.batch_size
        return (idxs_bloque * self.block_size).astype(np.int64)

    def get_state(self):
        # Se guarda n_bloques (la magnitud que realmente define la
        # permutacion). "n_tokens" se mantiene por compatibilidad con
        # checkpoints y lectores antiguos, pero YA NO se usa para decidir
        # si el fichero cambio -- ver from_state.
        return {"seed_base": self.seed_base, "epoca": self.epoca,
                "posicion": self.posicion, "n_bloques": self.n_bloques,
                "n_tokens": self.n_bloques * self.block_size}

    @classmethod
    def from_state(cls, n_tokens, block_size, batch_size, state):
        # Blindaje: si train.bin ha cambiado de tamano desde que se guardo
        # este estado (p.ej. se sustituyo por una version rebalanceada), la
        # posicion guardada ya no significa nada sobre el fichero nuevo --
        # continuar con ella saltaria ciegamente el principio del fichero
        # nuevo en vez de empezar limpio. Se detecta y se arranca una
        # epoca nueva (posicion 0) automaticamente.
        #
        # FIX 2026-08-24 (bug real, causo divergencia de train/val loss entre
        # iter 45.400 y 46.600): la comparacion se hacia entre DOS MAGNITUDES
        # DISTINTAS -- get_state guardaba `n_bloques * block_size` (27.132.309.504)
        # y aqui se comparaba contra `len(train_data)` (27.132.311.136). Como
        # n_bloques = (n_tokens-1)//block_size, el valor guardado es SIEMPRE
        # menor que el real (por el resto que no llena un bloque: 1.632 tokens
        # en este corpus), asi que NUNCA podian coincidir y el blindaje saltaba
        # en CADA reanudacion, reseteando posicion=0. Con la misma seed_base y
        # epoca, _generar_permutacion() regenera la permutacion IDENTICA, de
        # modo que tras cada reinicio el modelo volvia a entrenar sobre
        # exactamente los mismos bloques: ~1.550 iteraciones encerradas en el
        # 2,6% del corpus (350.000 de 13.248.198 bloques), vistos 2-3 veces.
        # Resultado: sobreajuste a esa franja y subida sostenida de train loss
        # (medida sobre el examen fijo, que si cubre todo el corpus) y de val.
        # Ahora se compara n_bloques con n_bloques, que es la magnitud que
        # define la permutacion.
        n_bloques_actual = (n_tokens - 1) // block_size
        n_bloques_guardado = state.get("n_bloques")
        if n_bloques_guardado is None:
            # Checkpoint anterior al fix: guardaba n_bloques*block_size bajo
            # la clave "n_tokens". Se recupera n_bloques dividiendo (division
            # exacta por construccion), asi los checkpoints ya escritos se
            # reanudan bien sin necesidad de regenerarlos.
            _n_tokens_guardado = state.get("n_tokens")
            n_bloques_guardado = (_n_tokens_guardado // block_size
                                  if _n_tokens_guardado is not None else None)
        if n_bloques_guardado is not None and n_bloques_guardado != n_bloques_actual:
            _log(
                f"[MUESTREADOR] train.bin ha cambiado de tamano "
                f"({n_bloques_guardado:,} -> {n_bloques_actual:,} bloques) desde el "
                f"ultimo checkpoint -- empezando epoca nueva sobre el "
                f"fichero actual en vez de continuar la posicion guardada."
            )
            return cls(n_tokens, block_size, batch_size, state["seed_base"])
        return cls(n_tokens, block_size, batch_size, state["seed_base"],
                    epoca=state["epoca"], posicion=state["posicion"])


# Carga inicial: SIEMPRE meseta al arrancar de cero. Si se reanuda mas
# alla de ITER_CAMBIO_A_FINAL, cargar_para_reanudar() hace el swap a
# final justo despues (ver mas abajo) -- cargar meseta aqui primero es
# gasto minimo (abrir un memmap no lee el fichero entero) y mantiene el
# arranque simple en todos los casos.
# CAMBIO 2026-08-23 (fix real, crash en el primer lanzamiento de v2):
# esta llamada tiene que ir DESPUES de que MuestreadorSecuencial este
# definida -- estaba antes (justo tras crear val_data), lo que provocaba
# NameError: name 'MuestreadorSecuencial' is not defined en el arranque,
# porque en Python el CUERPO de una funcion puede referenciar un nombre
# que se defina mas tarde (resolucion tardia), pero una llamada a nivel
# de modulo se ejecuta en el momento en que aparece en el fichero, y en
# ese punto la clase aun no existia.
_cargar_train_data(TRAIN_BIN_PATH_MESETA)


def get_batch(split, *, rng=None, positions=None, muestreador=None):
    source = train_data if split == "train" else val_data

    if positions is not None:
        ix = np.asarray(positions, dtype=np.int64)
        if ix.shape != (BATCH_SIZE,):
            raise ValueError(
                f"positions debe tener shape ({BATCH_SIZE},), recibido {ix.shape}"
            )
    elif muestreador is not None:
        ix = muestreador.siguiente_offsets()
    else:
        # El entrenamiento usa exclusivamente rng_train. La validacion nunca
        # llega aqui sin positions: siempre usa el examen fijo.
        if rng is None:
            if split == "train":
                rng = rng_train
            else:
                raise ValueError("La validacion debe usar positions fijas.")
        ix = rng.integers(
            0, len(source) - BLOCK_SIZE, size=BATCH_SIZE, dtype=np.int64
        )

    idx_matrix = ix[:, None] + _SEQ_OFFSETS
    batch = source[idx_matrix].astype(np.int64)  # (BATCH_SIZE, BLOCK_SIZE+1)

    x = torch.from_numpy(batch[:, :-1])
    y = torch.from_numpy(batch[:, 1:]).clone()

    # No aprender la transicion entre documentos:
    #   <|endofstory|> -> primer token del documento siguiente
    # Si x[t] es STORY_ID, y[t] es justamente ese primer token siguiente.
    # -100 ya es ignorado por F.cross_entropy(ignore_index=-100) en model.py.
    y[x == STORY_ID] = -100

    if DEVICE == "cuda":
        return (
            x.pin_memory().to(DEVICE, non_blocking=True),
            y.pin_memory().to(DEVICE, non_blocking=True),
        )
    return x.to(DEVICE), y.to(DEVICE)


def _resumen_losses(losses):
    mean = losses.mean().item()
    if len(losses) > 1:
        std = losses.std(unbiased=True).item()
        sem = std / math.sqrt(len(losses))
    else:
        std = 0.0
        sem = 0.0
    return mean, std, sem


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()

    train_losses = torch.empty(EVAL_ITERS_TRAIN, dtype=torch.float32)
    for k in range(EVAL_ITERS_TRAIN):
        X, Y = get_batch("train", positions=FIXED_TRAIN_EVAL_POSITIONS[k])
        with torch.amp.autocast(
            device_type=AMP_DEVICE_TYPE, dtype=AMP_DTYPE, enabled=USE_AMP
        ):
            _, loss, _ = model(X, Y)
        train_losses[k] = loss.item()

    val_losses = torch.empty(EVAL_ITERS_VAL, dtype=torch.float32)
    for k in range(EVAL_ITERS_VAL):
        X, Y = get_batch("val", positions=FIXED_VAL_EVAL_POSITIONS[k])
        with torch.amp.autocast(
            device_type=AMP_DEVICE_TYPE, dtype=AMP_DTYPE, enabled=USE_AMP
        ):
            _, loss, _ = model(X, Y)
        val_losses[k] = loss.item()

    out["train"], out["train_std"], out["train_sem"] = _resumen_losses(train_losses)
    out["val"], out["val_std"], out["val_sem"] = _resumen_losses(val_losses)

    model.train()
    return out


# ============================================================
# Modelo
# ============================================================

model = SanchoGPT(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER,
    dropout=DROPOUT,
    device=DEVICE,
    # --- v2 (arquitectura tipo Llama/Qwen) ---
    n_kv_head=globals().get("N_KV_HEAD", None),      # GQA; None -> MHA plano
    tie_weights=globals().get("TIE_WEIGHTS", True),
    rope_theta=globals().get("ROPE_THETA", 10000.0),
    ffn_hidden=globals().get("FFN_HIDDEN", None),    # None -> 8/3*n_embd auto
).to(DEVICE)

num_params = sum(p.numel() for p in model.parameters())

# FLOPs por token (aprox. PaLM/Chinchilla, YA incluye forward+backward):
#   6*N_sin_embeddings  -> proyecciones lineales (Q/K/V/O + SwiGLU)
#   12*n_layer*n_embd*T -> atencion (QK^T + attn@V), escala con block_size
# La matriz de embeddings se excluye del termino "6*N" por convencion (no es
# un matmul denso comparable); con tie_weights=True es la misma matriz que
# lm_head, asi que solo se resta una vez.
num_params_no_emb = num_params - model.token_embedding_table.weight.numel()
FLOPS_POR_TOKEN = 6 * num_params_no_emb + 12 * N_LAYER * N_EMBD * BLOCK_SIZE

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(BETA1, BETA2),
    weight_decay=WEIGHT_DECAY,
    fused=(DEVICE == "cuda"),
)

# raw_model = referencia SIN compilar. torch.compile envuelve el modulo y
# antepone "_orig_mod." a las claves del state_dict; si guardaramos/
# cargaramos checkpoints o exportaramos a HF desde el modelo compilado,
# romperiamos el formato de checkpoint existente y el export_to_hf (que
# depende de nombres exactos tipo "blocks.N.self_attn.q_proj.weight").
# Por eso: raw_model para guardar/cargar/exportar, model (compilado) SOLO
# para el forward en el bucle de entrenamiento.
raw_model = model

# El modo de torch.compile se lee de config_pretrain.py (TORCH_COMPILE_MODE),
# no va fijo aqui -- "max-autotune" tarda mas en compilar al arrancar pero
# suele dar mas rendimiento; "default" compila mas rapido, "reduce-overhead"
# usa CUDA graphs (util con microbatches pequenos y muchos).
TORCH_COMPILE_MODE = globals().get("TORCH_COMPILE_MODE", "max-autotune")

if DEVICE == "cuda":
    model = torch.compile(model, mode=TORCH_COMPILE_MODE)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_SCALER,
)


# ============================================================
# Scheduler de LR: warmup lineal + cosine decay, con recorte opcional
# ============================================================

# Estado interno del recorte: se rellena solo, en la primera llamada
# tras activar RECORTE_LR_ACTIVO. No tocar a mano.
_iter_activacion_recorte = None
_lr_en_activacion = None

# --- WSD (Warmup-Stable-Decay) -- ver config_pretrain.py ---
WSD_ACTIVO = globals().get("WSD_ACTIVO", False)
WSD_LR = globals().get("WSD_LR", None)
WSD_ANNEALING_ACTIVO = globals().get("WSD_ANNEALING_ACTIVO", False)
WSD_ANNEALING_ITERS = globals().get("WSD_ANNEALING_ITERS", 3000)
if WSD_ACTIVO and WSD_LR is None:
    raise ValueError(
        "WSD_ACTIVO=True pero WSD_LR no esta definido en config_pretrain.py -- "
        "sin un LR congelado numerico, lr_para_iteracion() devolveria None y "
        "rompería el optimizer de forma confusa. Define WSD_LR (float) o pon "
        "WSD_ACTIVO=False."
    )

# Iteracion en la que se activo el annealing final por primera vez, y LR
# de partida en ese momento (=WSD_LR normalmente). Se restauran desde el
# checkpoint en cargar_para_reanudar() si ya se habia activado en una
# corrida anterior -- a diferencia de _iter_activacion_recorte (que es
# puramente un global de proceso, se pierde en cada resume), este SI se
# persiste, precisamente para que activar el annealing sea seguro aunque
# el proceso se reinicie a mitad.
_iter_activacion_wsd_annealing = None
_lr_en_activacion_wsd_annealing = None


def _lr_coseno_original(it):
    """Coseno original, sin recorte (comportamiento previo del script,
    intacto). Se usa tanto si RECORTE_LR_ACTIVO=False como para calcular
    el punto de partida continuo cuando se activa el recorte."""
    if it < WARMUP_ITERS:
        return LEARNING_RATE * (it + 1) / WARMUP_ITERS
    if it >= LR_DECAY_ITERS:
        return MIN_LR
    ratio = (it - WARMUP_ITERS) / max(1, (LR_DECAY_ITERS - WARMUP_ITERS))
    coef = 0.5 * (1.0 + math.cos(math.pi * ratio))  # 1 -> 0
    return MIN_LR + coef * (LEARNING_RATE - MIN_LR)


def lr_para_iteracion(it):
    """LR con warmup lineal + cosine decay hasta MIN_LR.

    - it < WARMUP_ITERS:           sube lineal de 0 a LEARNING_RATE
    - WARMUP_ITERS <= it < DECAY:  cosine de LEARNING_RATE a MIN_LR
    - it >= LR_DECAY_ITERS:        se queda en MIN_LR

    Si RECORTE_LR_ACTIVO es False: comportamiento identico al original.

    Si RECORTE_LR_ACTIVO es True: en la PRIMERA llamada (justo al
    reanudar en iteracion_inicial) se captura el LR que habria dado el
    coseno ORIGINAL en ese punto exacto (continuidad garantizada, sin
    salto) y a partir de ahi se sigue un coseno nuevo que llega a
    MIN_LR en NUEVO_LR_DECAY_ITERS en vez de en LR_DECAY_ITERS.
    """
    global _iter_activacion_recorte, _lr_en_activacion
    global _iter_activacion_wsd_annealing, _lr_en_activacion_wsd_annealing

    if it < WARMUP_ITERS:
        return LEARNING_RATE * (it + 1) / WARMUP_ITERS

    # WSD tiene prioridad sobre el recorte de coseno -- ver config_pretrain.py.
    if WSD_ACTIVO:
        if not WSD_ANNEALING_ACTIVO:
            # Fase "stable": LR constante, el numero exacto congelado en
            # config_pretrain.py. Sin estado que trackear -- resume-safe
            # por construccion.
            return WSD_LR

        # Fase "decay": arranca en la iteracion en que WSD_ANNEALING_ACTIVO
        # se ve por primera vez True (en esta corrida, o restaurada de un
        # checkpoint si ya se habia activado antes).
        if _iter_activacion_wsd_annealing is None:
            _iter_activacion_wsd_annealing = it
            _lr_en_activacion_wsd_annealing = WSD_LR
            _log(f"[WSD-ANNEALING] Activado en iter {it} | "
                  f"LR de partida: {_lr_en_activacion_wsd_annealing:.6e} | "
                  f"decay a MIN_LR en {WSD_ANNEALING_ITERS} iteraciones "
                  f"(termina en iter {it + WSD_ANNEALING_ITERS})")

        it_fin = _iter_activacion_wsd_annealing + WSD_ANNEALING_ITERS
        if it >= it_fin:
            return MIN_LR
        ratio = (it - _iter_activacion_wsd_annealing) / max(1, WSD_ANNEALING_ITERS)
        coef = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return MIN_LR + coef * (_lr_en_activacion_wsd_annealing - MIN_LR)

    if not RECORTE_LR_ACTIVO:
        return _lr_coseno_original(it)

    if _iter_activacion_recorte is None:
        _iter_activacion_recorte = it
        _lr_en_activacion = _lr_coseno_original(it)
        _log(f"[LR-RECORTE] Activado en iter {it} | "
              f"LR de partida (continuo): {_lr_en_activacion:.6e} | "
              f"nuevo horizonte de decay: iter {NUEVO_LR_DECAY_ITERS} "
              f"(antes: iter {LR_DECAY_ITERS})")

    if it >= NUEVO_LR_DECAY_ITERS:
        return MIN_LR

    ratio = (it - _iter_activacion_recorte) / max(
        1, (NUEVO_LR_DECAY_ITERS - _iter_activacion_recorte)
    )
    coef = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return MIN_LR + coef * (_lr_en_activacion - MIN_LR)


checkpoint_config = {
    "version": "pretrain_bpe",
    "stage": "pretrain",
    "arch": "v2_llama",
    "vocab_size": vocab_size,
    "block_size": BLOCK_SIZE,
    "n_embd": N_EMBD,
    "n_head": N_HEAD,
    "n_kv_head": globals().get("N_KV_HEAD", N_HEAD),
    "n_layer": N_LAYER,
    "ffn_hidden": getattr(raw_model, "ffn_hidden", None),
    "rope_theta": globals().get("ROPE_THETA", 10000.0),
    "rms_eps": getattr(raw_model, "rms_eps", 1e-5),
    "tie_weights": globals().get("TIE_WEIGHTS", True),
    "dropout": DROPOUT,
    "tokenizer": "bpe",
    "bpe_vocab_size": BPE_VOCAB_SIZE,
    "bpe_min_pair_freq": BPE_MIN_PAIR_FREQ,
    "pretrain_files": PRETRAIN_FILES,
    "tokenizer_train_files": globals().get("TOKENIZER_TRAIN_FILES", PRETRAIN_FILES),
    "train_ratio": TRAIN_RATIO,
    "max_chars_bloque": MAX_CHARS_BLOQUE,
    "seed": SEED,
    "early_stopping": EARLY_STOPPING,
    "patience": PATIENCE,
    "min_delta": MIN_DELTA,
    "grad_clip": GRAD_CLIP,
    "validation_protocol": VALIDATION_PROTOCOL,
    "eval_iters_train": EVAL_ITERS_TRAIN,
    "eval_iters_val": EVAL_ITERS_VAL,
    "train_rng_seed": TRAIN_RNG_SEED,
    "train_eval_seed": TRAIN_EVAL_SEED,
    "validation_seed": VALIDATION_SEED,
    "recorte_lr_activo": RECORTE_LR_ACTIVO,
    "nuevo_lr_decay_iters": NUEVO_LR_DECAY_ITERS if RECORTE_LR_ACTIVO else None,
}


best_val_loss = float("inf")
best_checkpoint_path = globals().get(
    "BEST_CHECKPOINT_PATH", CHECKPOINT_PATH.replace(".pt", "_best.pt")
)
resume_checkpoint_path = CHECKPOINT_PATH.replace(".pt", "_resume.pt")

patience_counter = 0
ultima_iteracion = 0
iteracion_inicial = 0


def guardar_modelo(path, iteracion, best_val_loss_actual, patience_actual=0):
    torch.save(
        {
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "tokenizer_state": tokenizer.get_state(),
            "config": checkpoint_config,
            "iteracion": iteracion,
            "best_val_loss": best_val_loss_actual,
            "patience_counter": patience_actual,
            "validation_protocol": VALIDATION_PROTOCOL,
            # Estados RNG. Desde fixed_val_v1 tambien se guarda el Generator
            # NumPy exclusivo que decide los batches de ENTRENAMIENTO.
            "rng_train_numpy": rng_train.bit_generator.state,
            "muestreador_train_state": muestreador_train.get_state(),
            "corpus_activo": _corpus_activo,
            # WSD: iteracion/LR de arranque del annealing final, para que
            # activarlo sea seguro aunque el proceso se reinicie a mitad.
            "wsd_iter_activacion_annealing": _iter_activacion_wsd_annealing,
            "wsd_lr_activacion_annealing": _lr_en_activacion_wsd_annealing,
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
        },
        path,
    )


# ---------------------------------------------------------------------------
# Bateria de evaluacion automatica al guardar checkpoint
# ---------------------------------------------------------------------------

BATERIA_ACTIVA = globals().get("BATERIA_ACTIVA", False)
BATERIA_DOC = globals().get("BATERIA_DOC",
                            os.path.join(_dir_ckpt, "resultados_bateria.csv"))
BATERIA_TIMEOUT_S = globals().get("BATERIA_TIMEOUT_S", 3600)
BATERIA_COMANDOS = globals().get("BATERIA_COMANDOS", [])

_bateria_log_es_nuevo = not os.path.isfile(BATERIA_DOC)
_fh_bateria = None
if BATERIA_ACTIVA and BATERIA_COMANDOS:
    _fh_bateria = open(BATERIA_DOC, "a", newline="", encoding="utf-8")
    _csv_bateria = csv.writer(_fh_bateria)
    if _bateria_log_es_nuevo:
        _csv_bateria.writerow(["iteracion", "checkpoint", "prueba", "valor",
                               "estado", "timestamp"])
        _fh_bateria.flush()


def _parsear_informe_compacto(path):
    """Parsea el informe compacto que escribe benchmark_bcc_es.py
    (escribir_informe_compacto) y devuelve una lista de (prueba, valor):

      es_ppl, es_bpc, en_ppl, en_bpc, codigo_py_ppl, codigo_py_bpc,
      codigo_json_ppl, codigo_json_bpc,
      razonamiento_<categoria>..., razonamiento_global,
      conocimiento_<categoria>..., conocimiento_global

    Antes solo se guardaba la ULTIMA linea no vacia del stdout del
    comando, que no era un resumen (podia ser un ejemplo de texto
    generado o una fila de tabla suelta). Esto lee el informe compacto
    en si, que ya trae bits/char y perplejidad por ES/EN/codigo y el %
    de cada categoria de los bancos de razonamiento y conocimiento.
    """
    filas = []
    if not os.path.isfile(path):
        return filas

    with open(path, "r", encoding="utf-8") as f:
        lineas_txt = f.read().splitlines()

    CABECERAS_GEN = {
        "ESPAÑOL": "es",
        "INGLÉS": "en",
        "CÓDIGO-PYTHON": "codigo_py",
        "CÓDIGO-JSON": "codigo_json",
    }

    seccion = None
    prefijo_gen = None

    for linea in lineas_txt:
        stripped = linea.strip()

        if stripped in CABECERAS_GEN:
            seccion, prefijo_gen = "gen", CABECERAS_GEN[stripped]
            continue
        if stripped == "BANCO DE RAZONAMIENTO":
            seccion = "razonamiento"
            continue
        if stripped == "BANCO DE CONOCIMIENTO":
            seccion = "conocimiento"
            continue
        if not stripped or stripped.startswith("---") or stripped.startswith("==="):
            continue

        if seccion == "gen":
            m = re.match(r"Perplexity \(mediana\):\s*([\d.]+)", stripped)
            if m:
                filas.append((f"{prefijo_gen}_ppl", m.group(1)))
                continue
            m = re.match(r"Bits/char \(mediana\):\s*([\d.]+)", stripped)
            if m:
                filas.append((f"{prefijo_gen}_bpc", m.group(1)))
                continue

        elif seccion in ("razonamiento", "conocimiento"):
            m = re.match(r"(.+?):\s*([\d.]+)%\s*$", stripped)
            if m:
                nombre, valor = m.group(1).strip(), m.group(2)
                clave = "global" if nombre.lower() == "media total" else nombre
                filas.append((f"{seccion}_{clave}", valor))

    return filas


def lanzar_bateria(ruta_checkpoint, iteracion, train_loss=None, val_loss=None):
    """Lanza cada comando de BATERIA_COMANDOS sobre ruta_checkpoint.

    Ademas de la ultima linea de stdout (se conserva por compatibilidad),
    si la plantilla del comando usa {informe_compacto}, benchmark_bcc_es.py
    escribe ahi su informe compacto y aqui se parsea (_parsear_informe_compacto)
    para anadir UNA FILA POR METRICA al CSV: bits/char y perplejidad de
    ES/EN/codigo-Python/codigo-JSON, y el % de cada categoria de los bancos
    de razonamiento y conocimiento (BCC-ES). Para activarlo, la plantilla
    en BATERIA_COMANDOS (config_pretrain.py) debe incluir:
        --informe-compacto {informe_compacto}
    (ademas de --checkpoint {checkpoint} --iter {iter} --train-loss
    {train_loss} --val-loss {val_loss}, como ya tenias).

    train_loss/val_loss se pasan aqui porque el checkpoint (.pt) NO guarda
    train_loss, y benchmark_bcc_es.py los necesita para su cabecera -- se
    sustituyen en la plantilla via {train_loss}/{val_loss} si los usa.

    Se llama de forma sincrona justo al guardar un checkpoint (best o
    resume) -- pausa el entrenamiento mientras evalua, pero sobre una
    corrida de dias esto son minutos. Cualquier fallo de una prueba se
    registra y se sigue: nunca debe tumbar el entrenamiento.
    """
    if not BATERIA_ACTIVA or not BATERIA_COMANDOS:
        return

    import subprocess

    _log()
    _log(f"[BATERIA] Lanzando bateria de pruebas en iteracion {iteracion} "
          f"({len(BATERIA_COMANDOS)} prueba(s)) -- el entrenamiento queda "
          f"PAUSADO hasta que termine...")
    t0_bateria = time.time()

    # Ruta fija del informe compacto de esta iteracion. Si la plantilla del
    # comando la usa via {informe_compacto}, benchmark_bcc_es.py lo escribe
    # ahi y se parsea justo despues de ejecutar el comando.
    ruta_informe_compacto = os.path.join(
        _dir_ckpt, f"informe_compacto_it{iteracion}.txt"
    )

    for nombre, plantilla in BATERIA_COMANDOS:
        cmd = plantilla.format(
            checkpoint=ruta_checkpoint, iter=iteracion,
            train_loss=f"{train_loss:.4f}" if train_loss is not None else "",
            val_loss=f"{val_loss:.4f}" if val_loss is not None else "",
            informe_compacto=ruta_informe_compacto,
        )
        _log(f"[BATERIA] {nombre}...")
        try:
            res = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=BATERIA_TIMEOUT_S,
            )
            lineas = [l for l in (res.stdout or "").splitlines() if l.strip()]
            valor = lineas[-1].strip() if lineas else ""
            estado = "OK" if res.returncode == 0 else f"FALLO_rc{res.returncode}"
        except subprocess.TimeoutExpired:
            valor, estado = "", "TIMEOUT"
        except Exception as e:  # noqa: BLE001 - nunca tumbar el entrenamiento
            valor, estado = str(e), "ERROR"

        _csv_bateria.writerow([iteracion, ruta_checkpoint, nombre, valor,
                               estado, time.strftime("%Y-%m-%d %H:%M:%S")])
        _fh_bateria.flush()
        _log(f"[BATERIA] {nombre}: {valor} ({estado})")

        # Filas detalladas por metrica, si el comando genero el informe
        # compacto (ver docstring: requiere {informe_compacto} en la
        # plantilla). "prueba" queda como "<nombre>/<metrica>", p.ej.
        # "BCC-ES/es_bpc", "BCC-ES/razonamiento_global".
        filas_metricas = _parsear_informe_compacto(ruta_informe_compacto)
        for prueba, valor_metrica in filas_metricas:
            _csv_bateria.writerow([iteracion, ruta_checkpoint,
                                   f"{nombre}/{prueba}", valor_metrica,
                                   "OK", time.strftime("%Y-%m-%d %H:%M:%S")])
        if filas_metricas:
            _fh_bateria.flush()
            _log(f"[BATERIA] {nombre}: +{len(filas_metricas)} metricas "
                  f"detalladas desde {ruta_informe_compacto}")

    _log(f"[BATERIA] Bateria completada en {time.time()-t0_bateria:.1f}s "
          f"-- reanudando entrenamiento.")
    _log()


def cargar_para_reanudar(path):
    """Restaura modelo, optimizer, scaler, iteracion, best_val_loss, patience
    y RNG desde un checkpoint de reanudacion. Devuelve (iteracion_siguiente,
    best_val_loss, patience_counter)."""
    _log(f"[REANUDAR] Encontrado checkpoint de reanudacion: {path}")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    raw_model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    it_guardada = ckpt.get("iteracion", 0)
    bvl = ckpt.get("best_val_loss", float("inf"))
    pc = ckpt.get("patience_counter", 0)

    # WSD: restaurar el punto de arranque del annealing final si ya se
    # habia activado en una corrida anterior (None si el annealing nunca
    # se activo, o si el checkpoint es anterior a esta funcionalidad).
    global _iter_activacion_wsd_annealing, _lr_en_activacion_wsd_annealing
    _iter_activacion_wsd_annealing = ckpt.get("wsd_iter_activacion_annealing")
    _lr_en_activacion_wsd_annealing = ckpt.get("wsd_lr_activacion_annealing")
    if WSD_ACTIVO and WSD_ANNEALING_ACTIVO and _iter_activacion_wsd_annealing is not None:
        _log(f"[WSD-ANNEALING] Restaurado: activado originalmente en iter "
              f"{_iter_activacion_wsd_annealing}, LR de partida "
              f"{_lr_en_activacion_wsd_annealing:.6e}")

    # Si el checkpoint procede del protocolo antiguo (validacion aleatoria),
    # su best_val_loss NO es comparable con el nuevo examen fijo. Se conserva
    # el entrenamiento, pero se reinicia SOLO la referencia de validacion.
    protocolo_ckpt = ckpt.get("validation_protocol")
    if protocolo_ckpt != VALIDATION_PROTOCOL:
        _log(
            f"[VALIDACION] Cambio de protocolo: {protocolo_ckpt or 'legacy_random'} "
            f"-> {VALIDATION_PROTOCOL}. Reiniciando best_val_loss y patience; "
            f"pesos/optimizer/iteracion se conservan."
        )
        bvl = float("inf")
        pc = 0

    # Restaurar RNG NumPy del entrenamiento si el checkpoint ya es nuevo.
    estado_np = ckpt.get("rng_train_numpy")
    if estado_np is not None:
        rng_train.bit_generator.state = estado_np
    else:
        _log(
            "[RNG] Checkpoint legacy sin rng_train_numpy: se inicia una secuencia "
            "NumPy determinista nueva. Desde el proximo resume sera exacta."
        )

    # Restaurar la posicion del muestreador secuencial (cobertura exacta,
    # ver MuestreadorSecuencial). Si el checkpoint es anterior a este
    # cambio (no tiene la clave), se empieza una epoca nueva desde cero
    # sobre el train.bin actual -- es la unica opcion razonable, ya que
    # el esquema anterior (aleatorio con reemplazo) no llevaba ninguna
    # posicion que se pueda traducir a este nuevo esquema.
    global muestreador_train
    estado_muestreador = ckpt.get("muestreador_train_state")
    if estado_muestreador is not None:
        muestreador_train = MuestreadorSecuencial.from_state(
            len(train_data), BLOCK_SIZE, BATCH_SIZE, estado_muestreador
        )
        _log(f"[MUESTREADOR] Reanudado: epoca={muestreador_train.epoca} "
              f"posicion={muestreador_train.posicion}/{muestreador_train.n_bloques} bloques")
    else:
        _log("[MUESTREADOR] Checkpoint anterior al muestreo secuencial: "
              "empezando epoca nueva (cobertura exacta) desde el principio "
              "del train.bin actual.")

    # Restaurar RNG de Torch/CUDA.
    # IMPORTANTE: map_location=DEVICE mueve TODOS los tensores del
    # checkpoint a ese device, incluido rng_torch -- pero el estado del
    # generador de CPU debe restaurarse siempre como ByteTensor en CPU,
    # o torch.set_rng_state lanza TypeError. Se fuerza .cpu() aqui.
    if ckpt.get("rng_torch") is not None:
        rng_torch_cpu = ckpt["rng_torch"]
        if hasattr(rng_torch_cpu, "cpu"):
            rng_torch_cpu = rng_torch_cpu.cpu()
        torch.set_rng_state(rng_torch_cpu)
    if ckpt.get("rng_cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        except Exception:
            pass  # distinto numero de GPUs; no es critico

    # Continuamos en la iteracion SIGUIENTE a la guardada.
    it_siguiente = it_guardada + 1
    _log(f"[REANUDAR] Reanudando en iteracion {it_siguiente} "
          f"(best_val_loss={bvl:.4f}, patience={pc})")

    # CAMBIO 2026-08-23 (v2): si la iteracion de reanudacion ya supera el
    # umbral de swap a FINAL, cargar train_final.bin AHORA en vez de dejar
    # que el bucle principal lo detecte en la primera vuelta -- evita
    # arrancar sobre meseta (ya cargada por defecto al importar el script)
    # y tener que recargar en el primer paso, y deja los logs de arranque
    # ya reflejando el fichero correcto.
    global _corpus_activo
    if (TRAIN_BIN_PATH_FINAL and ITER_CAMBIO_A_FINAL
            and it_siguiente >= ITER_CAMBIO_A_FINAL and _corpus_activo != "final"):
        _log(f"[CORPUS] Reanudando en iteracion {it_siguiente} >= "
              f"ITER_CAMBIO_A_FINAL={ITER_CAMBIO_A_FINAL:,} -- cargando "
              f"train_final.bin directamente (no meseta).")
        _cargar_train_data(TRAIN_BIN_PATH_FINAL)
        _corpus_activo = "final"
        # El muestreador que _cargar_train_data acaba de crear es NUEVO
        # (permutacion desde cero) -- si el checkpoint ya tenia progreso
        # sobre train_final.bin (guardado con corpus_activo="final"),
        # restaurar esa posicion en vez de la recien creada.
        if estado_muestreador is not None and ckpt.get("corpus_activo") == "final":
            muestreador_train = MuestreadorSecuencial.from_state(
                len(train_data), BLOCK_SIZE, BATCH_SIZE, estado_muestreador
            )
            _log(f"[MUESTREADOR] Posicion sobre train_final.bin restaurada: "
                  f"epoca={muestreador_train.epoca} "
                  f"posicion={muestreador_train.posicion}/{muestreador_train.n_bloques}")

    return it_siguiente, bvl, pc


# Si existe un checkpoint de reanudacion y RESUME esta activo, continuar.
if RESUME and os.path.isfile(resume_checkpoint_path):
    iteracion_inicial, best_val_loss, patience_counter = cargar_para_reanudar(
        resume_checkpoint_path
    )
else:
    if RESUME:
        _log("[REANUDAR] No hay checkpoint de reanudacion previo. "
              "Empezando desde cero.")


# ============================================================
# Entrenamiento
# ============================================================

_log()
_log("Alfa1 V2 iniciado.")
_log("Etapa: pretraining BPE")
_log("Tokenizacion: BPE (datos desde .bin memmap)")
_log(f"Dispositivo: {DEVICE}")
_log(f"Corpus train activo: {_corpus_activo} ({TRAIN_BIN_PATH_MESETA})")
_log(f"Train tokens: {len(train_data)}")
_log(f"Val tokens: {len(val_data)}")
_log(f"Tamano del vocabulario: {vocab_size}")
_log(f"Parametros del modelo: {num_params:,}")
_log(f"Batch size: {BATCH_SIZE}")
_log(f"Block size: {BLOCK_SIZE}")
_log(f"Tokens por iteracion: {tokens_por_iteracion}")
_log(f"Iteraciones por epoca aproximada (meseta): {len(train_data)/tokens_por_iteracion:.2f}")
if TRAIN_BIN_PATH_FINAL and ITER_CAMBIO_A_FINAL:
    _log(f"Swap automatico a train_final.bin programado en iteracion {ITER_CAMBIO_A_FINAL:,}")
else:
    _log("[AVISO] TRAIN_BIN_PATH_FINAL/ITER_CAMBIO_A_FINAL no definidos -- "
         "no habra swap automatico a la fase FINAL.")
_log(f"Grad accum steps: {GRAD_ACCUM_STEPS}")
_log(f"Tokens por microbatch: {tokens_por_microbatch}")
_log(f"Tokens por iteracion/update: {tokens_por_iteracion}")
_log(f"Max iters: {MAX_ITERS}")
_log(f"Eval interval: {EVAL_INTERVAL}")
_log(f"Protocolo validacion: {VALIDATION_PROTOCOL} (offsets fijos)")
_log(
    f"Eval train fija: {EVAL_ITERS_TRAIN} batches = "
    f"{EVAL_ITERS_TRAIN * BATCH_SIZE * BLOCK_SIZE:,} tokens"
)
_log(
    f"Eval val fija: {EVAL_ITERS_VAL} batches = "
    f"{EVAL_ITERS_VAL * BATCH_SIZE * BLOCK_SIZE:,} tokens"
)
_log(f"Early stopping: {EARLY_STOPPING}")
_log(f"Patience: {PATIENCE}")
_log(f"Min delta: {MIN_DELTA}")
_log(f"Grad clip: {GRAD_CLIP}")
_log(f"Recorte de coseno activo: {RECORTE_LR_ACTIVO}"
      + (f" (nuevo horizonte: iter {NUEVO_LR_DECAY_ITERS})" if RECORTE_LR_ACTIVO else ""))
_log(f"WSD activo: {WSD_ACTIVO}"
      + (f" (LR congelado={WSD_LR:.6e}, annealing_activo={WSD_ANNEALING_ACTIVO}"
         f"{f', decay en {WSD_ANNEALING_ITERS} iters' if WSD_ANNEALING_ACTIVO else ''})"
         if WSD_ACTIVO else ""))
_log(f"Checkpoint final: {CHECKPOINT_PATH}")
_log(f"Mejor checkpoint: {best_checkpoint_path}")

# Ratio de Chinchilla: tokens VISTOS por parametro durante el entrenamiento.
# Lo que gobierna el aprendizaje no es el tamano del .bin sino cuantos tokens
# procesa el modelo: MAX_ITERS * tokens_por_iteracion. Con repeticion por
# fuente y muestreo con reemplazo, "tokens vistos" != "tokens unicos".
# El optimo compute-eficiente de Chinchilla es ~20 tokens vistos/parametro.
tokens_vistos = MAX_ITERS * tokens_por_iteracion
chinchilla_ratio = tokens_vistos / num_params
factor_optimo = chinchilla_ratio / 20 if chinchilla_ratio else float("inf")
_log(
    f"Tokens unicos en train.bin: {len(train_data):,}"
)
_log(
    f"Tokens VISTOS (MAX_ITERS x tok/iter): {tokens_vistos:,} "
    f"({tokens_vistos/len(train_data):.2f} pasadas medias sobre el corpus)"
)
_log(
    f"Ratio Chinchilla: {chinchilla_ratio:.2f} tokens vistos/param "
    f"({factor_optimo:.2f}x el ratio de referencia ~20)"
)
_log()

tiempo_ciclo_inicio = time.time()

# Abrir el CSV de losses en modo append. Si el fichero ya existia (p.ej.
# tras un RESUME) no se reescribe la cabecera, se sigue anadiendo.
_log_es_nuevo = not os.path.isfile(LOSS_LOG_PATH)
_fh_loss_log = open(LOSS_LOG_PATH, "a", newline="", encoding="utf-8")
_csv_loss_log = csv.writer(_fh_loss_log)
if _log_es_nuevo:
    _csv_loss_log.writerow([
        "iteracion", "train_loss", "train_sem", "val_loss", "val_sem",
        "val_ic95_halfwidth", "lr", "validation_protocol", "val_tokens",
        "timestamp", "grad_norm",
    ])
    _fh_loss_log.flush()
_log(f"Log de losses: {LOSS_LOG_PATH}")

# Ultimas losses conocidas (solo se recalculan cada EVAL_INTERVAL) --
# se muestran en la barra de progreso entre evaluaciones para no dejarla
# en blanco.
_ultimo_train_loss = None
_ultimo_val_loss = None
# Norma del gradiente del ultimo paso de entrenamiento (ver donde se
# captura, junto a clip_grad_norm_). None hasta que se haga el primer
# paso -- si una evaluacion cae antes (p.ej. justo al reanudar), se
# registra vacio en el CSV en vez de romper.
_grad_norm_ultimo = None

# CAMBIO 2026-08-23 (fix real: barra rota en la ventana visible): con
# dynamic_ncols=True, tqdm intenta leer el ancho real de la terminal --
# pero stdout va a traves de `tee` (pipe), no es un TTY directo, asi que
# tqdm no puede detectar el ancho real y la linea (barra + postfix largo)
# acaba siendo mas ancha que la ventana. El terminal la envuelve en varias
# filas fisicas, y el `\r` de tqdm solo vuelve al principio de la fila
# envuelta, no al principio real -- efecto visual de fragmentos apilados
# reportado por el usuario. Fijar un ancho FIJO y conservador evita
# depender de una deteccion de TTY que no funciona a traves del pipe.
_barra = tqdm(
    range(iteracion_inicial, MAX_ITERS + 1),
    desc="Pretraining",
    unit="it",
    ncols=110,
)
for iteracion in _barra:
    ultima_iteracion = iteracion
    _t_iter_inicio = time.time()

    # CAMBIO 2026-08-23 (v2): swap automatico meseta->final en caliente.
    # Comprobacion barata (dos comparaciones) en cada iteracion; solo hace
    # trabajo real la vez que realmente cruza el umbral (_corpus_activo
    # evita repetir el swap en cada vuelta siguiente). No afecta a
    # torch.compile/CUDA Graphs: solo cambia el ORIGEN de los tokens
    # (train_data/muestreador_train), no la forma de los tensores
    # (BLOCK_SIZE/BATCH_SIZE no cambian aqui) ni el grafo del modelo.
    if (TRAIN_BIN_PATH_FINAL and ITER_CAMBIO_A_FINAL
            and iteracion >= ITER_CAMBIO_A_FINAL and _corpus_activo != "final"):
        _log(f"[CORPUS] Iteracion {iteracion:,} >= ITER_CAMBIO_A_FINAL="
              f"{ITER_CAMBIO_A_FINAL:,} -- cambiando a train_final.bin.")
        _cargar_train_data(TRAIN_BIN_PATH_FINAL)
        _corpus_activo = "final"
        _log(f"[CORPUS] train_final.bin activo: {len(train_data):,} tokens, "
              f"{muestreador_train.n_bloques:,} bloques de {BLOCK_SIZE}.")

    if iteracion % EVAL_INTERVAL == 0:
        losses = estimate_loss(model)

        train_loss = losses["train"]
        val_loss = losses["val"]
        train_sem = losses["train_sem"]
        val_sem = losses["val_sem"]
        val_ic95 = 1.96 * val_sem
        _ultimo_train_loss = train_loss
        _ultimo_val_loss = val_loss

        # LR de esta iteracion, solo para el log (el scheduler real vuelve
        # a llamar a lr_para_iteracion mas abajo antes del paso de
        # entrenamiento; llamarla aqui tambien es seguro porque es
        # deterministica para el mismo "iteracion").
        _lr_log = lr_para_iteracion(iteracion)
        _csv_loss_log.writerow([
            iteracion, f"{train_loss:.6f}", f"{train_sem:.6f}",
            f"{val_loss:.6f}", f"{val_sem:.6f}", f"{val_ic95:.6f}",
            f"{_lr_log:.8e}", VALIDATION_PROTOCOL,
            EVAL_ITERS_VAL * BATCH_SIZE * BLOCK_SIZE,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            # grad_norm va la ULTIMA a proposito: los lectores existentes
            # del CSV (p.ej. el vigilante de loss) acceden por indice a las
            # columnas anteriores, asi que anadir al final no los rompe.
            f"{_grad_norm_ultimo:.4f}" if _grad_norm_ultimo is not None else "",
        ])
        _fh_loss_log.flush()

        tiempo_transcurrido = time.time() - tiempo_ciclo_inicio
        iters_en_ciclo = max(1, iteracion - getattr(estimate_loss, "_ultima_iter", iteracion_inicial))
        s_por_iter = tiempo_transcurrido / iters_en_ciclo
        iters_restantes = MAX_ITERS - iteracion
        eta_horas = (iters_restantes * s_por_iter) / 3600

        # tok/s: en pretraining tokens_por_iteracion es constante (secuencias
        # densas de BLOCK_SIZE, sin padding), asi que el rendimiento en tokens
        # es directo. Util para estimar el coste de un presupuesto de tokens
        # dado: por ejemplo, 1e9 / tok_s segundos para 1B tokens.
        tok_s = tokens_por_iteracion / s_por_iter if s_por_iter > 0 else 0.0
        horas_1b = (1_000_000_000 / tok_s / 3600) if tok_s > 0 else float("inf")

        # TFLOPS/s reales alcanzados (no el pico teorico de la tarjeta):
        # FLOPS_POR_TOKEN es constante (fijada al crear el modelo), asi que
        # basta multiplicar por los tokens de la iteracion y dividir por el
        # tiempo. Compara esto contra el pico BF16 de tu GPU para saber que
        # % de utilizacion real estas consiguiendo (MFU).
        flops_iter = FLOPS_POR_TOKEN * tokens_por_iteracion
        tflops_s = (flops_iter / s_por_iter / 1e12) if s_por_iter > 0 else 0.0

        _log(
            f"Iteracion {iteracion}: "
            f"train loss {train_loss:.4f}, "
            f"val loss {val_loss:.4f} +/- {val_ic95:.4f} (IC95 aprox.)"
        )
        _log(
            f"  Tiempo ciclo: {tiempo_transcurrido:.1f}s "
            f"({s_por_iter:.2f} s/iter | {tok_s:,.0f} tok/s | {tflops_s:,.1f} TFLOPS/s) "
            f"| ETA: {eta_horas:.1f}h"
        )
        _log(
            f"  A este ritmo: 1B tokens = {horas_1b:.2f}h "
            f"({horas_1b/24:.2f} dias)"
        )

        estimate_loss._ultima_iter = iteracion
        tiempo_ciclo_inicio = time.time()

        if val_loss < best_val_loss - MIN_DELTA:
            _val_loss_anterior = best_val_loss
            best_val_loss = val_loss
            patience_counter = 0

            guardar_modelo(
                best_checkpoint_path,
                iteracion=iteracion,
                best_val_loss_actual=best_val_loss,
                patience_actual=patience_counter,
            )

            _mejora = (_val_loss_anterior - val_loss
                       if _val_loss_anterior != float("inf") else None)
            _mejora_txt = f"mejora={_mejora:.4f}" if _mejora is not None else "primer best"
            _log(f"[BEST] Nuevo mejor checkpoint guardado -- iteracion {iteracion:,} "
                  f"-- val_loss={val_loss:.4f} ({_mejora_txt}) -- {best_checkpoint_path}")
            lanzar_bateria(best_checkpoint_path, iteracion,
                          train_loss=train_loss, val_loss=val_loss)

        else:
            patience_counter += 1

            _log(
                f"Sin mejora suficiente en val loss "
                f"({patience_counter}/{PATIENCE})"
            )

            if EARLY_STOPPING and patience_counter >= PATIENCE:
                _log()
                _log("Early stopping activado.")
                _log(f"Mejor val loss: {best_val_loss:.4f}")
                _log(f"Mejor checkpoint: {best_checkpoint_path}")
                break

    # Checkpoint de REANUDACION periodico (se sobrescribe siempre; su unico
    # fin es poder continuar tras una caida). Distinto del best.
    if (iteracion > 0 and CKPT_RESUME_INTERVAL > 0
            and iteracion % CKPT_RESUME_INTERVAL == 0):
        guardar_modelo(
            resume_checkpoint_path,
            iteracion=iteracion,
            best_val_loss_actual=best_val_loss,
            patience_actual=patience_counter,
        )
        _log(f"[REANUDAR] Checkpoint de reanudacion guardado en iter "
              f"{iteracion}: {resume_checkpoint_path}")
        # train_loss/val_loss son los de la ULTIMA evaluacion (EVAL_INTERVAL),
        # no necesariamente de esta misma iteracion. Si CKPT_RESUME_INTERVAL
        # dispara antes de la primera evaluacion, aun no existen -> None.
        lanzar_bateria(resume_checkpoint_path, iteracion,
                      train_loss=locals().get("train_loss"),
                      val_loss=locals().get("val_loss"))

    if iteracion == MAX_ITERS:
        break

    # Scheduler: fijar el LR de esta iteracion (warmup + cosine, con
    # recorte opcional -- ver lr_para_iteracion)
    lr_actual = lr_para_iteracion(iteracion)
    for grupo in optimizer.param_groups:
        grupo["lr"] = lr_actual

    optimizer.zero_grad(set_to_none=True)

    for micro_step in range(GRAD_ACCUM_STEPS):
        xb, yb = get_batch("train", muestreador=muestreador_train)

        # CUDA Graphs (activas con TORCH_COMPILE_MODE="max-autotune" o
        # "reduce-overhead") reutilizan los mismos buffers de salida entre
        # invocaciones del modelo compilado. Con grad accumulation llamamos
        # al modelo GRAD_ACCUM_STEPS veces seguidas antes de que el
        # optimizer de el paso; sin avisar explicitamente de que empieza
        # una invocacion nueva, CUDA Graphs puede sobrescribir el output de
        # una llamada anterior antes de que su backward() lo haya consumido
        # del todo, lanzando "accessing tensor output of CUDAGraphs that
        # has been overwritten by a subsequent run". Es la mitigacion que
        # el propio mensaje de error de PyTorch recomienda.
        if (DEVICE == "cuda" and hasattr(torch, "compiler")
                and hasattr(torch.compiler, "cudagraph_mark_step_begin")):
            torch.compiler.cudagraph_mark_step_begin()

        with torch.amp.autocast(
            device_type=AMP_DEVICE_TYPE,
            dtype=AMP_DTYPE,
            enabled=USE_AMP,
        ):
            _, loss, _ = model(xb, yb)

            # Importante: dividir la loss para que el gradiente total
            # no sea GRAD_ACCUM_STEPS veces más grande.
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

    if GRAD_CLIP is not None and GRAD_CLIP > 0:
        scaler.unscale_(optimizer)
        # clip_grad_norm_ YA calcula la norma total para poder recortar --
        # capturarla aqui no cuesta computo extra, y es la senal de alarma
        # mas temprana que hay para divergencia: la norma del gradiente se
        # dispara ANTES de que la loss se mueva. Se registra en el CSV
        # (columna grad_norm) en cada evaluacion.
        _grad_norm_ultimo = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        )

    scaler.step(optimizer)
    scaler.update()

    _t_iter = time.time() - _t_iter_inicio
    _tok_s_inst = tokens_por_iteracion / _t_iter if _t_iter > 0 else 0.0
    # % de decaimiento del coseno completado (0% = justo tras el warmup, en
    # LEARNING_RATE pico; 100% = en LR_DECAY_ITERS, en MIN_LR). Durante el
    # propio warmup se queda a 0% (aun no ha empezado a decaer) en vez de
    # dar una cifra sin sentido -- ver discusion en el cambio que anadio esto.
    _iters_desde_warmup = max(0, iteracion - WARMUP_ITERS)
    _decay_pct = min(100.0, 100.0 * _iters_desde_warmup
                      / max(1, LR_DECAY_ITERS - WARMUP_ITERS))
    # Postfix compacto a proposito (ver comentario junto a ncols=110 mas
    # arriba): sin comas de miles ni etiquetas largas, para que la linea
    # completa quepa en un ancho de terminal modesto sin envolver.
    _barra.set_postfix({
        "tr": f"{_ultimo_train_loss:.3f}" if _ultimo_train_loss is not None else "-",
        "vl": f"{_ultimo_val_loss:.3f}" if _ultimo_val_loss is not None else "-",
        "tok/s": f"{_tok_s_inst:.0f}",
        "lr": f"{lr_actual:.1e}",
        "dec": f"{_decay_pct:.0f}%",
    }, refresh=False)


_fh_loss_log.close()
if _fh_bateria is not None:
    _fh_bateria.close()

guardar_modelo(
    CHECKPOINT_PATH,
    iteracion=ultima_iteracion,
    best_val_loss_actual=best_val_loss,
    patience_actual=patience_counter,
)

_log()
_log(f"Checkpoint final guardado: {CHECKPOINT_PATH}")
_log(f"Mejor checkpoint: {best_checkpoint_path}")
_log(f"Mejor val loss: {best_val_loss:.4f}")
_log(f"Ultima iteracion: {ultima_iteracion}")
_log(f"Log de losses: {LOSS_LOG_PATH}")
