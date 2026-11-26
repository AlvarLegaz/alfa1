# train_midtrain.py
"""
Fase de MIDTRAIN (annealing/decay final) de Alfa1, en script y checkpoints
PROPIOS -- separado de train_pretrain_v2.py.

REQUISITO: el corpus de midtrain (12BT: conocimiento 6 / razonamiento 3 /
codigo 3) ya tiene que estar tokenizado en TRAIN_BIN_PATH / VAL_BIN_PATH
(ver config_annealing.py) antes de lanzar esto.

--------------------------------------------------------------------
DISENO (ver conversacion del 2026-08-27):
--------------------------------------------------------------------
- Warm start: arranca cargando SOLO los pesos de un checkpoint de
  pretrain (INIT_FROM_CHECKPOINT). Optimizer, iteracion, RNG y
  muestreador se reinician a cero -- no se hereda estado de
  entrenamiento del pretrain, solo el modelo.
- Sin warmup: el LR arranca directo en LR_PICO en la iteracion 0 de
  ESTE script (el modelo ya viene "caliente").
- Scheduler: UN SOLO decay 1-sqrt (Hagele et al. 2024 / MiniCPM) de
  LR_PICO a MIN_LR a lo largo de TODO MAX_ITERS. Al ser toda la vida de
  este script la fase de decay, no hace falta ningun mecanismo de
  "activacion a mitad de entrenamiento" como el WSD_ANNEALING_ACTIVO del
  pretrain -- lr_para_iteracion(it) es una funcion pura de (it,
  MAX_ITERS), sin estado que persistir ni que se pueda perder si el
  proceso muere de golpe.
- MAX_ITERS se deriva SOLO de la longitud real de train_midtrain.bin
  (una pasada completa) si config_annealing.py lo deja en None -- la
  duracion de esta fase la fija el tamano del corpus nuevo.
- BLOCK_SIZE sube a 8192 (trazas de razonamiento largas) con ROPE_THETA
  ajustado a juego -- ver config_annealing.py.
- Reanudacion propia: si el proceso muere y se relanza, prioriza el
  checkpoint de REANUDACION de midtrain (estado completo) sobre
  INIT_FROM_CHECKPOINT (que solo se usa en el arranque en frio).
--------------------------------------------------------------------

Reutiliza (copiado casi literal, sin cambios de fondo) de
train_pretrain_v2.py: el CSV de losses, el checkpoint de reanudacion
periodico, la bateria de evaluacion automatica, el MuestreadorSecuencial
de cobertura exacta, y la validacion fija/reproducible (fixed_val_v1).
"""

import os
import re
import math
import random
import time
import csv
import json
from tqdm import tqdm

# Ver train_pretrain_v2.py para la explicacion completa de por que _log
# usa tqdm.write() en vez de print() mientras la barra esta activa.
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

from alfa1.config_annealing import *
from alfa1.model import SanchoGPT
from alfa1.tokenizer_bpe import BPETokenizer


# ============================================================
# Ajustes por defecto si no existen en config_annealing.py
# ============================================================

EARLY_STOPPING = globals().get("EARLY_STOPPING", False)
PATIENCE = globals().get("PATIENCE", 20)
MIN_DELTA = globals().get("MIN_DELTA", 0.0)

GRAD_CLIP = globals().get("GRAD_CLIP", 1.0)

SEED = globals().get("SEED", 1337)

EVAL_ITERS_TRAIN = globals().get("EVAL_ITERS_TRAIN", 100)
EVAL_ITERS_VAL = globals().get("EVAL_ITERS_VAL", 1500)
VALIDATION_PROTOCOL = globals().get("VALIDATION_PROTOCOL", "fixed_val_v1")
TRAIN_RNG_SEED = globals().get("TRAIN_RNG_SEED", SEED + 100)
TRAIN_EVAL_SEED = globals().get("TRAIN_EVAL_SEED", SEED + 200)
VALIDATION_SEED = globals().get("VALIDATION_SEED", SEED + 300)

GRAD_ACCUM_STEPS = globals().get("GRAD_ACCUM_STEPS", 1)

RESUME = globals().get("RESUME", True)
CKPT_RESUME_INTERVAL = globals().get("CKPT_RESUME_INTERVAL", 100)

WEIGHT_DECAY = globals().get("WEIGHT_DECAY", 0.1)
BETA1 = globals().get("BETA1", 0.9)
BETA2 = globals().get("BETA2", 0.95)

LR_PICO = globals().get("LR_PICO", None)
MIN_LR = globals().get("MIN_LR", 1e-6)
if LR_PICO is None:
    raise ValueError(
        "LR_PICO no esta definido en config_annealing.py -- es el LR "
        "congelado del pretrain del que arranca el decay. Sin el, no "
        "hay forma segura de saber donde empieza la curva 1-sqrt."
    )

INIT_FROM_CHECKPOINT = globals().get("INIT_FROM_CHECKPOINT", None)

_dir_ckpt = os.path.dirname(CHECKPOINT_PATH) or "."
os.makedirs(_dir_ckpt, exist_ok=True)

_dir_datos = globals().get("DATA_BIN_DIR", os.path.dirname(TRAIN_BIN_PATH) or ".")
LOSS_LOG_PATH = globals().get("LOSS_LOG_PATH", os.path.join(_dir_datos, "loss_log_midtrain.csv"))

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
np.random.seed(SEED)

rng_train = np.random.default_rng(TRAIN_RNG_SEED)

USE_AMP = DEVICE == "cuda"
AMP_DEVICE_TYPE = "cuda" if DEVICE == "cuda" else "cpu"

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

for ruta in (TRAIN_BIN_PATH, VAL_BIN_PATH, TOKENIZER_STATE_PATH):
    if not os.path.isfile(ruta):
        raise FileNotFoundError(
            f"No existe {ruta}. Genera primero el corpus de midtrain "
            f"(12BT: conocimiento/razonamiento/codigo) con el pipeline "
            f"de mezcla antes de lanzar train_midtrain.py."
        )

# ============================================================
# Tokenizador
# ============================================================

tokenizer = BPETokenizer()
with open(TOKENIZER_STATE_PATH, "r", encoding="utf-8") as f:
    tokenizer.load_state(json.load(f))

vocab_size = tokenizer.vocab_size()

STORY_ID = tokenizer.story_id
if STORY_ID is None:
    raise RuntimeError("No se encontro <|endofstory|> en el tokenizador.")


# ============================================================
# Datos: memmap a los .bin (RAM plana) -- un solo corpus, sin swap
# ============================================================

train_data = np.memmap(TRAIN_BIN_PATH, dtype=DTYPE_TOKENS, mode="r")
val_data = np.memmap(VAL_BIN_PATH, dtype=DTYPE_TOKENS, mode="r")

if len(train_data) <= BLOCK_SIZE:
    raise ValueError(
        f"Train data demasiado pequeno para BLOCK_SIZE={BLOCK_SIZE}. "
        f"Train tokens: {len(train_data)}"
    )
if len(val_data) <= BLOCK_SIZE:
    raise ValueError(
        f"Val data demasiado pequeno para BLOCK_SIZE={BLOCK_SIZE}. "
        f"Val tokens: {len(val_data)}"
    )

tokens_por_microbatch = BATCH_SIZE * BLOCK_SIZE
tokens_por_iteracion = BATCH_SIZE * BLOCK_SIZE * GRAD_ACCUM_STEPS

# MAX_ITERS: si config_annealing.py lo deja en None, se deriva SOLO de la
# longitud real del corpus (una pasada completa) -- ver cabecera del
# fichero. Se hace aqui, justo tras abrir train_data, porque es el primer
# punto en que len(train_data) existe.
MAX_ITERS = globals().get("MAX_ITERS", None)
if MAX_ITERS is None:
    MAX_ITERS = len(train_data) // tokens_por_iteracion
    _log(
        f"[MIDTRAIN] MAX_ITERS no fijado en config_annealing.py -- "
        f"calculado automaticamente a partir de train_midtrain.bin: "
        f"{len(train_data):,} tokens // {tokens_por_iteracion:,} tok/iter "
        f"= {MAX_ITERS:,} iteraciones (una pasada completa)."
    )


# ============================================================
# Batches + validacion fija/reproducible (identico al pretrain)
# ============================================================

_rng_val_setup = np.random.default_rng(VALIDATION_SEED)
FIXED_VAL_EVAL_POSITIONS = _rng_val_setup.integers(
    0, len(val_data) - BLOCK_SIZE,
    size=(EVAL_ITERS_VAL, BATCH_SIZE), dtype=np.int64,
)

_rng_train_eval = np.random.default_rng(TRAIN_EVAL_SEED)
FIXED_TRAIN_EVAL_POSITIONS = _rng_train_eval.integers(
    0, len(train_data) - BLOCK_SIZE,
    size=(EVAL_ITERS_TRAIN, BATCH_SIZE), dtype=np.int64,
)

_SEQ_OFFSETS = np.arange(BLOCK_SIZE + 1, dtype=np.int64)[None, :]


class MuestreadorSecuencial:
    """Copia exacta de la clase en train_pretrain_v2.py: recorre train_data
    en bloques NO solapados de BLOCK_SIZE tokens, en un orden barajado
    (permutacion determinista), con posicion PERSISTENTE (se guarda/
    restaura en el checkpoint de midtrain). Cobertura exacta del corpus
    de midtrain en cada epoca, sin huecos ni repeticion dentro de ella.
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
            self.epoca += 1
            self.posicion = 0
            self._generar_permutacion()
        idxs_bloque = self.permutacion[self.posicion:self.posicion + self.batch_size]
        self.posicion += self.batch_size
        return (idxs_bloque * self.block_size).astype(np.int64)

    def get_state(self):
        return {"seed_base": self.seed_base, "epoca": self.epoca,
                "posicion": self.posicion, "n_bloques": self.n_bloques,
                "n_tokens": self.n_bloques * self.block_size}

    @classmethod
    def from_state(cls, n_tokens, block_size, batch_size, state):
        n_bloques_actual = (n_tokens - 1) // block_size
        n_bloques_guardado = state.get("n_bloques")
        if n_bloques_guardado is None:
            _n_tokens_guardado = state.get("n_tokens")
            n_bloques_guardado = (_n_tokens_guardado // block_size
                                  if _n_tokens_guardado is not None else None)
        if n_bloques_guardado is not None and n_bloques_guardado != n_bloques_actual:
            _log(
                f"[MUESTREADOR] train_midtrain.bin ha cambiado de tamano "
                f"({n_bloques_guardado:,} -> {n_bloques_actual:,} bloques) desde "
                f"el ultimo checkpoint -- empezando epoca nueva."
            )
            return cls(n_tokens, block_size, batch_size, state["seed_base"])
        return cls(n_tokens, block_size, batch_size, state["seed_base"],
                    epoca=state["epoca"], posicion=state["posicion"])


muestreador_train = MuestreadorSecuencial(
    len(train_data), BLOCK_SIZE, BATCH_SIZE, TRAIN_RNG_SEED
)


def get_batch(split, *, positions=None, muestreador=None):
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
        raise ValueError("get_batch necesita positions o muestreador.")

    idx_matrix = ix[:, None] + _SEQ_OFFSETS
    batch = source[idx_matrix].astype(np.int64)

    x = torch.from_numpy(batch[:, :-1])
    y = torch.from_numpy(batch[:, 1:]).clone()

    # No aprender la transicion entre documentos: <|endofstory|> -> primer
    # token del siguiente. Ver train_pretrain_v2.py para el detalle.
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
    n_kv_head=globals().get("N_KV_HEAD", None),
    tie_weights=globals().get("TIE_WEIGHTS", True),
    rope_theta=ROPE_THETA,
    ffn_hidden=globals().get("FFN_HIDDEN", None),
).to(DEVICE)

num_params = sum(p.numel() for p in model.parameters())

num_params_no_emb = num_params - model.token_embedding_table.weight.numel()
FLOPS_POR_TOKEN = 6 * num_params_no_emb + 12 * N_LAYER * N_EMBD * BLOCK_SIZE

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR_PICO,
    betas=(BETA1, BETA2),
    weight_decay=WEIGHT_DECAY,
    fused=(DEVICE == "cuda"),
)

raw_model = model

TORCH_COMPILE_MODE = globals().get("TORCH_COMPILE_MODE", "default")
if DEVICE == "cuda":
    model = torch.compile(model, mode=TORCH_COMPILE_MODE)

scaler = torch.amp.GradScaler("cuda", enabled=USE_SCALER)


# ============================================================
# Scheduler: SOLO decay 1-sqrt, sin warmup, sin estado que persistir
# ============================================================
#
# Toda la vida de este script ES la fase de decay -- a diferencia del
# WSD del pretrain (que necesitaba recordar "en que iteracion se activo
# el annealing" porque convivia con una fase estable previa dentro del
# MISMO proceso), aqui it=0 siempre coincide con el inicio del decay. No
# hay nada que guardar en el checkpoint para esto: es una funcion pura.
#
#   f(it) = 1 - sqrt(it / MAX_ITERS)      1 -> 0
#   LR(it) = MIN_LR + f(it) * (LR_PICO - MIN_LR)
#
# Ver Hagele et al. 2024 (arxiv 2508.01483) / MiniCPM -- 1-sqrt superando
# a coseno y lineal en comparaciones controladas de forma de decay.

def lr_para_iteracion(it):
    if it >= MAX_ITERS:
        return MIN_LR
    frac = it / max(1, MAX_ITERS)
    factor = 1.0 - math.sqrt(frac)
    return MIN_LR + factor * (LR_PICO - MIN_LR)


checkpoint_config = {
    "version": "midtrain_bpe",
    "stage": "midtrain",
    "arch": "v2_llama",
    "vocab_size": vocab_size,
    "block_size": BLOCK_SIZE,
    "n_embd": N_EMBD,
    "n_head": N_HEAD,
    "n_kv_head": globals().get("N_KV_HEAD", N_HEAD),
    "n_layer": N_LAYER,
    "ffn_hidden": getattr(raw_model, "ffn_hidden", None),
    "rope_theta": ROPE_THETA,
    "rms_eps": getattr(raw_model, "rms_eps", 1e-5),
    "tie_weights": globals().get("TIE_WEIGHTS", True),
    "dropout": DROPOUT,
    "tokenizer": "bpe",
    "bpe_vocab_size": BPE_VOCAB_SIZE,
    "bpe_min_pair_freq": BPE_MIN_PAIR_FREQ,
    "init_from_checkpoint": INIT_FROM_CHECKPOINT,
    "lr_pico": LR_PICO,
    "min_lr": MIN_LR,
    "max_iters": MAX_ITERS,
}

CHECKPOINT_PATH = globals()["CHECKPOINT_PATH"]
best_checkpoint_path = globals()["BEST_CHECKPOINT_PATH"]
resume_checkpoint_path = CHECKPOINT_PATH.replace(".pt", "_resume.pt")

best_val_loss = float("inf")
patience_counter = 0
ultima_iteracion = 0
iteracion_inicial = 0


def guardar_modelo(path, iteracion, best_val_loss_actual, patience_actual=0):
    """Guarda el checkpoint de forma ATOMICA: escribe primero a un fichero
    temporal y solo cuando ha terminado del todo lo mueve sobre el nombre
    definitivo con os.replace() (operacion atomica del sistema de ficheros,
    tambien en Windows).

    Por que: torch.save() sobre el fichero definitivo tarda varios segundos
    con un modelo de 520M params + estado de AdamW. Si el proceso muere en
    ESE intervalo (corte de luz, OOM, Ctrl-C desafortunado), el .pt queda
    TRUNCADO y al relanzar da "EOFError: Ran out of input" -- te quedas sin
    checkpoint de reanudacion justo cuando mas falta hace. Comprobado en
    pruebas: matar el proceso durante el guardado deja el fichero
    inservible. Con os.replace(), o esta el checkpoint anterior intacto, o
    esta el nuevo completo; nunca un fichero a medias.
    """
    path_tmp = f"{path}.tmp"
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
            "rng_train_numpy": rng_train.bit_generator.state,
            "muestreador_train_state": muestreador_train.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
        },
        path_tmp,
    )
    os.replace(path_tmp, path)


# ---------------------------------------------------------------------------
# Bateria de evaluacion automatica al guardar checkpoint (identica al pretrain)
# ---------------------------------------------------------------------------

BATERIA_ACTIVA = globals().get("BATERIA_ACTIVA", False)
BATERIA_DOC = globals().get("BATERIA_DOC",
                            os.path.join(_dir_ckpt, "resultados_bateria_midtrain.csv"))
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
    """Identica a train_pretrain_v2.py -- ver ahi la explicacion completa."""
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
    """Identica a train_pretrain_v2.py -- ver ahi la explicacion completa."""
    if not BATERIA_ACTIVA or not BATERIA_COMANDOS:
        return

    import subprocess

    _log()
    _log(f"[BATERIA] Lanzando bateria de pruebas en iteracion {iteracion} "
          f"({len(BATERIA_COMANDOS)} prueba(s)) -- el entrenamiento queda "
          f"PAUSADO hasta que termine...")
    t0_bateria = time.time()

    ruta_informe_compacto = os.path.join(
        _dir_ckpt, f"informe_compacto_midtrain_it{iteracion}.txt"
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
    """Restaura TODO el estado de entrenamiento de midtrain (pesos,
    optimizer, scaler, iteracion, best_val_loss, patience, RNG,
    muestreador) desde un checkpoint de REANUDACION propio de midtrain --
    para continuar tras una caida a mitad de esta fase. Distinto de
    cargar_warm_start(), que solo se usa UNA VEZ, en el arranque en frio,
    para traer los pesos del pretrain."""
    _log(f"[REANUDAR] Encontrado checkpoint de reanudacion de midtrain: {path}")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    raw_model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    it_guardada = ckpt.get("iteracion", 0)
    bvl = ckpt.get("best_val_loss", float("inf"))
    pc = ckpt.get("patience_counter", 0)

    estado_np = ckpt.get("rng_train_numpy")
    if estado_np is not None:
        rng_train.bit_generator.state = estado_np

    global muestreador_train
    estado_muestreador = ckpt.get("muestreador_train_state")
    if estado_muestreador is not None:
        muestreador_train = MuestreadorSecuencial.from_state(
            len(train_data), BLOCK_SIZE, BATCH_SIZE, estado_muestreador
        )
        _log(f"[MUESTREADOR] Reanudado: epoca={muestreador_train.epoca} "
              f"posicion={muestreador_train.posicion}/{muestreador_train.n_bloques} bloques")

    if ckpt.get("rng_torch") is not None:
        rng_torch_cpu = ckpt["rng_torch"]
        if hasattr(rng_torch_cpu, "cpu"):
            rng_torch_cpu = rng_torch_cpu.cpu()
        torch.set_rng_state(rng_torch_cpu)
    if ckpt.get("rng_cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        except Exception:
            pass

    it_siguiente = it_guardada + 1
    _log(f"[REANUDAR] Reanudando midtrain en iteracion {it_siguiente} "
          f"(best_val_loss={bvl:.4f}, patience={pc})")

    return it_siguiente, bvl, pc


def cargar_warm_start(path):
    """Carga SOLO los pesos del modelo desde un checkpoint de pretrain (o
    de cualquier otra fase). Optimizer, scaler, iteracion, RNG y
    muestreador quedan en sus valores por defecto (arranque limpio) --
    a diferencia de cargar_para_reanudar(), que restaura TODO el estado.

    Como BLOCK_SIZE/ROPE_THETA cambian respecto al checkpoint de origen
    (contexto ampliado a 8K), los buffers de RoPE del checkpoint pueden
    tener OTRA FORMA (p.ej. [4096] vs [8192]) o valores calculados con el
    theta viejo.

    OJO: strict=False NO basta para esto. strict=False solo ignora claves
    ausentes/sobrantes; un desajuste de FORMA en una clave que existe en
    ambos lados hace que load_state_dict lance RuntimeError igualmente
    ("size mismatch for ..."). Por eso aqui se filtran a mano las claves
    incompatibles ANTES de cargar.

    Solo se permite descartar buffers de RoPE (nombres tipo
    *rope*/*freq*/*inv_freq*), que el modelo recalcula solo al
    construirse con el BLOCK_SIZE/ROPE_THETA nuevos. Si el filtro se
    topa con un desajuste de forma en un peso ENTRENABLE de verdad, se
    aborta con error: eso no seria una diferencia de contexto sino una
    arquitectura incompatible, y seguir adelante entrenaria medio modelo
    desde pesos aleatorios sin avisar.
    """
    _log(f"[WARM-START] Cargando SOLO pesos desde: {path}")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state_origen = ckpt["model_state_dict"]

    _PATRON_ROPE = re.compile(r"rope|freq", re.IGNORECASE)

    state_actual = raw_model.state_dict()
    state_filtrado = {}
    descartadas_rope = []
    incompatibles_graves = []

    for clave, tensor in state_origen.items():
        if clave in state_actual and tensor.shape != state_actual[clave].shape:
            if _PATRON_ROPE.search(clave):
                descartadas_rope.append(
                    f"{clave} {tuple(tensor.shape)}->{tuple(state_actual[clave].shape)}"
                )
            else:
                incompatibles_graves.append(
                    f"{clave} {tuple(tensor.shape)}->{tuple(state_actual[clave].shape)}"
                )
            continue
        state_filtrado[clave] = tensor

    if incompatibles_graves:
        raise RuntimeError(
            "[WARM-START] Desajuste de forma en pesos que NO son buffers de "
            "RoPE -- el checkpoint de origen no es compatible con esta "
            "arquitectura, y cargarlo dejaria parte del modelo con pesos "
            "aleatorios sin avisar. Revisa config_annealing.py contra el "
            f"checkpoint de origen. Claves: {incompatibles_graves}"
        )

    if descartadas_rope:
        _log(f"[WARM-START] Buffers de RoPE descartados por cambio de forma "
              f"(el modelo los recalcula con BLOCK_SIZE={BLOCK_SIZE}, "
              f"ROPE_THETA={ROPE_THETA}): {descartadas_rope}")

    resultado = raw_model.load_state_dict(state_filtrado, strict=False)

    for etiqueta, claves in (
        ("ausentes (se quedan con la init del modelo nuevo)", resultado.missing_keys),
        ("del origen ignoradas", resultado.unexpected_keys),
    ):
        if not claves:
            continue
        sospechosas = [k for k in claves if not _PATRON_ROPE.search(k)]
        if sospechosas:
            _log(f"[WARM-START] !!! ATENCION !!! Claves {etiqueta} que NO parecen "
                  f"buffers de RoPE -- revisa esto a mano antes de seguir: {sospechosas}")
        normales = [k for k in claves if _PATRON_ROPE.search(k)]
        if normales:
            _log(f"[WARM-START] Claves {etiqueta} (buffers de RoPE, esperado): {normales}")

    _log(f"[WARM-START] Pesos cargados desde iteracion {ckpt.get('iteracion', '?')} de origen "
          f"(best_val_loss origen: {ckpt.get('best_val_loss', float('nan')):.4f})")


# Prioridad: checkpoint de reanudacion PROPIO de midtrain (si ya se
# arranco antes y murio a mitad) > warm start desde el pretrain (solo en
# el arranque en frio, primera vez que se lanza este script).
#
# Si el checkpoint de reanudacion existe pero esta ILEGIBLE (truncado por
# una caida durante el guardado, en una corrida anterior a que
# guardar_modelo fuese atomico), NO se tumba el arranque: se avisa muy
# visiblemente y se cae al warm start. Perder el progreso del midtrain es
# malo, pero quedarse sin poder arrancar es peor -- y el aviso deja claro
# lo que ha pasado en vez de que parezca un arranque normal.
_reanudado = False
if RESUME and os.path.isfile(resume_checkpoint_path):
    try:
        iteracion_inicial, best_val_loss, patience_counter = cargar_para_reanudar(
            resume_checkpoint_path
        )
        _reanudado = True
    except Exception as e:  # noqa: BLE001
        _log()
        _log(f"[REANUDAR] !!! ATENCION !!! El checkpoint de reanudacion existe "
              f"pero NO se puede cargar ({type(e).__name__}: {e}).")
        _log(f"[REANUDAR] Lo mas probable: el proceso murio mientras se escribia "
              f"y quedo truncado. Se ignora y se arranca desde el warm start del "
              f"pretrain -- SE PIERDE el progreso de midtrain que hubiera.")
        _log(f"[REANUDAR] Si crees que ese progreso era valioso, PARA AHORA y "
              f"mira si {best_checkpoint_path} sirve como punto de partida.")
        _log()

if not _reanudado:
    if RESUME and not os.path.isfile(resume_checkpoint_path):
        _log("[REANUDAR] No hay checkpoint de reanudacion propio de midtrain "
              "todavia -- arranque en frio.")
    if not INIT_FROM_CHECKPOINT:
        raise ValueError(
            "INIT_FROM_CHECKPOINT no esta definido en config_annealing.py y no "
            "hay checkpoint de reanudacion de midtrain -- el midtrain no debe "
            "arrancar con pesos aleatorios. Define INIT_FROM_CHECKPOINT."
        )
    if not os.path.isfile(INIT_FROM_CHECKPOINT):
        raise FileNotFoundError(
            f"INIT_FROM_CHECKPOINT no existe: {INIT_FROM_CHECKPOINT}"
        )
    cargar_warm_start(INIT_FROM_CHECKPOINT)


# ============================================================
# Entrenamiento
# ============================================================

_log()
_log("Alfa1 MIDTRAIN iniciado.")
_log("Etapa: annealing/decay final (12BT: conocimiento/razonamiento/codigo)")
_log(f"Dispositivo: {DEVICE}")
_log(f"Corpus train: {TRAIN_BIN_PATH}")
_log(f"Train tokens: {len(train_data):,}")
_log(f"Val tokens: {len(val_data):,}")
_log(f"Tamano del vocabulario: {vocab_size}")
_log(f"Parametros del modelo: {num_params:,}")
_log(f"Batch size: {BATCH_SIZE}")
_log(f"Block size: {BLOCK_SIZE}")
_log(f"RoPE theta: {ROPE_THETA}")
_log(f"Tokens por iteracion: {tokens_por_iteracion:,}")
_log(f"Max iters: {MAX_ITERS:,}")
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
_log(f"Grad clip: {GRAD_CLIP}")
_log(f"Scheduler: decay 1-sqrt puro, sin warmup -- LR_PICO={LR_PICO:.6e} -> "
      f"MIN_LR={MIN_LR:.2e} en {MAX_ITERS:,} iteraciones")
_log(f"Checkpoint final: {CHECKPOINT_PATH}")
_log(f"Mejor checkpoint: {best_checkpoint_path}")

tokens_totales_midtrain = MAX_ITERS * tokens_por_iteracion
_log(f"Tokens totales de midtrain (MAX_ITERS x tok/iter): {tokens_totales_midtrain:,}")
_log()

tiempo_ciclo_inicio = time.time()

_log_es_nuevo = not os.path.isfile(LOSS_LOG_PATH)
os.makedirs(os.path.dirname(LOSS_LOG_PATH) or ".", exist_ok=True)
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

_ultimo_train_loss = None
_ultimo_val_loss = None
_grad_norm_ultimo = None

_barra = tqdm(
    range(iteracion_inicial, MAX_ITERS + 1),
    desc="Midtrain",
    unit="it",
    ncols=110,
)
for iteracion in _barra:
    ultima_iteracion = iteracion
    _t_iter_inicio = time.time()

    if iteracion % EVAL_INTERVAL == 0:
        losses = estimate_loss(model)

        train_loss = losses["train"]
        val_loss = losses["val"]
        train_sem = losses["train_sem"]
        val_sem = losses["val_sem"]
        val_ic95 = 1.96 * val_sem
        _ultimo_train_loss = train_loss
        _ultimo_val_loss = val_loss

        _lr_log = lr_para_iteracion(iteracion)
        _csv_loss_log.writerow([
            iteracion, f"{train_loss:.6f}", f"{train_sem:.6f}",
            f"{val_loss:.6f}", f"{val_sem:.6f}", f"{val_ic95:.6f}",
            f"{_lr_log:.8e}", VALIDATION_PROTOCOL,
            EVAL_ITERS_VAL * BATCH_SIZE * BLOCK_SIZE,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            f"{_grad_norm_ultimo:.4f}" if _grad_norm_ultimo is not None else "",
        ])
        _fh_loss_log.flush()

        tiempo_transcurrido = time.time() - tiempo_ciclo_inicio
        iters_en_ciclo = max(1, iteracion - getattr(estimate_loss, "_ultima_iter", iteracion_inicial))
        s_por_iter = tiempo_transcurrido / iters_en_ciclo
        iters_restantes = MAX_ITERS - iteracion
        eta_horas = (iters_restantes * s_por_iter) / 3600

        tok_s = tokens_por_iteracion / s_por_iter if s_por_iter > 0 else 0.0

        flops_iter = FLOPS_POR_TOKEN * tokens_por_iteracion
        tflops_s = (flops_iter / s_por_iter / 1e12) if s_por_iter > 0 else 0.0

        _log(
            f"Iteracion {iteracion}: "
            f"train loss {train_loss:.4f}, "
            f"val loss {val_loss:.4f} +/- {val_ic95:.4f} (IC95 aprox.), "
            f"lr {_lr_log:.4e}"
        )
        _log(
            f"  Tiempo ciclo: {tiempo_transcurrido:.1f}s "
            f"({s_por_iter:.2f} s/iter | {tok_s:,.0f} tok/s | {tflops_s:,.1f} TFLOPS/s) "
            f"| ETA: {eta_horas:.1f}h"
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
        lanzar_bateria(resume_checkpoint_path, iteracion,
                      train_loss=locals().get("train_loss"),
                      val_loss=locals().get("val_loss"))

    if iteracion == MAX_ITERS:
        break

    lr_actual = lr_para_iteracion(iteracion)
    for grupo in optimizer.param_groups:
        grupo["lr"] = lr_actual

    optimizer.zero_grad(set_to_none=True)

    for micro_step in range(GRAD_ACCUM_STEPS):
        xb, yb = get_batch("train", muestreador=muestreador_train)

        if (DEVICE == "cuda" and hasattr(torch, "compiler")
                and hasattr(torch.compiler, "cudagraph_mark_step_begin")):
            torch.compiler.cudagraph_mark_step_begin()

        with torch.amp.autocast(
            device_type=AMP_DEVICE_TYPE,
            dtype=AMP_DTYPE,
            enabled=USE_AMP,
        ):
            _, loss, _ = model(xb, yb)
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

    if GRAD_CLIP is not None and GRAD_CLIP > 0:
        scaler.unscale_(optimizer)
        _grad_norm_ultimo = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        )

    scaler.step(optimizer)
    scaler.update()

    _t_iter = time.time() - _t_iter_inicio
    _tok_s_inst = tokens_por_iteracion / _t_iter if _t_iter > 0 else 0.0
    _decay_pct = min(100.0, 100.0 * iteracion / max(1, MAX_ITERS))
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
