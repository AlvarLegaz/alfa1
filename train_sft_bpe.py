# train_sft_bpe.py
"""
Full SFT (fine-tuning completo, sin LoRA) de Alpha1, partiendo del
checkpoint de pretraining (SFT_INIT_FROM en config_sft.py).

REQUISITO: los .txt de SFT_DATA_DIR ya deben estar formateados como
<USER>...<ASSISTANT>...<FIN>. No hay separador de documento aparte: cada
<FIN> marca el final de un ejemplo, asi que el fichero se trocea por
<FIN> (cada trozo se queda con su propio <FIN> al final). Mismo formato
que usa sft_lora_alpha1.py -- este script solo tokeniza, no formatea.

--------------------------------------------------------------------

Uso:
    python train_sft_bpe.py
"""

import os
import gc
import hashlib
import json
import math
import random
import time

from tqdm import tqdm

import torch
import torch.nn.functional as F

from alfa1.config_sft import *
from alfa1.model import SanchoGPT
from alfa1.tokenizer_bpe import BPETokenizer


# ============================================================
# Ajustes por defecto (mismo patron que train_pretrain_bpe.py)
# ============================================================

EARLY_STOPPING = globals().get("EARLY_STOPPING", True)
PATIENCE = globals().get("PATIENCE", 5)
MIN_DELTA = globals().get("MIN_DELTA", 0.001)
GRAD_CLIP = globals().get("GRAD_CLIP", 1.0)
SEED = globals().get("SEED", 1337)
GRAD_ACCUM_STEPS = globals().get("GRAD_ACCUM_STEPS", 1)
MASK_PROMPT_LOSS = globals().get("MASK_PROMPT_LOSS", True)

torch.manual_seed(SEED)
random.seed(SEED)

torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
# Tokenizador (el MISMO que en pretraining, congelado -- no se reentrena)
# ============================================================

if not os.path.isfile(TOKENIZER_STATE_PATH):
    raise FileNotFoundError(f"No existe {TOKENIZER_STATE_PATH}")

tokenizer = BPETokenizer()
with open(TOKENIZER_STATE_PATH, "r", encoding="utf-8") as f:
    tokenizer.load_state(json.load(f))

vocab_size = tokenizer.vocab_size()

user_id = tokenizer.encode(USER_TAG)[0]        # asume 1 token por tag especial
assistant_id = tokenizer.encode(ASSISTANT_TAG)[0]
fin_id = tokenizer.encode(FIN_TAG)[0]
pad_id = tokenizer.encode(PAD_TAG)[0]
# v39: <turno> marca el resultado de herramienta entrante -- se enmascara
# igual que <USER> (ver comentario en config_sft.py). No existe en corpus
# anteriores a v39, asi que esto no cambia el comportamiento sobre ellos.
turn_id = tokenizer.encode(TURN_TAG)[0]


# ============================================================
# Datos: leer los .txt YA FORMATEADOS, tokenizar
# ============================================================

def _trocear_por_fin(contenido: str):
    """Divide el texto por <FIN>, devolviendo cada trozo CON su <FIN> al
    final (es el terminador del ejemplo, no un separador a descartar).
    El ultimo trozo tras el <FIN> final suele quedar vacio/residual y se
    descarta."""
    partes = contenido.split(FIN_TAG)
    documentos = []
    for parte in partes[:-1]:
        parte = parte.strip()
        if parte:
            documentos.append(parte + FIN_TAG)
    return documentos


def cargar_ejemplos(rutas):
    if not rutas:
        raise FileNotFoundError(f"No se encontraron .txt en {SFT_DATA_DIR}")

    ejemplos = []
    for ruta in rutas:
        with open(ruta, encoding="utf-8") as f:
            contenido = f.read()

        documentos = _trocear_por_fin(contenido)
        n_fichero = 0
        for doc in documentos:
            # normalize_chat=False es OBLIGATORIO aqui: normalize_chat_text()
            # hace raw_line.strip() en CADA linea del documento, y estos .txt
            # ya vienen formateados como <USER>...<ASSISTANT>...<FIN> (no como
            # "Usuario:"/"Sancho:", que es lo unico que esa funcion normaliza).
            # Con el default (normalize_chat=True) se estaba destruyendo toda
            # la indentacion de las respuestas de codigo del corpus -- 5.500
            # ejemplos (7.6%) perdian sus 4 espacios de sangria y quedaban
            # como "if x:\nreturn 1" en vez de "if x:\n    return 1". Confirmado
            # en auditoria 2026-09-05: con el bug, 0/40 respuestas de codigo
            # generadas por el checkpoint v9 compilaban (la mayoria por
            # "expected an indented block"), lo que probablemente explica que
            # dominio_programacion llevara 9 experimentos estancado en 6-8/16.
            ids = tokenizer.encode(doc, normalize_chat=False)
            if len(ids) < 2 or len(ids) > BLOCK_SIZE + 1:
                continue  # descarta vacios y los que no caben en BLOCK_SIZE
            ejemplos.append((doc, ids))   # se guarda el texto crudo para _clave_estable
            n_fichero += 1

        print(f"  {ruta}: {n_fichero} ejemplos de {len(documentos)} trozos encontrados")

    return ejemplos


def _clave_estable(doc_text):
    """Clave de ordenacion determinista basada en el TEXTO CRUDO del
    ejemplo (el .txt tal cual, antes de tokenizar) y en SEED.

    CRITICO: debe hashear el texto crudo, NO `ids` (el bug real detectado en
    la auditoria 2026-09-05). Hashear `ids` liga el orden de entrenamiento al
    tokenizador: el fix de indentacion de este mismo commit (normalize_chat=
    False) no cambia ni un caracter del corpus, solo COMO se tokeniza, pero
    al hashear ids cambiaba el hash de ~14.8% de los ejemplos -- lo bastante
    para que el 98.5% de las posiciones de train_ejemplos apuntaran a un
    documento distinto entre Exp.9 y Exp.10 (verificado empiricamente), y que
    random.sample() en get_batch() sirviera una secuencia de entrenamiento
    COMPLETAMENTE distinta desde el primer microbatch. Con el texto crudo
    como clave, un cambio puramente de tokenizacion no mueve nada: solo un
    cambio real del contenido del .txt reordena su propio ejemplo.

    Por que no random.shuffle: shuffle consume la secuencia aleatoria en
    funcion del TAMANO de la lista, asi que al anadir o quitar ejemplos
    cambia la posicion de TODOS los demas, no solo la de los nuevos. En un
    fine-tuning completo el orden de los batches afecta a la trayectoria,
    de modo que dos corpus casi identicos (p.ej. Exp.4 70.812 vs Exp.5
    71.008 ejemplos, un 0,3% de diferencia) producian ordenes de
    entrenamiento completamente distintos -- lo que hacia imposible saber
    si un cambio de puntuacion venia del dato nuevo o del reordenamiento.
    Con esta clave, anadir datos intercala los nuevos sin mover los viejos.
    """
    return hashlib.sha256(f"{SEED}|{doc_text}".encode()).hexdigest()


print("Tokenizando corpus SFT (.txt ya formateados)...")
train_files = sorted(glob.glob(os.path.join(SFT_DATA_DIR, "train", "*.txt")))
val_files = sorted(glob.glob(os.path.join(SFT_DATA_DIR, "val", "*.txt")))
if train_files or val_files:
    if not train_files or not val_files:
        raise FileNotFoundError("La release debe contener ambos directorios: train/ y val/")
    train_pairs = cargar_ejemplos(train_files)
    val_pairs = cargar_ejemplos(val_files)
    train_texts = {doc for doc, _ in train_pairs}
    val_texts = {doc for doc, _ in val_pairs}
    overlap = train_texts & val_texts
    if overlap:
        raise ValueError(f"Fuga de datos: {len(overlap)} documentos exactos aparecen en train y val")
    train_pairs.sort(key=lambda par: _clave_estable(par[0]))
    val_pairs.sort(key=lambda par: _clave_estable(par[0]))
    train_ejemplos = [ids for _, ids in train_pairs]
    val_ejemplos = [ids for _, ids in val_pairs]
    todos_los_ejemplos = train_ejemplos + val_ejemplos
    print("Split predefinido cargado; no se vuelve a dividir ni a mezclar")
else:
    print("AVISO: release legacy sin train/val; se aplica TRAIN_RATIO")
    todos_los_pares = cargar_ejemplos(SFT_DATA_FILES)
    todos_los_pares.sort(key=lambda par: _clave_estable(par[0]))
    todos_los_ejemplos = [ids for _, ids in todos_los_pares]
    corte = int(len(todos_los_ejemplos) * TRAIN_RATIO)
    train_ejemplos = todos_los_ejemplos[:corte]
    val_ejemplos = todos_los_ejemplos[corte:]

print(f"Ejemplos SFT: {len(todos_los_ejemplos)} totales "
      f"({len(train_ejemplos)} train / {len(val_ejemplos)} val)")

if not train_ejemplos or not val_ejemplos:
    raise ValueError("No hay suficientes ejemplos para train/val. Revisa SFT_DATA_DIR.")


# ============================================================
# Batches: shift causal + padding + mascara de loss del prompt
# ============================================================

def _labels_enmascaradas(ids):
    """labels = ids[1:], con -100 desde <USER> hasta el siguiente
    <ASSISTANT> (inclusive) en cada turno, si MASK_PROMPT_LOSS.

    OJO: hay que recorrer `ids` completo (no `ids[1:]`), porque <USER> es
    siempre el primer token de cada ejemplo (posicion 0) -- si el bucle
    itera sobre ids[1:], la posicion 0 nunca se ve y el enmascarado no se
    activa NUNCA, dejando la loss sobre la secuencia entera (prompt
    incluido) en vez de solo sobre la respuesta. Confirmado tokenizando
    ejemplos reales: con el bucle antiguo, 0 tokens enmascarados siempre.
    """
    labels = list(ids[1:])
    if not MASK_PROMPT_LOSS:
        return labels
    dentro_de_prompt = False
    for i, tok_id in enumerate(ids):
        if tok_id == user_id or tok_id == turn_id:
            dentro_de_prompt = True
        if i > 0 and dentro_de_prompt:
            labels[i - 1] = -100
        if tok_id == assistant_id:
            dentro_de_prompt = False
    return labels


# Cursor ciclico por split: NO usar random.sample(fuente, k) aqui.
#
# Auditoria 2026-09-05, en dos pasos:
#   1) random.sample() consume el stream de random.random() global y
#      selecciona por INDICE dentro de `fuente`. Si el CONTENIDO en un
#      indice dado cambia (aunque la longitud de la lista sea identica),
#      random.sample() con la misma semilla elige los mismos indices pero
#      devuelve documentos distintos. Confirmado empiricamente entre Exp.9
#      y Exp.10 (mismo corpus, solo cambio el tokenizador): random.sample()
#      servia una secuencia de entrenamiento COMPLETAMENTE distinta desde
#      el microbatch #0 (0.1% de coincidencia en 2000 microbatches).
#   2) La causa de que el CONTENIDO por indice cambiara pese al "shuffle
#      estable" era que _clave_estable() hasheaba `ids` (ya tokenizados),
#      no el texto crudo -- asi que cualquier cambio de tokenizador (como
#      el fix de indentacion de este mismo commit, que no toca ni un
#      caracter del corpus) recalculaba el hash del 14.8% de los ejemplos y
#      los reintercalaba por toda la lista, desplazando el indice absoluto
#      del 98.5% de los DEMAS ejemplos (verificado). _clave_estable() ahora
#      hashea el texto crudo del .txt (ver cargar_ejemplos), asi que un
#      cambio puro de tokenizacion ya no mueve nada.
#
# Con ambos fixes (clave sobre texto crudo + cursor ciclico en vez de
# random.sample), un cambio puro de tokenizacion produce el 100% de
# microbatches identicos (2000/2000, verificado); un cambio real de
# contenido solo afecta a los batches cercanos a los documentos que de
# verdad cambiaron, no a la secuencia entera de entrenamiento.
_cursor = {"train": 0, "val": 0}


def get_batch(split):
    fuente = train_ejemplos if split == "train" else val_ejemplos
    n = len(fuente)
    k = min(BATCH_SIZE, n)
    inicio = _cursor[split]
    muestra = [fuente[(inicio + i) % n] for i in range(k)]
    _cursor[split] = (inicio + k) % n

    x_batch, y_unmasked_batch, y_masked_batch = [], [], []
    for ids in muestra:
        x_batch.append(ids[:-1])
        y_unmasked_batch.append(ids[1:])
        y_masked_batch.append(_labels_enmascaradas(ids))

    max_len = max(len(s) for s in x_batch)

    x = torch.full((len(x_batch), max_len), pad_id, dtype=torch.long)
    y_unmasked = torch.full((len(x_batch), max_len), pad_id, dtype=torch.long)
    y_masked = torch.full((len(x_batch), max_len), -100, dtype=torch.long)

    for i in range(len(x_batch)):
        L = len(x_batch[i])
        x[i, :L] = torch.tensor(x_batch[i], dtype=torch.long)
        y_unmasked[i, :L] = torch.tensor(y_unmasked_batch[i], dtype=torch.long)
        y_masked[i, :L] = torch.tensor(y_masked_batch[i], dtype=torch.long)

    if DEVICE == "cuda":
        return (x.pin_memory().to(DEVICE, non_blocking=True),
                y_unmasked.pin_memory().to(DEVICE, non_blocking=True),
                y_masked.pin_memory().to(DEVICE, non_blocking=True))
    return x.to(DEVICE), y_unmasked.to(DEVICE), y_masked.to(DEVICE)


@torch.no_grad()
def estimate_loss(model):
    """OJO (fix 2026-09-06): 'val' se evalua SIEMPRE sobre el split de
    validacion COMPLETO, no sobre una muestra.

    Bug real detectado: con EVAL_ITERS=50 y BATCH_SIZE=2, cada llamada a
    estimate_loss() solo evaluaba 100 ejemplos de un val set de ~6-7k
    (~1.5%). Ademas _cursor["val"] es un cursor CICLICO que nunca se
    reinicia, asi que cada eval veia un TROZO DISTINTO y no solapado del
    val set (hacen falta ~68 evals para dar una vuelta completa). Esto
    explicaba gran parte del "ruido" que parecia mejora/regresion real
    entre puntos de la curva de val loss (ej. Exp.14: 1.19 -> 1.26 -> 1.03
    -> 1.25 en evals consecutivos) -- no era el modelo empeorando y
    mejorando, era medir sobre muestras distintas de dificultad distinta.
    Grave porque EARLY_STOPPING y "mejor checkpoint" se decidian con ese
    numero ruidoso: el checkpoint guardado como _best podia ser simplemente
    el que caia en un trozo facil, no el mejor modelo real.

    Fix: resetear el cursor de 'val' a 0 antes de cada evaluacion (mismo
    punto de partida siempre) y recorrer el split ENTERO, no una muestra --
    asi el numero es reproducible y comparable de verdad entre evals. El
    split 'train' se deja con la muestra de EVAL_ITERS (mas barato, y no
    se usa para elegir el checkpoint, solo como referencia de overfitting).
    """
    out = {}
    model.eval()
    for split in ["train", "val"]:
        if split == "val":
            _cursor["val"] = 0
            n_iters = max(1, -(-len(val_ejemplos) // BATCH_SIZE))  # ceil(n/batch)
        else:
            n_iters = EVAL_ITERS
        losses = torch.zeros(n_iters)
        for k in range(n_iters):
            x, y_unmasked, y_masked = get_batch(split)
            with torch.amp.autocast(device_type=AMP_DEVICE_TYPE, dtype=AMP_DTYPE, enabled=USE_AMP):
                logits, _, _ = model(x, y_unmasked)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y_masked.reshape(-1),
                    ignore_index=-100,
                )
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ============================================================
# Modelo: misma arquitectura que el pretraining, pesos cargados del
# checkpoint de pretraining (SFT_INIT_FROM)
# ============================================================

model = SanchoGPT(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER,
    dropout=0.0,          # sin dropout en SFT sobre un checkpoint ya preentrenado
    device=DEVICE,
    n_kv_head=N_KV_HEAD,
    tie_weights=TIE_WEIGHTS,
    rope_theta=ROPE_THETA,
    ffn_hidden=FFN_HIDDEN,
).to(DEVICE)

if not os.path.isfile(SFT_INIT_FROM):
    raise FileNotFoundError(f"No existe el checkpoint de pretraining: {SFT_INIT_FROM}")

print(f"Cargando checkpoint de pretraining: {SFT_INIT_FROM}")
ckpt_pretrain = torch.load(SFT_INIT_FROM, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt_pretrain["model_state_dict"])

# torch.compile: 1.5-2x en Blackwell (RTX 5090) con bf16. El primer paso
# tarda mas (compila), luego acelera el resto. Si diera problemas de
# compilacion, poner COMPILE_MODEL = False en config_sft.py.
if globals().get("COMPILE_MODEL", True) and DEVICE == "cuda":
    print("Compilando el modelo con torch.compile (el primer paso sera lento)...")
    model = torch.compile(model)

num_params = sum(p.numel() for p in model.parameters())

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY,
)
scaler = torch.amp.GradScaler("cuda", enabled=USE_SCALER)


# ============================================================
# Scheduler: warmup lineal + cosine decay hasta MIN_LR en EPOCHS
# ============================================================

iters_por_epoca = max(1, len(train_ejemplos) // (BATCH_SIZE * GRAD_ACCUM_STEPS))
MAX_ITERS = iters_por_epoca * EPOCHS


def lr_para_iteracion(it):
    if it < WARMUP_ITERS:
        return LEARNING_RATE * (it + 1) / WARMUP_ITERS
    if it >= MAX_ITERS:
        return MIN_LR
    ratio = (it - WARMUP_ITERS) / max(1, (MAX_ITERS - WARMUP_ITERS))
    coef = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return MIN_LR + coef * (LEARNING_RATE - MIN_LR)


# ============================================================
# Checkpointing (mismo formato que train_pretrain_bpe.py + stage="sft")
# ============================================================

checkpoint_config = {
    "version": "sft_bpe_full",
    "stage": "sft",
    "arch": "v2_llama",
    "vocab_size": vocab_size,
    "block_size": BLOCK_SIZE,
    "n_embd": N_EMBD,
    "n_head": N_HEAD,
    "n_kv_head": N_KV_HEAD,
    "n_layer": N_LAYER,
    "ffn_hidden": FFN_HIDDEN,
    "rope_theta": ROPE_THETA,
    "tie_weights": TIE_WEIGHTS,
    "dropout": 0.0,  # faltaba: lm_eval_adaptador.py/benchmark_bcc_es.py lo leen sin fallback
    "mask_prompt_loss": MASK_PROMPT_LOSS,
    "sft_data_dir": SFT_DATA_DIR,
    "init_from": SFT_INIT_FROM,
    "seed": SEED,
    "early_stopping": EARLY_STOPPING,
    "patience": PATIENCE,
    "min_delta": MIN_DELTA,
    "grad_clip": GRAD_CLIP,
}

best_val_loss = float("inf")
best_checkpoint_path = CHECKPOINT_PATH.replace(".pt", "_best.pt")
patience_counter = 0


def guardar_modelo(path, iteracion, best_val_loss_actual, patience_actual=0):
    # Copiar los pesos a CPU antes de serializar: evita el pico de RAM que
    # deja torch.save al copiar tensores GPU->RAM, y libera explicitamente
    # despues. (Los escalones de RAM en cada guardado venian de aqui.)
    #
    # OJO: con COMPILE_MODEL=True, `model` es el wrapper de torch.compile y
    # su state_dict() prefija todas las claves con "_orig_mod." -- eso rompe
    # la carga en lm_eval_adaptador.py/benchmark_bcc_es.py (esperan las
    # claves "limpias", como las guarda el checkpoint de pretrain sin
    # compile). Se quita el prefijo aqui para que el checkpoint sea cargable
    # igual este o no compilado. Confirmado real: las 165 claves del primer
    # checkpoint SFT salieron con el prefijo y hubo que parchearlas a mano.
    cpu_model_state = {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v.detach().cpu()
        for k, v in model.state_dict().items()
    }
    cpu_optim_state = optimizer.state_dict()

    torch.save(
        {
            "model_state_dict": cpu_model_state,
            "optimizer_state_dict": cpu_optim_state,
            "scaler_state_dict": scaler.state_dict(),
            "tokenizer_state": tokenizer.get_state(),
            "config": checkpoint_config,
            "iteracion": iteracion,
            "best_val_loss": best_val_loss_actual,
            "patience_counter": patience_actual,
        },
        path,
    )

    del cpu_model_state, cpu_optim_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Entrenamiento
# ============================================================

print()
print("ProyectoSancho iniciado.")
print("Etapa: SFT completo (full fine-tuning, sin LoRA)")
print(f"Dispositivo: {DEVICE}")
print(f"Parametros del modelo: {num_params:,}")
print(f"Batch size: {BATCH_SIZE} | Grad accum: {GRAD_ACCUM_STEPS}")
print(f"Iteraciones por epoca: {iters_por_epoca} | Epochs: {EPOCHS} | Max iters: {MAX_ITERS}")
print(f"Mascara de prompt en la loss: {MASK_PROMPT_LOSS}")
print(f"Checkpoint final: {CHECKPOINT_PATH}")
print(f"Mejor checkpoint: {best_checkpoint_path}")
print()

tiempo_ciclo_inicio = time.time()

# Acumuladores para el indicador de tokens/s (se reinician en cada eval)
tokens_desde_eval = 0
t_tokens_inicio = time.time()

barra = tqdm(range(MAX_ITERS + 1), desc="SFT", unit="it", dynamic_ncols=True)
for iteracion in barra:

    if iteracion % EVAL_INTERVAL == 0:
        losses = estimate_loss(model)
        train_loss, val_loss = losses["train"], losses["val"]

        dt = time.time() - tiempo_ciclo_inicio
        # tokens/s reales procesados en entrenamiento desde el ultimo eval
        dt_tok = time.time() - t_tokens_inicio
        tok_s = tokens_desde_eval / dt_tok if dt_tok > 0 else 0.0
        barra.write(
            f"Iteracion {iteracion}/{MAX_ITERS}: train loss {train_loss:.4f}, "
            f"val loss {val_loss:.4f} | {tok_s:,.0f} tok/s ({dt:.1f}s desde el ultimo eval)"
        )
        tiempo_ciclo_inicio = time.time()
        tokens_desde_eval = 0
        t_tokens_inicio = time.time()

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            guardar_modelo(best_checkpoint_path, iteracion, best_val_loss, patience_counter)
            barra.write(f"  Nuevo mejor modelo guardado: {best_checkpoint_path}")
        else:
            patience_counter += 1
            barra.write(f"  Sin mejora suficiente ({patience_counter}/{PATIENCE})")
            if EARLY_STOPPING and patience_counter >= PATIENCE:
                barra.write("Early stopping activado.")
                break

    if iteracion == MAX_ITERS:
        break

    lr_actual = lr_para_iteracion(iteracion)
    for grupo in optimizer.param_groups:
        grupo["lr"] = lr_actual

    optimizer.zero_grad(set_to_none=True)

    for micro_step in range(GRAD_ACCUM_STEPS):
        x, y_unmasked, y_masked = get_batch("train")
        tokens_desde_eval += x.numel()   # tokens realmente procesados (incluye padding)

        with torch.amp.autocast(device_type=AMP_DEVICE_TYPE, dtype=AMP_DTYPE, enabled=USE_AMP):
            logits, _, _ = model(x, y_unmasked)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y_masked.reshape(-1),
                ignore_index=-100,
            )
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

    # tok/s en vivo en la barra (media movil desde el ultimo eval)
    dt_vivo = time.time() - t_tokens_inicio
    if dt_vivo > 0:
        barra.set_postfix(tok_s=f"{tokens_desde_eval / dt_vivo:,.0f}", lr=f"{lr_actual:.1e}")

    if GRAD_CLIP is not None and GRAD_CLIP > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

    scaler.step(optimizer)
    scaler.update()

guardar_modelo(CHECKPOINT_PATH, iteracion, best_val_loss, patience_counter)
print()
print(f"Checkpoint final guardado: {CHECKPOINT_PATH}")
print(f"Mejor checkpoint: {best_checkpoint_path}")
print(f"Mejor val loss: {best_val_loss:.4f}")
