#!/usr/bin/env python3
"""
benchmark_bcc_es.py
-------------------
BCC-ES — Banco de Conocimiento y Competencias en Español.

Este script es el banco de pruebas para evaluar un checkpoint de modelo de
lenguaje en español bajo BCC-ES: un conjunto de bloques de evaluación
propios (no un único banco de preguntas, sino varios componentes
complementarios) pensado para medir, de forma objetiva y reproducible,
tanto la calidad de generación de texto como el conocimiento/razonamiento
del modelo en distintos dominios.

Componentes de BCC-ES evaluados por este script:
  1. ESPAÑOL       -- generación libre sobre un conjunto fijo de prompts
                       (narrativa, noticias, instructivo, receta, técnico...),
                       midiendo perplejidad, bits/char y tokens/segundo.
  2. INGLÉS        -- lo mismo que (1), pero en inglés, para medir el idioma
                       secundario del corpus.
  3. PROGRAMACIÓN  -- generación libre de código (Python y JSON, por
                       separado) a partir de fragmentos incompletos.
  4. BCC-ES        -- 360+ preguntas de elección múltiple (4 opciones) en 13
                       categorías de cultura general/dominio, evaluadas por
                       likelihood (estilo lm-eval-harness: gana la opción
                       con mayor log-prob). Incluye 'fisica_quimica' (160
                       ítems de 1º de Bachillerato, convertidos a tipo test
                       a partir de un banco con solución verificada a mano;
                       distractores generados programáticamente por
                       perturbación del resultado numérico -- no revisados
                       ítem a ítem como el resto de categorías, conviene
                       hacer una comprobación manual de una muestra).
  5. LIKELIHOOD    -- entity tracking, razonamiento, código, sentido común,
                       comprensión lectora, concordancia gramatical,
                       correferencia y causalidad (COMPETENCIAS de BCC-ES),
                       evaluados por elección forzada (log-prob de la
                       continuación correcta vs. incorrecta), para medir
                       capacidades del modelo base sin SFT.

Todos los bloques van activos por defecto salvo la generación en español,
que es obligatoria (los demás se pueden desactivar con --sin-ingles,
--sin-codigo, --sin-bcc-es, --sin-likelihood).

Pensado para uso DIDÁCTICO: mide y tabula
  - tiempo de carga del checkpoint
  - tiempo de generación por prompt
  - tokens/segundo
  - perplejidad del modelo sobre cada prompt (medida objetiva, no subjetiva)
  - la continuación/respuesta generada (para comparar a ojo)

También admite comparar varios checkpoints entre sí en la misma ejecución
(por ejemplo, dos puntos distintos del mismo entrenamiento, o incluso
arquitecturas distintas -- ver --model-modulo/--model-clase más abajo),
pero esa comparativa es un uso adicional, no la finalidad principal: lo
normal es lanzarlo sobre un único checkpoint para ver dónde está BCC-ES
en ese momento del entrenamiento.

IMPORTANTE sobre el tipo de modelo:
Un checkpoint de PRETRAINING no conversa: continúa texto. Por eso, en modo
'pretrain' (por defecto) el script trata cada prompt como SEMILLA y muestra
cómo el modelo lo continúa. La batería BCC-ES (tipo test, elección
múltiple) SÍ funciona bien con un modelo base pese a esto: no se le pide
generar texto, se mide por likelihood cuál de las 4 opciones el modelo
considera más probable como continuación, que es exactamente lo que un
modelo base sabe hacer.
En modo 'chat' (para cuando tengas el checkpoint SFT) envuelve el prompt con
el formato "Usuario: ... \nSancho:" como en tu chat_bpe.py.

Uso:
    # Lo más simple: ejecútalo directamente desde la carpeta del checkpoint,
    # sin ningún flag. Autodetecta el .pt (prioriza *_best.pt); todos los
    # bloques (español, inglés, código, BCC-ES, likelihood) van activos por
    # defecto y no requieren ningún fichero externo -- están embebidos en
    # el propio script:
    python benchmark_bcc_es.py

    # Indicando el checkpoint explícitamente:
    python benchmark_bcc_es.py --checkpoints checkpoint.pt

    # Cuando tengas SFT:
    python benchmark_bcc_es.py --checkpoints checkpoint_sft.pt --modo chat

    # Desactivar bloques que no necesites (todos activos por defecto):
    python benchmark_bcc_es.py --sin-ingles --sin-codigo --sin-bcc-es --sin-likelihood

Nota sobre checkpoint/arquitectura: --checkpoints es la ruta al fichero
.pt; la arquitectura (n_embd, n_head, n_layer, vocab...) NO se pide
aparte, se lee directamente del dict 'config' guardado dentro del
propio checkpoint (ver cargar_modelo()). Esto evita tener que repetir
a mano los hiperparámetros y previene errores de mismatch.

Requiere que este script vea tu paquete 'modelo' (ejecútalo desde la raíz del
proyecto, o ajusta PYTHONPATH).
"""

import argparse
import hashlib
import json
import re
import sys
import time
import math
from pathlib import Path

# Windows redirige stdout al codepage por defecto (cp1252) cuando no hay una
# consola UTF-8 real detras (p.ej. salida capturada a fichero en segundo
# plano) -- el caracter "⏎" usado mas abajo para mostrar saltos de linea
# revienta con UnicodeEncodeError en ese caso, matando el benchmark a mitad
# sin llegar a escribir ningun informe. Forzar UTF-8 en stdout/stderr es
# seguro en cualquier entorno (consola real o redirigido).
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch

def cargar_modelo(path, device, model_modulo, model_clase, tok_modulo, tok_clase):
    """Carga un checkpoint de forma genérica, importando dinámicamente la
    clase de modelo y de tokenizador indicadas. Esto permite evaluar en la
    misma tanda checkpoints de arquitecturas distintas sin tener nada 
    hardcodeado: cada checkpoint especifica su propio módulo/clase vía
    --model-modulo/--model-clase/--tokenizer-modulo/--tokenizer-clase.

    Los hiperparámetros de arquitectura (n_embd, n_head, n_layer...) se
    toman siempre del propio 'config' guardado en el checkpoint, nunca a
    mano, para evitar mismatches. Si tu clase de modelo tiene un
    constructor con parámetros distintos a los de abajo, ajusta el dict
    kwargs_extra según lo que necesite (ver comentario más abajo)."""
    import importlib

    mod_modelo = importlib.import_module(model_modulo)
    ClaseModelo = getattr(mod_modelo, model_clase)
    mod_tok = importlib.import_module(tok_modulo)
    ClaseTokenizer = getattr(mod_tok, tok_clase)

    # --- Camino alternativo para modelos HuggingFace (p.ej. Qwen2.5-0.5B) ---
    # Se detecta por duck typing: si la clase de modelo expone
    # 'desde_huggingface', 'path' se trata como model_id/ruta HF en vez de
    # como fichero .pt propio. Ver adaptador_hf.py.
    if hasattr(ClaseModelo, "desde_huggingface"):
        model = ClaseModelo.desde_huggingface(path, device)
        tokenizer = ClaseTokenizer(path)
        config = model.config_benchmark
        return model, tokenizer, config, {}

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    tokenizer = ClaseTokenizer()
    tokenizer.load_state(checkpoint["tokenizer_state"])

    # Kwargs comunes a ambas arquitecturas conocidas (v1 SanchoGPT clásico,
    # Alpha1 GQA/RoPE). Si tu clase de Alpha1 espera nombres distintos
    # (p.ej. 'n_kv_heads' en plural, o 'rope_theta'), añádelos aquí con
    # config.get(...) y un fallback razonable, en vez de tocar el resto
    # del script.
    kwargs = dict(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        dropout=config.get("dropout", 0.0),  # checkpoints SFT no guardan dropout (siempre 0.0 en eval)
        device=device,
        n_kv_head=config.get("n_kv_head", config["n_head"]),
        tie_weights=config.get("tie_weights", True),
    )
    model = ClaseModelo(**kwargs)
    # Defensivo: si el checkpoint se guardo desde un modelo con torch.compile
    # sin quitar el prefijo "_orig_mod.", las claves no casarian.
    sd = checkpoint["model_state_dict"]
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device)
    model.to(torch.bfloat16)
    model.eval()
    return model, tokenizer, config, checkpoint


def formatear_prompt(prompt, modo):
    """En chat, envuelve como en chat_bpe.py. En pretrain, el prompt es la semilla cruda."""
    if modo == "chat":
        return f"Usuario: {prompt}\nSancho:"
    return prompt


def perplejidad(model, tokenizer, texto, device):
    """Perplejidad y bits/char del modelo sobre 'texto'.

    - Perplejidad: exp(loss medio por token). Más baja = modela mejor, PERO
      depende del tokenizer: un vocab pequeño da ppl menor solo por granularidad.
    - bits/char: loss total convertida a bits y dividida entre caracteres. Es
      INVARIANTE al tokenizer, asi que permite comparar modelos con vocabs
      distintos de forma justa. Más baja = mejor.
    Reaprovecha la loss (media por token, en nats) que devuelve el modelo.
    """
    ids = tokenizer.encode(texto, normalize_chat=(False))
    if len(ids) < 2:
        return float("nan"), float("nan")
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    with torch.no_grad():
        _, loss = _extraer_logits_loss(model(x, y))
    if loss is None:
        return float("nan"), float("nan")

    loss_nats = loss.item()           # loss media por token, en nats
    ppl = math.exp(loss_nats)

    # bits/char: loss total (nats) = loss_media * n_targets; a bits = /ln(2);
    # normalizada por nº de caracteres del texto original.
    n_targets = len(ids) - 1
    n_chars = len(texto)
    if n_chars > 0:
        bits_por_char = (loss_nats * n_targets) / (math.log(2) * n_chars)
    else:
        bits_por_char = float("nan")
    return ppl, bits_por_char


# ============================================================
# EVALUACIÓN POR LIKELIHOOD (capacidades en el modelo BASE, sin SFT)
# ------------------------------------------------------------
# En vez de pedir al modelo que "responda" (un modelo base no sabe), le damos
# un contexto y dos continuaciones —correcta e incorrecta— y medimos a cuál
# asigna mayor log-prob por token. Si prefiere la correcta, "sabe". Funciona
# sobre el pretrain porque solo mide probabilidad de continuación.
# ============================================================
import torch.nn.functional as F


@torch.no_grad()
def _logits_crudos(model, x):
    """Obtiene logits sin targets.

    Compatible con modelos que devuelven:
      - logits
      - (logits, loss)
      - (logits, loss, ...)
    """
    salida = model(x)
    if isinstance(salida, tuple):
        return salida[0]
    return salida


def _extraer_logits_loss(salida):
    """Normaliza la salida del modelo.

    Algunas versiones del modelo devuelven exactamente:
        (logits, loss)

    Otras pueden devolver:
        (logits, loss, extra_1, extra_2, ...)

    El benchmark solo necesita logits y loss, así que ignoramos el resto.
    """
    if isinstance(salida, tuple):
        logits = salida[0] if len(salida) >= 1 else None
        loss = salida[1] if len(salida) >= 2 else None
        return logits, loss

    return salida, None


@torch.no_grad()
def logprob_continuacion(model, tokenizer, contexto, continuacion, device):
    """Log-prob media por token que el modelo asigna a 'continuacion' tras
    'contexto'. Normalizada por longitud para no favorecer continuaciones cortas."""
    ids_ctx = tokenizer.encode(contexto, normalize_chat=False)
    ids_cont = tokenizer.encode(continuacion, normalize_chat=False)
    if len(ids_cont) == 0:
        return float("-inf"), 0

    ids = ids_ctx + ids_cont
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits = _logits_crudos(model, x)[0]  # (T, vocab)

    inicio = len(ids_ctx)
    total = 0.0
    for i in range(inicio, len(ids)):
        log_probs = F.log_softmax(logits[i - 1], dim=-1)
        total += log_probs[ids[i]].item()
    return total / len(ids_cont), len(ids_cont)


def elige_correcta(model, tokenizer, contexto, correcta, incorrecta, device):
    lp_ok, _ = logprob_continuacion(model, tokenizer, contexto, correcta, device)
    lp_no, _ = logprob_continuacion(model, tokenizer, contexto, incorrecta, device)
    return (lp_ok > lp_no), lp_ok, lp_no


# --- Directorio de datos (bancos externalizados a JSON, no hardcodeados) ---
# generar_bcc_es_likelihood.py escribe directamente en estos mismos ficheros,
# así que ampliar el banco (o corregir un ítem) no requiere tocar este script.
DATOS_DIR = Path(__file__).resolve().parent / "datos"


def _cargar_json(nombre):
    ruta = DATOS_DIR / nombre
    if not ruta.exists():
        print(f"[ERROR] falta {ruta}. Ejecuta este script desde la carpeta que contiene "
              f"'datos/', o ajusta DATOS_DIR arriba.", file=sys.stderr)
        sys.exit(1)
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


# --- Prompts fijos (generación libre ES/EN/código) ---
PROMPTS_FIJOS = _cargar_json("prompts_es.json")
PROMPTS_FIJOS_EN = _cargar_json("prompts_en.json")
PROMPTS_FIJOS_CODIGO_PYTHON = _cargar_json("prompts_codigo_python.json")
PROMPTS_FIJOS_CODIGO_JSON = _cargar_json("prompts_codigo_json.json")

# --- CONOCIMIENTO: BCC-ES, opción múltiple (13 categorías, incl. fisica_quimica) ---
# Cada ítem en JSON: {"pregunta": ..., "opciones": [4], "correcta": idx}.
# Se reconstruye aquí la tupla (pregunta, [opciones], idx) que usa el resto del script.
_conocimiento_json = _cargar_json("bcc_es_conocimiento.json")
BCC_ES = {
    categoria: [(it["pregunta"], it["opciones"], it["correcta"]) for it in items]
    for categoria, items in _conocimiento_json.items()
}

# --- COMPETENCIAS: bancos de likelihood (elección forzada) ---
# Cada ítem en JSON: {"contexto": ..., "correcta": " ...", "incorrecta": " ..."}.
# Se reconstruye aquí la tupla (contexto, correcta, incorrecta).
_competencias_json = _cargar_json("bcc_es_competencias.json")
BANCOS_LIKELIHOOD = {
    categoria: [(it["contexto"], it["correcta"], it["incorrecta"]) for it in items]
    for categoria, items in _competencias_json.items()
}

# Nivel de azar de cada bloque. Son distintos (competencias es elección entre
# 2, conocimiento entre 4), así que los porcentajes brutos NO son comparables
# entre bloques: un 71% con azar 50% discrimina menos que un 52% con azar 25%.
AZAR_LIKELIHOOD = 50.0
AZAR_BCC_ES = 25.0


def margen_normalizado(pct, azar):
    """Reescala el acierto para que 0% = azar y 100% = perfecto.

    Permite comparar de un vistazo bloques con distinto número de opciones.
    """
    return max(0.0, (pct - azar) / (100.0 - azar) * 100.0)


def ic95(pct, n):
    """Semiamplitud del intervalo de confianza al 95% del acierto, en puntos.

    Con n≈100 por categoría ronda los ±10 puntos: sin esta referencia es fácil
    leer como mejora una diferencia entre checkpoints que es solo ruido.
    """
    if n <= 0:
        return 0.0
    p = pct / 100.0
    return 1.96 * (p * (1 - p) / n) ** 0.5 * 100.0


# Métricas extra de la última llamada a evaluar_likelihood(). Van aquí y no
# dentro del dict de resultados porque varios consumidores lo recorren
# desempaquetando (ok, n, pct).
DIAGNOSTICO_LIKELIHOOD = {}


def _huella_banco(datos):
    """Hash corto del contenido de un banco, para estampar en los resultados.

    Comparar dos corridas hechas con bancos distintos es un error silencioso
    (nos pasó al comparar un 48% con un 52% siendo bancos diferentes). Con la
    huella guardada en cada resultado, esa comparación se puede detectar.
    """
    serializado = json.dumps(datos, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serializado.encode("utf-8")).hexdigest()[:12]


HUELLA_CONOCIMIENTO = _huella_banco(_conocimiento_json)
HUELLA_COMPETENCIAS = _huella_banco(_competencias_json)


def evaluar_likelihood(model, tokenizer, device, verbose=False):
    """Evalúa los bancos de COMPETENCIAS por likelihood. Devuelve dict {banco: (ok, total, pct)}."""
    resultados = {}
    margenes_por_banco = {}
    print(f"  {'escenario':<22}{'aciertos':>10}{'%':>7}{'IC95':>10}{'margen':>9}")
    print("  " + "-" * 58)
    for nombre, banco in BANCOS_LIKELIHOOD.items():
        aciertos = 0
        margenes = []
        for ctx, ok, no in banco:
            acierto, lp_ok, lp_no = elige_correcta(model, tokenizer, ctx, ok, no, device)
            aciertos += int(acierto)
            # El margen lp_ok - lp_no es continuo: detecta que el modelo mejora
            # aunque todavia no haya cambiado de lado ningun item. Con ~100
            # items por categoria el acierto binario se mueve a saltos de 1 pt
            # y es demasiado ruidoso para comparar checkpoints proximos.
            margenes.append(lp_ok - lp_no)
            if verbose:
                marca = "✓" if acierto else "✗"
                ctx_corto = (ctx[:46] + "…") if len(ctx) > 47 else ctx
                print(f"      {marca} [{lp_ok:+.2f} vs {lp_no:+.2f}] {ctx_corto}")
        pct = aciertos / len(banco) * 100
        margen_medio = sum(margenes) / len(margenes)
        resultados[nombre] = (aciertos, len(banco), pct)
        margenes_por_banco[nombre] = margen_medio
        print(f"  {nombre:<22}{aciertos:>4}/{len(banco):<4}{pct:>6.0f}%"
              f"{'±' + format(ic95(pct, len(banco)), '.0f'):>10}"
              f"{margen_medio:>+9.3f}")
    tot_ok = sum(r[0] for r in resultados.values())
    tot_n = sum(r[1] for r in resultados.values())
    pct_global = tot_ok / tot_n * 100
    print("  " + "-" * 38)
    print(f"  {'GLOBAL':<18}{tot_ok:>4}/{tot_n:<4}{pct_global:>6.0f}%")
    # Este banco es binario: el azar es el 50%, no el 25%. Sin esta referencia
    # un 71% bruto parece mejor que un 52% del banco de conocimiento (azar 25%)
    # cuando en realidad discrimina menos. Ver AZAR_* y margen_sobre_azar().
    print(f"  {'':<18}{'':>4} {'':<4}{'':>6}   azar = {AZAR_LIKELIHOOD:.0f}%"
          f"  |  margen = {pct_global - AZAR_LIKELIHOOD:+.0f} pts"
          f"  |  normalizado = {margen_normalizado(pct_global, AZAR_LIKELIHOOD):.0f}%")
    # Macro: media de los porcentajes por categoria, sin ponderar por tamaño.
    # El global de arriba (micro) lo dominan las categorias mas grandes, que son
    # grandes por accidente de cuantas preguntas se escribieron, no por importancia.
    macro = sum(r[2] for r in resultados.values()) / len(resultados)
    margen_lp = sum(margenes_por_banco.values()) / len(margenes_por_banco)
    print(f"  {'':<18}{'':>4} {'':<4}{'':>6}   macro = {macro:.0f}%"
          f"  |  IC95 global = ±{ic95(pct_global, tot_n):.0f} pts"
          f"  |  margen logprob = {margen_lp:+.3f}")
    resultados["_global"] = (tot_ok, tot_n, pct_global)
    # Las metricas extra NO van dentro de 'resultados': varios consumidores
    # recorren ese dict desempaquetando (ok, n, pct) y un float o un dict ahi
    # dentro los romperia. Se publican aparte.
    DIAGNOSTICO_LIKELIHOOD.clear()
    DIAGNOSTICO_LIKELIHOOD.update({
        "macro": macro,
        "ic95_global": ic95(pct_global, tot_n),
        "margen_logprob": margen_lp,
        "margen_logprob_por_banco": margenes_por_banco,
    })
    return resultados


@torch.no_grad()
def evaluar_bcc_es(model, tokenizer, device, verbose=False):
    """Evalúa la batería BCC-ES por likelihood, al estilo lm-eval-harness:
    la opción con mayor log-prob media por token gana. Devuelve
    (resultados_por_categoria, detalle_por_pregunta)."""
    resultados = {}
    detalle = []
    print(f"  {'categoría':<24}{'aciertos':>10}{'%':>7}")
    print("  " + "-" * 44)
    for categoria, banco in BCC_ES.items():
        aciertos = 0
        for pregunta, opciones, idx_ok in banco:
            contexto = f"Pregunta: {pregunta}\nRespuesta:"
            lps = [logprob_continuacion(model, tokenizer, contexto,
                                        " " + op, device)[0]
                   for op in opciones]
            pred = max(range(len(lps)), key=lambda i: lps[i])
            acierto = (pred == idx_ok)
            aciertos += int(acierto)
            detalle.append({
                "categoria": categoria, "pregunta": pregunta,
                "opciones": opciones, "correcta": idx_ok,
                "prediccion": pred, "acierto": acierto,
                "logprobs": [round(lp, 3) for lp in lps],
            })
            if verbose:
                marca = "✓" if acierto else "✗"
                print(f"      {marca} {pregunta[:50]:<52} "
                      f"pred: {opciones[pred][:20]} | ok: {opciones[idx_ok][:20]}")
        pct = aciertos / len(banco) * 100
        resultados[categoria] = (aciertos, len(banco), pct)
        print(f"  {categoria:<24}{aciertos:>4}/{len(banco):<4}{pct:>6.0f}%")
    tot_ok = sum(r[0] for r in resultados.values())
    tot_n = sum(r[1] for r in resultados.values())
    pct_global = tot_ok / tot_n * 100
    print("  " + "-" * 44)
    print(f"  {'GLOBAL':<24}{tot_ok:>4}/{tot_n:<4}{pct_global:>6.0f}%"
          f"   (azar = {AZAR_BCC_ES:.0f}%)")
    print(f"  {'':<24}{'':>4} {'':<4}{'':>6}   margen = "
          f"{pct_global - AZAR_BCC_ES:+.0f} pts"
          f"  |  normalizado = {margen_normalizado(pct_global, AZAR_BCC_ES):.0f}%")
    resultados["_global"] = (tot_ok, tot_n, pct_global)
    return resultados, detalle



# ============================================================
# VAL LOSS
# ============================================================
def extraer_val_loss_checkpoint(checkpoint):
    """Busca la val loss (y la iteración) guardadas en el checkpoint,
    probando las claves habituales. Devuelve (val_loss, iter) o (None, None)."""
    val = None
    for clave in ("val_loss", "best_val_loss", "loss_val", "val"):
        if clave in checkpoint:
            val = float(checkpoint[clave])
            break
        if isinstance(checkpoint.get("config"), dict) and clave in checkpoint["config"]:
            val = float(checkpoint["config"][clave])
            break
    it = None
    for clave in ("iter", "iteration", "step", "iter_num", "iteracion"):
        if clave in checkpoint:
            it = int(checkpoint[clave])
            break
    return val, it


@torch.no_grad()
def calcular_val_loss_fichero(model, tokenizer, path, device, block_size,
                              max_bloques=64, batch_size=8):
    """Val loss media sobre un fichero de texto de validación: se tokeniza
    (hasta cubrir max_bloques ventanas de block_size) y se promedia la loss
    por bloque. Determinista: usa los primeros bloques del fichero."""
    with open(path, encoding="utf-8") as f:
        # límite generoso de lectura para no tokenizar gigas enteros
        texto = f.read(block_size * max_bloques * 8)
    ids = tokenizer.encode(texto, normalize_chat=False)
    n_bloques = min(max_bloques, (len(ids) - 1) // block_size)
    if n_bloques == 0:
        return float("nan"), 0
    total, hechos = 0.0, 0
    for inicio in range(0, n_bloques, batch_size):
        lote = []
        for b in range(inicio, min(inicio + batch_size, n_bloques)):
            o = b * block_size
            lote.append(ids[o:o + block_size + 1])
        x = torch.tensor([s[:-1] for s in lote], dtype=torch.long, device=device)
        y = torch.tensor([s[1:] for s in lote], dtype=torch.long, device=device)
        _, loss = _extraer_logits_loss(model(x, y))
        if loss is None:
            return float("nan"), hechos
        total += loss.item() * len(lote)
        hechos += len(lote)
    return total / hechos, hechos


def generar(model, tokenizer, prompt_fmt, device, max_new_tokens, temperature, top_k,
            deterministic=True, normalize_chat=True):
    """Genera continuación/respuesta y mide tiempo y tokens producidos.
    deterministic=True -> greedy (reproducible, propenso a bucles).
    deterministic=False -> muestreo con temperature/top_k (mas variado).

    normalize_chat=True hace tokenizer.normalize_chat_text() sobre el prompt,
    que hace .strip() a CADA linea -- inofensivo para prosa (MMLU/GSM8K), pero
    CATASTROFICO para prompts de codigo (HumanEval): borra toda la indentacion
    Python antes de tokenizar (bug real encontrado el 21/08/2026 al investigar
    por que HumanEval daba 0.0% exacto -- el docstring de la funcion llegaba al
    modelo sin indentar, a la misma columna que el "def", en vez de indentado
    dentro del cuerpo). Los llamadores con prompts de codigo deben pasar
    normalize_chat=False."""
    input_ids = tokenizer.encode(prompt_fmt, normalize_chat=normalize_chat)
    context = torch.tensor([input_ids], dtype=torch.long, device=device)

    # ID de fin de documento de pretraining. Para modelos externos via
    # adaptador_hf se usa su EOS equivalente si lo exponen.
    stop_id = getattr(tokenizer, "story_id", None)
    if stop_id is None:
        stop_id = getattr(tokenizer, "eos_id", None)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            deterministic=deterministic,
            fin_id=stop_id,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    nuevos_ids = output[0].tolist()[len(input_ids):]
    n_generados = len(nuevos_ids)

    # Cortar en <|endofstory|> (fin de documento). OBLIGATORIO hacerlo aqui,
    # sobre IDs: decode(skip_special=True) borra el token y el salto de
    # documento quedaria invisible en el informe (el modelo puede estar
    # cerrando el documento correctamente sin que se vea en el texto).
    pos_fin_doc = None
    if stop_id is not None and stop_id in nuevos_ids:
        pos_fin_doc = nuevos_ids.index(stop_id)
        nuevos_ids = nuevos_ids[:pos_fin_doc]

    # Cortar en <FIN> si existe
    if hasattr(tokenizer, "fin_id") and tokenizer.fin_id in nuevos_ids:
        nuevos_ids = nuevos_ids[:nuevos_ids.index(tokenizer.fin_id)]

    # CAMBIO 2026-08-22: .strip() (no .rstrip()) se comia la indentacion
    # inicial del cuerpo generado -- catastrofico para HumanEval, igual que
    # el bug de normalize_chat de mas arriba, pero en el otro extremo del
    # pipeline (aqui, no en el tokenizer): la continuacion de una funcion
    # SIEMPRE empieza indentada (p.ej. "    return x"), asi que quitar los
    # espacios/saltos de linea iniciales garantiza IndentationError al
    # ejecutar contra los tests -> 0% pase lo que pase, sin reflejar la
    # capacidad real del modelo. rstrip() conserva la indentacion inicial y
    # solo recorta el sobrante final; el corte por marcadores de chat de
    # abajo no depende de que el principio este limpio.
    #
    # CAMBIO 2026-08-23: decodificar SOLO nuevos_ids en aislado (sin su
    # contexto previo) hace que tokenizers tipo SentencePiece (TinyLlama/
    # Llama, via adaptador_hf) devuelvan el primer token SIN su espacio
    # inicial (el marcador "_" de inicio de palabra se pierde si no hay
    # texto previo pegado) -- resultado: "   return x" con 3 espacios en vez
    # de 4, IndentationError garantizado en HumanEval (bug real: TinyLlama
    # daba 0/80 con 78 IndentationError tras aplicar el rstrip() de arriba,
    # mientras Qwen/Pythia, con tokenizer BPE via la misma adaptador_hf, iban
    # bien). No pasa con la BPETokenizer propia de Alfa1 (no usa el marcador
    # "_" de SentencePiece). Fix: decodificar con 1 token de contexto previo
    # pegado (para que el merge de espacio se resuelva igual que en
    # generacion real) y luego recortar ese prefijo conocido del resultado.
    if len(input_ids) >= 1:
        ids_con_contexto = input_ids[-1:] + nuevos_ids
        texto_con_contexto = tokenizer.decode(ids_con_contexto, skip_special=True)
        prefijo = tokenizer.decode(input_ids[-1:], skip_special=True)
        if texto_con_contexto.startswith(prefijo):
            texto_crudo = texto_con_contexto[len(prefijo):]
        else:
            texto_crudo = tokenizer.decode(nuevos_ids, skip_special=True)
    else:
        texto_crudo = tokenizer.decode(nuevos_ids, skip_special=True)
    texto = texto_crudo.rstrip()
    for corte in ("Usuario:", "Sancho:", "<USER>", "<ASSISTANT>", "<FIN>"):
        if corte in texto:
            texto = texto.split(corte, 1)[0].rstrip()

    tok_s = n_generados / dt if dt > 0 else 0.0
    return texto, dt, n_generados, tok_s, pos_fin_doc


def _mediana(valores):
    vs = sorted(v for v in valores if not math.isnan(v))
    if not vs:
        return float("nan")
    m = len(vs)
    return vs[m // 2] if m % 2 else 0.5 * (vs[m // 2 - 1] + vs[m // 2])


def evaluar_bloque_prompts(model, tokenizer, prompts, args, device, nombre_bloque):
    """Ejecuta el mismo pipeline de perplejidad + generación greedy
    (y opcionalmente muestreo) sobre una lista de prompts dada. Reutilizada
    para los tres bloques: español (PROMPTS_FIJOS), inglés
    (PROMPTS_FIJOS_EN) y programación (PROMPTS_FIJOS_CODIGO_PYTHON,
    PROMPTS_FIJOS_CODIGO_JSON)."""
    filas = []
    suma_t = suma_tok = 0.0
    suma_ppl = 0.0
    n_ppl = 0
    n_cierra_doc = 0
    posiciones_cierre = []

    for prompt in prompts:
        prompt_fmt = formatear_prompt(prompt, args.modo)
        ppl, bpc = perplejidad(model, tokenizer, prompt_fmt, device)

        texto_g, dt, n_gen, tok_s, pos_fin_doc = generar(
            model, tokenizer, prompt_fmt, device,
            args.max_new_tokens, args.temperature, args.top_k,
            deterministic=True,
        )
        texto_s = None
        if args.comparar_muestreo:
            texto_s, _, _, _, _ = generar(
                model, tokenizer, prompt_fmt, device,
                args.max_new_tokens, args.temperature, args.top_k,
                deterministic=False,
            )

        filas.append((prompt, ppl, bpc, dt, tok_s, texto_g, texto_s))
        suma_t += dt
        suma_tok += n_gen
        if not math.isnan(ppl):
            suma_ppl += ppl
            n_ppl += 1
        if pos_fin_doc is not None:
            n_cierra_doc += 1
            posiciones_cierre.append(pos_fin_doc)

    print(f"  {'prompt':<16}{'ppl':>9}{'b/char':>8}{'tok/s':>8}   continuación (greedy) [{nombre_bloque}]")
    print("  " + "-" * 70)
    for prompt, ppl, bpc, dt, tok_s, texto_g, texto_s in filas:
        ppl_s = f"{ppl:8.1f}" if not math.isnan(ppl) else "     n/a"
        bpc_s = f"{bpc:7.2f}" if not math.isnan(bpc) else "    n/a"
        recorte = (texto_g[:38] + "…") if len(texto_g) > 39 else texto_g
        recorte = recorte.replace("\n", " ⏎ ")
        print(f"  {prompt[:16]:<16}{ppl_s:>9}{bpc_s:>8}{tok_s:>8.1f}   {recorte}")
        if args.comparar_muestreo and texto_s is not None:
            rec_s = (texto_s[:38] + "…") if len(texto_s) > 39 else texto_s
            rec_s = rec_s.replace("\n", " ⏎ ")
            print(f"  {'  (muestreo)':<16}{'':>9}{'':>8}{'':>8}   {rec_s}")
    print("  " + "-" * 70)

    ppl_media = suma_ppl / n_ppl if n_ppl else float("nan")
    ppl_mediana = _mediana([f[1] for f in filas])
    bpc_mediana = _mediana([f[2] for f in filas])
    tok_s_global = suma_tok / suma_t if suma_t else 0.0
    n_total = len(filas)
    pos_mediana_cierre = _mediana(posiciones_cierre) if posiciones_cierre else float("nan")
    pct_cierre = 100.0 * n_cierra_doc / n_total if n_total else float("nan")
    print(f"  RESUMEN [{nombre_bloque}]: ppl media {ppl_media:.1f} | ppl mediana {ppl_mediana:.1f} | "
          f"bits/char mediana {bpc_mediana:.3f} | {tok_s_global:.1f} tok/s")
    if pos_mediana_cierre == pos_mediana_cierre:  # descarta nan sin importar math
        print(f"  CIERRE DE DOCUMENTO [{nombre_bloque}]: {n_cierra_doc}/{n_total} "
              f"({pct_cierre:.0f}%) emiten <|endofstory|> antes de max_new_tokens="
              f"{args.max_new_tokens} | posición mediana: {pos_mediana_cierre:.0f} tokens")
    else:
        print(f"  CIERRE DE DOCUMENTO [{nombre_bloque}]: 0/{n_total} (0%) -- ninguna "
              f"generación emite <|endofstory|> dentro de max_new_tokens={args.max_new_tokens}")
    print()

    return {
        "filas": filas, "ppl_media": ppl_media, "ppl_mediana": ppl_mediana,
        "bpc_mediana": bpc_mediana, "t_total": suma_t, "tok_s": tok_s_global,
        "n_cierra_doc": n_cierra_doc, "n_total": n_total,
        "pct_cierra_doc": pct_cierre, "pos_mediana_cierre": pos_mediana_cierre,
    }



def evaluar_checkpoint(path, etiqueta, args, device, arquitectura):
    print("=" * 70)
    print(f"CHECKPOINT: {etiqueta}  ({path})")
    print(f"  arquitectura: {arquitectura['model_modulo']}.{arquitectura['model_clase']} | "
          f"tokenizer: {arquitectura['tok_modulo']}.{arquitectura['tok_clase']}")
    print("=" * 70)

    t0 = time.time()
    model, tokenizer, config, checkpoint = cargar_modelo(
        path, device,
        arquitectura["model_modulo"], arquitectura["model_clase"],
        arquitectura["tok_modulo"], arquitectura["tok_clase"],
    )
    t_carga = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Carga: {t_carga:.2f}s | parámetros: {n_params:,} | "
          f"block_size: {config['block_size']} | vocab: {config['vocab_size']}")

    # --- Val loss: la guardada en el checkpoint y, opcionalmente, la
    # recalculada sobre un fichero de validación (--val-file) ---
    val_ckpt, iter_ckpt = extraer_val_loss_checkpoint(checkpoint)
    val_fichero, n_bloques_val = None, 0
    if val_ckpt is not None:
        extra = f" (iter {iter_ckpt})" if iter_ckpt is not None else ""
        print(f"  Val loss (checkpoint): {val_ckpt:.4f}{extra}")
    if args.val_file:
        val_fichero, n_bloques_val = calcular_val_loss_fichero(
            model, tokenizer, args.val_file, device, config["block_size"])
        print(f"  Val loss ({args.val_file}, {n_bloques_val} bloques): "
              f"{val_fichero:.4f}")
    del checkpoint  # liberar memoria
    print()

    filas = []
    suma_t = suma_tok = 0.0
    suma_ppl = 0.0
    n_ppl = 0

    print(f"  BLOQUE ESPAÑOL ({etiqueta}):")
    bloque_es = evaluar_bloque_prompts(model, tokenizer, PROMPTS_FIJOS, args, device, "español")
    filas = bloque_es["filas"]
    ppl_media, ppl_mediana = bloque_es["ppl_media"], bloque_es["ppl_mediana"]
    bpc_mediana = bloque_es["bpc_mediana"]
    tok_s_global = bloque_es["tok_s"]
    n_cierra_doc = bloque_es["n_cierra_doc"]
    n_total_gen = bloque_es["n_total"]
    pct_cierra_doc = bloque_es["pct_cierra_doc"]
    pos_mediana_cierre = bloque_es["pos_mediana_cierre"]
    print("  (ppl depende del vocab; bits/char es comparable entre tokenizers distintos.)")
    print()

    # --- Bloque en inglés (opcional, activo por defecto) ---
    bloque_en = None
    if not args.sin_ingles:
        print(f"  BLOQUE INGLÉS ({etiqueta}):")
        bloque_en = evaluar_bloque_prompts(model, tokenizer, PROMPTS_FIJOS_EN, args, device, "inglés")

    # --- Bloques de programación, generativos (opcional, activos por defecto) ---
    bloque_codigo_python = bloque_codigo_json = None
    if not args.sin_codigo:
        print(f"  BLOQUE CÓDIGO PYTHON ({etiqueta}):")
        bloque_codigo_python = evaluar_bloque_prompts(
            model, tokenizer, PROMPTS_FIJOS_CODIGO_PYTHON, args, device, "código-python")
        print(f"  BLOQUE CÓDIGO JSON ({etiqueta}):")
        bloque_codigo_json = evaluar_bloque_prompts(
            model, tokenizer, PROMPTS_FIJOS_CODIGO_JSON, args, device, "código-json")

    # --- Evaluación por likelihood (capacidades en el modelo base, activa por defecto) ---
    resultados_ll = None
    if not args.sin_likelihood:
        print(f"  EVALUACIÓN POR LIKELIHOOD ({etiqueta}):")
        print("  (mide si el modelo PREFIERE la continuación correcta, sin necesitar SFT)")
        resultados_ll = evaluar_likelihood(model, tokenizer, device, verbose=args.verbose_ll)
        print()

    # --- BCC-ES: Banco de Conocimiento y Competencias en Español, elección
    # múltiple evaluada por likelihood (estilo lm-eval-harness). Incluye 13
    # categorías: las 12 de cultura general/dominio, más 'fisica_quimica'
    # (1º Bachillerato, 160 ítems convertidos a tipo test a partir de un
    # banco con solución verificada -- ver comentario junto a BCC_ES).
    # ------------------------------------------------------------
    tipo_test = None
    if not args.sin_bcc_es:
        n_preg_bcc_es = sum(len(b) for b in BCC_ES.values())
        print(f"  BCC-ES ({etiqueta}): {n_preg_bcc_es} preguntas de cultura "
              f"general/dominio en {len(BCC_ES)} categorías")
        print("  (estilo lm-eval-harness: gana la opción con mayor log-prob por token)")
        resultados_tipo_test, detalle_tipo_test = evaluar_bcc_es(
            model, tokenizer, device, verbose=args.verbose_bcc_es)
        tipo_test = {"resultados": resultados_tipo_test, "detalle": detalle_tipo_test}
        print()

    bcc_es = {"tipo_test": tipo_test}

    return {"etiqueta": etiqueta, "ppl_media": ppl_media, "ppl_mediana": ppl_mediana,
            "bpc_mediana": bpc_mediana, "t_total": suma_t, "tok_s": tok_s_global,
            "t_carga": t_carga, "filas": filas, "likelihood": resultados_ll,
            "bcc_es": bcc_es,
            "n_cierra_doc": n_cierra_doc, "n_total_gen": n_total_gen,
            "pct_cierra_doc": pct_cierra_doc, "pos_mediana_cierre": pos_mediana_cierre,
            "val_loss_ckpt": val_ckpt, "iter_ckpt": iter_ckpt,
            "val_loss_fichero": val_fichero, "n_bloques_val": n_bloques_val,
            "n_params": n_params, "config": config, "path": path,
            "bloque_en": bloque_en,
            "bloque_codigo_python": bloque_codigo_python,
            "bloque_codigo_json": bloque_codigo_json}


def escribir_informe(path, resultados, args, device):
    """Vuelca a un .txt el informe completo del benchmark:
      1. respuestas/continuaciones generadas para los prompts fijos
      2. batería BCC-ES: porcentaje de aciertos por categoría y detalle
      3. val loss (checkpoint y/o fichero) y métricas globales
    Un bloque por checkpoint + comparativa final si hay varios."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("INFORME DE BENCHMARK - BCC-ES\n")
        f.write(f"Fecha: {time.strftime('%Y-%m-%d %H:%M')} | dispositivo: {device} | "
                f"modo: {args.modo}\n")
        f.write(f"Generación: temp {args.temperature}, top_k {args.top_k}, "
                f"max_new {args.max_new_tokens} (greedy en tabla)\n")
        # Huella del banco: dos informes solo son comparables si coinciden.
        # Comparar resultados de bancos distintos es un error silencioso.
        f.write(f"Banco conocimiento: {HUELLA_CONOCIMIENTO} "
                f"({sum(len(v) for v in BCC_ES.values())} preguntas)\n")
        f.write(f"Banco competencias: {HUELLA_COMPETENCIAS} "
                f"({sum(len(v) for v in BANCOS_LIKELIHOOD.values())} ítems)\n")
        f.write("=" * 72 + "\n\n")

        for r in resultados:
            f.write("#" * 72 + "\n")
            f.write(f"CHECKPOINT: {r['etiqueta']}  ({r['path']})\n")
            f.write(f"Parámetros: {r['n_params']:,} | vocab: {r['config']['vocab_size']} | "
                    f"block_size: {r['config']['block_size']}\n")

            # --- Val loss ---
            if r["val_loss_ckpt"] is not None:
                extra = f" (iter {r['iter_ckpt']})" if r["iter_ckpt"] is not None else ""
                f.write(f"Val loss (checkpoint): {r['val_loss_ckpt']:.4f}{extra}\n")
            if r["val_loss_fichero"] is not None:
                f.write(f"Val loss ({args.val_file}, {r['n_bloques_val']} bloques de "
                        f"{r['config']['block_size']}): {r['val_loss_fichero']:.4f}\n")
            if r["val_loss_ckpt"] is None and r["val_loss_fichero"] is None:
                f.write("Val loss: no disponible (ni en checkpoint ni --val-file)\n")
            f.write("#" * 72 + "\n\n")

            # --- 1. Generaciones para los prompts fijos (español) ---
            f.write("-" * 72 + "\n")
            f.write("1. RESPUESTAS GENERADAS PARA LOS PROMPTS FIJOS - ESPAÑOL (greedy)\n")
            f.write("-" * 72 + "\n")
            for prompt, ppl, bpc, dt, tok_s, texto_g, texto_s in r["filas"]:
                f.write(f"\nPROMPT: {prompt}\n")
                f.write(f"  ppl: {ppl:.1f} | bits/char: {bpc:.3f} | {tok_s:.1f} tok/s\n")
                f.write(f"  GREEDY  : {texto_g}\n")
                if texto_s is not None:
                    f.write(f"  MUESTREO: {texto_s}\n")
            f.write("\n")

            # --- 1b. Generaciones en inglés ---
            if r.get("bloque_en"):
                f.write("-" * 72 + "\n")
                f.write("1b. RESPUESTAS GENERADAS PARA LOS PROMPTS FIJOS - INGLÉS (greedy)\n")
                f.write("-" * 72 + "\n")
                for prompt, ppl, bpc, dt, tok_s, texto_g, texto_s in r["bloque_en"]["filas"]:
                    f.write(f"\nPROMPT: {prompt}\n")
                    f.write(f"  ppl: {ppl:.1f} | bits/char: {bpc:.3f} | {tok_s:.1f} tok/s\n")
                    f.write(f"  GREEDY  : {texto_g}\n")
                    if texto_s is not None:
                        f.write(f"  MUESTREO: {texto_s}\n")
                b = r["bloque_en"]
                f.write(f"\nResumen inglés: ppl mediana {b['ppl_mediana']:.1f} | "
                        f"bits/char mediana {b['bpc_mediana']:.3f}\n\n")

            # --- 1c. Generaciones de código Python ---
            if r.get("bloque_codigo_python"):
                f.write("-" * 72 + "\n")
                f.write("1c. COMPLETADO DE CÓDIGO - PYTHON (greedy)\n")
                f.write("-" * 72 + "\n")
                for prompt, ppl, bpc, dt, tok_s, texto_g, texto_s in r["bloque_codigo_python"]["filas"]:
                    f.write(f"\nPROMPT:\n{prompt}\n")
                    f.write(f"  ppl: {ppl:.1f} | bits/char: {bpc:.3f} | {tok_s:.1f} tok/s\n")
                    f.write(f"  GREEDY  : {texto_g}\n")
                    if texto_s is not None:
                        f.write(f"  MUESTREO: {texto_s}\n")
                b = r["bloque_codigo_python"]
                f.write(f"\nResumen código-python: ppl mediana {b['ppl_mediana']:.1f} | "
                        f"bits/char mediana {b['bpc_mediana']:.3f}\n\n")

            # --- 1d. Generaciones de código JSON ---
            if r.get("bloque_codigo_json"):
                f.write("-" * 72 + "\n")
                f.write("1d. COMPLETADO DE CÓDIGO - JSON (greedy)\n")
                f.write("-" * 72 + "\n")
                for prompt, ppl, bpc, dt, tok_s, texto_g, texto_s in r["bloque_codigo_json"]["filas"]:
                    f.write(f"\nPROMPT:\n{prompt}\n")
                    f.write(f"  ppl: {ppl:.1f} | bits/char: {bpc:.3f} | {tok_s:.1f} tok/s\n")
                    f.write(f"  GREEDY  : {texto_g}\n")
                    if texto_s is not None:
                        f.write(f"  MUESTREO: {texto_s}\n")
                b = r["bloque_codigo_json"]
                f.write(f"\nResumen código-json: ppl mediana {b['ppl_mediana']:.1f} | "
                        f"bits/char mediana {b['bpc_mediana']:.3f}\n\n")

            # --- 2. BCC-ES: Banco de Conocimiento y Competencias en Español
            # (13 categorías de elección múltiple, incluye 'fisica_quimica') ---
            tt = r["bcc_es"]["tipo_test"]
            if tt:
                f.write("-" * 72 + "\n")
                n_preg_bcc_es = sum(len(b) for b in BCC_ES.values())
                f.write(f"2. BCC-ES ({n_preg_bcc_es} preguntas, 4 opciones, azar = 25%)\n")
                f.write("-" * 72 + "\n")
                f.write(f"{'categoría':<26}{'aciertos':>10}{'%':>8}\n")
                for cat, (ok, n, pct) in tt["resultados"].items():
                    if cat == "_global":
                        continue
                    f.write(f"{cat:<26}{ok:>5}/{n:<4}{pct:>7.0f}%\n")
                g_ok, g_n, g_pct = tt["resultados"]["_global"]
                f.write(f"{'GLOBAL':<26}{g_ok:>5}/{g_n:<4}{g_pct:>7.0f}%\n\n")
                f.write("Detalle por pregunta (✓ acierto / ✗ fallo):\n")
                for d in tt["detalle"]:
                    marca = "✓" if d["acierto"] else "✗"
                    f.write(f"  {marca} [{d['categoria']}] {d['pregunta']}\n")
                    f.write(f"      elegida: {d['opciones'][d['prediccion']]}"
                            f" | correcta: {d['opciones'][d['correcta']]}\n")
                f.write("\n")

            # --- 3. Likelihood clásica (si se pidió) ---
            if r["likelihood"]:
                f.write("-" * 72 + "\n")
                f.write("3. EVALUACIÓN POR LIKELIHOOD / COMPETENCIAS (entity tracking / "
                        "razonamiento / código / sentido común / comprensión lectora / "
                        "concordancia gramatical / correferencia / causalidad)\n")
                f.write("-" * 72 + "\n")
                for esc, (ok, n, pct) in r["likelihood"].items():
                    nombre = "GLOBAL" if esc == "_global" else esc
                    f.write(f"{nombre:<26}{ok:>5}/{n:<4}{pct:>7.0f}%\n")
                f.write("\n")

            # --- Resumen del checkpoint ---
            f.write("-" * 72 + "\n")
            f.write("RESUMEN\n")
            f.write("-" * 72 + "\n")
            f.write(f"ppl mediana: {r['ppl_mediana']:.1f} | bits/char mediana: "
                    f"{r['bpc_mediana']:.3f} | {r['tok_s']:.1f} tok/s | "
                    f"carga: {r['t_carga']:.2f}s\n")
            if r.get("n_total_gen"):
                pos_c = r["pos_mediana_cierre"]
                pos_c_s = f"{pos_c:.0f} tok" if pos_c == pos_c else "n/a"
                f.write(f"Cierre de documento (ES, greedy): {r['n_cierra_doc']}/"
                        f"{r['n_total_gen']} ({r['pct_cierra_doc']:.0f}%) emiten "
                        f"<|endofstory|> | posición mediana: {pos_c_s}\n")
            if r["bcc_es"]["tipo_test"]:
                f.write(f"BCC-ES global: "
                        f"{r['bcc_es']['tipo_test']['resultados']['_global'][2]:.0f}% "
                        f"(azar = 25%)\n")
            f.write("\n\n")

        # --- Comparativa final ---
        if len(resultados) >= 2:
            f.write("=" * 72 + "\n")
            f.write("COMPARATIVA\n")
            f.write("=" * 72 + "\n")
            cab = f"{'métrica':<26}" + "".join(f"{r['etiqueta']:>18}" for r in resultados)
            f.write(cab + "\n")
            f.write("-" * len(cab) + "\n")

            def fila(nombre, valores, fmt="{:.3f}"):
                f.write(f"{nombre:<26}" + "".join(
                    (fmt.format(v) if v is not None and not (
                        isinstance(v, float) and math.isnan(v)) else "n/a").rjust(18)
                    for v in valores) + "\n")

            fila("val loss (checkpoint)", [r["val_loss_ckpt"] for r in resultados], "{:.4f}")
            if args.val_file:
                fila("val loss (fichero)", [r["val_loss_fichero"] for r in resultados], "{:.4f}")
            fila("cierre doc (ES) %", [r.get("pct_cierra_doc") for r in resultados], "{:.0f}")
            fila("bits/char mediana (ES)", [r["bpc_mediana"] for r in resultados])
            fila("ppl mediana (ES)", [r["ppl_mediana"] for r in resultados], "{:.1f}")
            if all(r.get("bloque_en") for r in resultados):
                fila("bits/char mediana (EN)", [r["bloque_en"]["bpc_mediana"] for r in resultados])
                fila("ppl mediana (EN)", [r["bloque_en"]["ppl_mediana"] for r in resultados], "{:.1f}")
            if all(r.get("bloque_codigo_python") for r in resultados):
                fila("bits/char mediana (código-py)", [r["bloque_codigo_python"]["bpc_mediana"] for r in resultados])
                fila("ppl mediana (código-py)", [r["bloque_codigo_python"]["ppl_mediana"] for r in resultados], "{:.1f}")
            if all(r.get("bloque_codigo_json") for r in resultados):
                fila("bits/char mediana (código-json)", [r["bloque_codigo_json"]["bpc_mediana"] for r in resultados])
                fila("ppl mediana (código-json)", [r["bloque_codigo_json"]["ppl_mediana"] for r in resultados], "{:.1f}")
            if all(r["bcc_es"]["tipo_test"] for r in resultados):
                for cat in list(BCC_ES.keys()) + ["_global"]:
                    nombre = "BCC-ES GLOBAL" if cat == "_global" else f"BCC-ES {cat}"
                    fila(nombre, [r["bcc_es"]["tipo_test"]["resultados"][cat][2] for r in resultados], "{:.0f}%")
            fila("tok/s", [r["tok_s"] for r in resultados], "{:.1f}")

    print(f"\n[INFORME] guardado en: {path}")


def escribir_informe_compacto(path, resultado, args):
    """Informe compacto de UN checkpoint, solo con los valores pedidos (no
    todas las generaciones ni el detalle pregunta a pregunta):

      Iteracion: train loss y val loss
      Generacion de texto: ppl/bits-char mediana por idioma (ES/EN/
        codigo-python/codigo-json) + UN ejemplo de generacion por idioma
      Banco de razonamiento (competencias/likelihood): % por subbanco + media
      Banco de conocimiento (BCC-ES tipo test): % por subbanco + media
    """
    r = resultado
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("TEST PRETRAIN\n")
        f.write("=" * 60 + "\n\n")

        # --- Iteracion: train loss y val loss -----------------------------
        it = args.iter if args.iter is not None else r.get("iter_ckpt")
        tl = args.train_loss
        vl = args.val_loss if args.val_loss is not None else r.get("val_loss_ckpt")
        f.write(f"Iteracion: {it if it is not None else 'n/d'}\n")
        f.write(f"Train loss: {tl:.4f}\n" if tl is not None else "Train loss: n/d\n")
        f.write(f"Val loss: {vl:.4f}\n" if vl is not None else "Val loss: n/d\n")
        f.write("\n")

        # --- Generacion de texto: ppl/bits-char mediana + 1 ejemplo -------
        f.write("-" * 60 + "\n")
        f.write("GENERACION DE TEXTO\n")
        f.write("-" * 60 + "\n\n")

        bloques = [("ESPAÑOL", {"filas": r["filas"], "ppl_mediana": r["ppl_mediana"],
                                "bpc_mediana": r["bpc_mediana"]}),
                  ("INGLÉS", r.get("bloque_en")),
                  ("CÓDIGO-PYTHON", r.get("bloque_codigo_python")),
                  ("CÓDIGO-JSON", r.get("bloque_codigo_json"))]

        for nombre, b in bloques:
            if not b:
                f.write(f"{nombre}: no evaluado\n\n")
                continue
            f.write(f"{nombre}\n")
            f.write(f"  Perplexity (mediana): {b['ppl_mediana']:.2f}\n")
            f.write(f"  Bits/char (mediana): {b['bpc_mediana']:.4f}\n")
            if b["filas"]:
                prompt, ppl, bpc, dt, tok_s, texto_g, texto_s = b["filas"][0]
                f.write(f"  Ejemplo -- prompt: {prompt}\n")
                f.write(f"  Ejemplo -- generado: {texto_g}\n")
            f.write("\n")

        # --- Banco de razonamiento (competencias/likelihood) --------------
        f.write("-" * 60 + "\n")
        f.write("BANCO DE RAZONAMIENTO\n")
        f.write("-" * 60 + "\n\n")
        if r.get("likelihood"):
            margenes = DIAGNOSTICO_LIKELIHOOD.get("margen_logprob_por_banco", {})
            for nombre, (ok, n, pct) in r["likelihood"].items():
                if nombre == "_global":
                    continue
                linea = f"  {nombre}: {pct:.1f}% (±{ic95(pct, n):.0f})"
                if nombre in margenes:
                    linea += f"  margen logprob {margenes[nombre]:+.3f}"
                f.write(linea + "\n")
            _, n_global, pct_global = r["likelihood"]["_global"]
            f.write(f"\n  Media total (micro): {pct_global:.1f}% "
                    f"(±{ic95(pct_global, n_global):.0f})\n")
            if DIAGNOSTICO_LIKELIHOOD:
                f.write(f"  Media por categoría (macro): "
                        f"{DIAGNOSTICO_LIKELIHOOD['macro']:.1f}%\n")
                f.write(f"  Margen logprob medio: "
                        f"{DIAGNOSTICO_LIKELIHOOD['margen_logprob']:+.3f}\n")
            # El azar aquí es el 50% (elección entre 2), no el 25% del banco de
            # conocimiento: sin esta línea los dos bloques parecen comparables
            # y no lo son.
            f.write(f"  Azar: {AZAR_LIKELIHOOD:.0f}%  |  normalizado: "
                    f"{margen_normalizado(pct_global, AZAR_LIKELIHOOD):.0f}%\n\n")
        else:
            f.write("  no evaluado\n\n")

        # --- Banco de conocimiento (BCC-ES tipo test) ----------------------
        f.write("-" * 60 + "\n")
        f.write("BANCO DE CONOCIMIENTO\n")
        f.write("-" * 60 + "\n\n")
        tt = r.get("bcc_es", {}).get("tipo_test")
        if tt:
            for nombre, (ok, n, pct) in tt["resultados"].items():
                if nombre == "_global":
                    continue
                f.write(f"  {nombre}: {pct:.1f}%\n")
            _, _, pct_global = tt["resultados"]["_global"]
            f.write(f"\n  Media total: {pct_global:.1f}%\n\n")
        else:
            f.write("  no evaluado\n\n")

    print(f"[INFORME COMPACTO] guardado en: {path}")


def autodetectar_checkpoints():
    """Si no se pasa --checkpoints, busca .pt en el directorio actual (donde
    se ejecuta el script -- se asume que es la carpeta del modelo en
    cuestión). Prioriza los que acaban en '_best.pt'; si no hay ninguno,
    cae a cualquier .pt del directorio. Devuelve la lista ordenada de
    rutas encontradas, o [] si no hay nada."""
    aqui = Path(".")
    candidatos = sorted(aqui.glob("*_best.pt"))
    if not candidatos:
        candidatos = sorted(aqui.glob("*.pt"))
    return [str(p) for p in candidatos]


def main():
    ap = argparse.ArgumentParser(description="Benchmark comparativo de checkpoints (BCC-ES).")
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="Uno o más .pt a comparar. Si se omite, se autodetectan "
                         "los .pt del directorio actual (prioriza *_best.pt).")
    ap.add_argument("--etiquetas", nargs="+", default=None,
                    help="Nombres legibles para cada checkpoint (mismo orden)")
    ap.add_argument("--model-modulo", nargs="+", default=["alfa1.model"],
                    help="Módulo Python del que importar la clase de modelo, uno por "
                         "checkpoint (mismo orden que --checkpoints), o un único valor "
                         "para aplicar a todos. Permite mezclar arquitecturas distintas "
                         "en la misma comparativa (p.ej. v1 vs Alpha 1). "
                         "Default: alfa1.model")
    ap.add_argument("--model-clase", nargs="+", default=["SanchoGPT"],
                    help="Nombre de la clase de modelo dentro de --model-modulo, uno "
                         "por checkpoint o un único valor para todos. Default: SanchoGPT")
    ap.add_argument("--tokenizer-modulo", nargs="+", default=["alfa1.tokenizer_bpe"],
                    help="Módulo Python del que importar la clase de tokenizador, uno "
                         "por checkpoint o un único valor para todos. "
                         "Default: alfa1.tokenizer_bpe")
    ap.add_argument("--tokenizer-clase", nargs="+", default=["BPETokenizer"],
                    help="Nombre de la clase de tokenizador dentro de --tokenizer-modulo, "
                         "uno por checkpoint o un único valor para todos. "
                         "Default: BPETokenizer")
    ap.add_argument("--modo", choices=["pretrain", "chat"], default="pretrain",
                    help="'pretrain': prompt como semilla. 'chat': formato Usuario/Sancho (para SFT).")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--device", default=None, help="cuda/cpu (autodetecta si se omite)")
    ap.add_argument("--guardar-csv", default=None,
                    help="Ruta de CSV donde volcar la tabla por prompt de todos los checkpoints")
    ap.add_argument("--comparar-muestreo", action="store_true",
                    help="Genera tambien una version con muestreo (temperature/top_k) ademas "
                         "de la greedy, para ver si los bucles son del modelo o del decoding.")
    ap.add_argument("--sin-likelihood", action="store_true",
                    help="Desactiva la evaluación por likelihood (entity tracking, "
                         "razonamiento, código, sentido común, comprensión lectora) "
                         "que mide capacidades en el modelo base sin SFT. Activa por "
                         "defecto.")
    ap.add_argument("--verbose-ll", action="store_true",
                    help="Muestra el detalle por ítem de la evaluación por likelihood.")
    ap.add_argument("--informe", default=None,
                    help="Ruta del .txt de informe completo (generaciones + BCC-ES + "
                         "val loss). Por defecto: informe_benchmark_<fecha>.txt")
    ap.add_argument("--val-file", default=None,
                    help="Fichero de texto de validación para recalcular la val loss "
                         "(además de la guardada en el checkpoint, si existe).")
    ap.add_argument("--sin-bcc-es", action="store_true",
                    help="Desactiva la batería BCC-ES (activa por defecto; ver BCC_ES "
                         "para el nº de preguntas y categorías).")
    ap.add_argument("--sin-ingles", action="store_true",
                    help="Desactiva el bloque de completado en inglés (PROMPTS_FIJOS_EN, "
                         "activo por defecto).")
    ap.add_argument("--sin-codigo", action="store_true",
                    help="Desactiva el bloque de completado de código generativo "
                         "(PROMPTS_FIJOS_CODIGO_PYTHON/_JSON, activos por "
                         "defecto). No afecta al banco "
                         "CODIGO de likelihood (ver --sin-likelihood).")
    ap.add_argument("--verbose-bcc-es", action="store_true",
                    help="Muestra el detalle por pregunta de la batería BCC-ES en consola.")
    ap.add_argument("--iter", type=int, default=None,
                    help="Iteración de entrenamiento del checkpoint (la pasa "
                         "train_pretrain_bpe.py al lanzar la batería). Se usa "
                         "para el nombre del informe compacto "
                         "(test_pretrain_it<iter>.txt) y para su cabecera.")
    ap.add_argument("--train-loss", type=float, default=None,
                    help="Train loss en el momento de guardar el checkpoint. "
                         "El .pt no la guarda, así que se pasa por CLI.")
    ap.add_argument("--val-loss", type=float, default=None,
                    help="Val loss en el momento de guardar el checkpoint. Si "
                         "se omite, se usa la guardada dentro del checkpoint.")
    ap.add_argument("--informe-compacto", default=None,
                    help="Ruta del informe compacto (solo valores: loss, ppl/"
                         "bits-char mediana por idioma+1 ejemplo, % bancos de "
                         "razonamiento y conocimiento). Si se omite pero se "
                         "pasó --iter, se genera solo con un checkpoint como "
                         "test_pretrain_it<iter>.txt. Para desactivarlo del "
                         "todo usa --sin-informe-compacto.")
    ap.add_argument("--sin-informe-compacto", action="store_true",
                    help="No generar el informe compacto aunque se pase --iter.")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device} | modo: {args.modo} | "
          f"temp {args.temperature} top_k {args.top_k} max_new {args.max_new_tokens}\n")

    # --- Autodetección de checkpoints si no se pasaron por CLI ---
    if not args.checkpoints:
        args.checkpoints = autodetectar_checkpoints()
        if not args.checkpoints:
            print("[ERROR] No se pasó --checkpoints y no se encontró ningún .pt "
                  "en el directorio actual. Ejecuta el script desde la carpeta "
                  "del modelo, o indica --checkpoints explícitamente.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"[AUTO] Checkpoints detectados en el directorio actual: "
              f"{', '.join(args.checkpoints)}\n")

    # física/química es ahora una categoría más dentro de BCC_ES (elección
    # múltiple), evaluada automáticamente junto a las otras 12 categorías
    # cuando corre BCC-ES. Se desactiva con --sin-bcc-es, igual que el resto.

    etiquetas = args.etiquetas or [f"modelo_{i+1}" for i in range(len(args.checkpoints))]
    if len(etiquetas) != len(args.checkpoints):
        print("[ERROR] nº de etiquetas != nº de checkpoints", file=sys.stderr)
        sys.exit(1)

    def _broadcast(valores, n, nombre_flag):
        """Si se pasó un único valor, lo repite para los n checkpoints. Si se
        pasaron varios, exige que coincidan exactamente con nº de checkpoints
        (permite mezclar arquitecturas distintas, una por checkpoint)."""
        if len(valores) == 1:
            return valores * n
        if len(valores) != n:
            print(f"[ERROR] {nombre_flag}: se dieron {len(valores)} valores para "
                  f"{n} checkpoints (o pasas 1 para todos, o exactamente {n})",
                  file=sys.stderr)
            sys.exit(1)
        return valores

    n_ckpt = len(args.checkpoints)
    model_modulos = _broadcast(args.model_modulo, n_ckpt, "--model-modulo")
    model_clases = _broadcast(args.model_clase, n_ckpt, "--model-clase")
    tok_modulos = _broadcast(args.tokenizer_modulo, n_ckpt, "--tokenizer-modulo")
    tok_clases = _broadcast(args.tokenizer_clase, n_ckpt, "--tokenizer-clase")

    resultados = []
    for path, etq, mm, mc, tm, tc in zip(
        args.checkpoints, etiquetas, model_modulos, model_clases, tok_modulos, tok_clases
    ):
        arquitectura = {"model_modulo": mm, "model_clase": mc,
                         "tok_modulo": tm, "tok_clase": tc}
        try:
            resultados.append(evaluar_checkpoint(path, etq, args, device, arquitectura))
        except FileNotFoundError:
            print(f"[ERROR] no existe el checkpoint: {path}", file=sys.stderr)
        except Exception as e:
            print(f"[ERROR] fallo evaluando {path}: {e}", file=sys.stderr)

    # Informe completo a fichero (generaciones + BCC-ES + val loss)
    if resultados:
        ruta_informe = args.informe or \
            f"informe_benchmark_{time.strftime('%Y%m%d_%H%M')}.txt"
        escribir_informe(ruta_informe, resultados, args, device)

    # Informe COMPACTO (solo valores) del ultimo checkpoint evaluado -- el
    # pensado para el enganche automatico con train_pretrain_bpe.py.
    if resultados and not args.sin_informe_compacto:
        it = args.iter if args.iter is not None else resultados[-1].get("iter_ckpt")
        ruta_compacta = args.informe_compacto or (
            f"test_pretrain_it{it}.txt" if it is not None
            else f"test_pretrain_{time.strftime('%Y%m%d_%H%M')}.txt"
        )
        escribir_informe_compacto(ruta_compacta, resultados[-1], args)

    # Volcado a CSV (acumula todos los checkpoints en filas largas)
    if args.guardar_csv and resultados:
        import csv
        with open(args.guardar_csv, "w", newline="", encoding="utf-8") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["checkpoint", "bloque", "prompt", "perplejidad", "bits_char",
                        "tok_s", "continuacion_greedy", "continuacion_muestreo"])
            for r in resultados:
                for prompt, ppl, bpc, dt, tok_s, texto_g, texto_s in r["filas"]:
                    w.writerow([
                        r["etiqueta"], "espanol", prompt, f"{ppl:.2f}", f"{bpc:.4f}",
                        f"{tok_s:.1f}",
                        texto_g.replace("\n", " ⏎ "),
                        (texto_s or "").replace("\n", " ⏎ "),
                    ])
                if r.get("bloque_en"):
                    for prompt, ppl, bpc, dt, tok_s, texto_g, texto_s in r["bloque_en"]["filas"]:
                        w.writerow([
                            r["etiqueta"], "ingles", prompt, f"{ppl:.2f}", f"{bpc:.4f}",
                            f"{tok_s:.1f}",
                            texto_g.replace("\n", " ⏎ "),
                            (texto_s or "").replace("\n", " ⏎ "),
                        ])
                if r.get("bloque_codigo_python"):
                    for prompt, ppl, bpc, dt, tok_s, texto_g, texto_s in r["bloque_codigo_python"]["filas"]:
                        w.writerow([
                            r["etiqueta"], "codigo_python", prompt.replace("\n", " ⏎ "),
                            f"{ppl:.2f}", f"{bpc:.4f}", f"{tok_s:.1f}",
                            texto_g.replace("\n", " ⏎ "),
                            (texto_s or "").replace("\n", " ⏎ "),
                        ])
                if r.get("bloque_codigo_json"):
                    for prompt, ppl, bpc, dt, tok_s, texto_g, texto_s in r["bloque_codigo_json"]["filas"]:
                        w.writerow([
                            r["etiqueta"], "codigo_json", prompt.replace("\n", " ⏎ "),
                            f"{ppl:.2f}", f"{bpc:.4f}", f"{tok_s:.1f}",
                            texto_g.replace("\n", " ⏎ "),
                            (texto_s or "").replace("\n", " ⏎ "),
                        ])
        print(f"\n[CSV] tabla guardada en: {args.guardar_csv}")

    # Comparativa final
    if len(resultados) >= 2:
        print("=" * 70)
        print("COMPARATIVA")
        print("=" * 70)
        print(f"  {'modelo':<18}{'ppl mediana':>13}{'bits/char med':>15}{'tok/s':>10}{'carga(s)':>10}{'cierre doc':>12}")
        print("  " + "-" * 78)
        for r in resultados:
            cierre_s = f"{r['pct_cierra_doc']:.0f}%" if r.get("pct_cierra_doc") == r.get("pct_cierra_doc") else "n/a"
            print(f"  {r['etiqueta']:<18}{r['ppl_mediana']:>13.1f}{r['bpc_mediana']:>15.3f}"
                  f"{r['tok_s']:>10.1f}{r['t_carga']:>10.2f}{cierre_s:>12}")
        print("  " + "-" * 78)

        base, nuevo = resultados[0], resultados[-1]

        # bits/char es la métrica justa cuando los vocabularios difieren
        if not (math.isnan(base["bpc_mediana"]) or math.isnan(nuevo["bpc_mediana"])):
            mejora_bpc = 100 * (base["bpc_mediana"] - nuevo["bpc_mediana"]) / base["bpc_mediana"]
            signo = "mejor" if mejora_bpc > 0 else "peor"
            print(f"\n  BITS/CHAR (metrica justa entre vocabs distintos): "
                  f"{nuevo['etiqueta']} es un {abs(mejora_bpc):.1f}% {signo} que {base['etiqueta']}.")
            print("  (Mas bajo = el modelo predice mejor el español por caracter, "
                  "independiente del tamaño de vocab.)")

        if not (math.isnan(base["ppl_mediana"]) or math.isnan(nuevo["ppl_mediana"])):
            mejora = 100 * (base["ppl_mediana"] - nuevo["ppl_mediana"]) / base["ppl_mediana"]
            signo = "mejor" if mejora > 0 else "peor"
            print(f"\n  Perplejidad MEDIANA (ES): {nuevo['etiqueta']} es un {abs(mejora):.1f}% {signo} "
                  f"que {base['etiqueta']}.")
            print("  (OJO: la ppl NO es comparable si los vocabularios difieren; "
                  "usa bits/char para eso. Util solo a igualdad de tokenizer.)")

        if base.get("bloque_en") and nuevo.get("bloque_en"):
            b_base, b_nuevo = base["bloque_en"], nuevo["bloque_en"]
            print(f"\n  INGLÉS: ppl mediana {b_base['ppl_mediana']:.1f} -> {b_nuevo['ppl_mediana']:.1f} | "
                  f"bits/char mediana {b_base['bpc_mediana']:.3f} -> {b_nuevo['bpc_mediana']:.3f}")

        if base.get("bloque_codigo_python") and nuevo.get("bloque_codigo_python"):
            b_base, b_nuevo = base["bloque_codigo_python"], nuevo["bloque_codigo_python"]
            print(f"\n  CÓDIGO-PYTHON: ppl mediana {b_base['ppl_mediana']:.1f} -> {b_nuevo['ppl_mediana']:.1f} | "
                  f"bits/char mediana {b_base['bpc_mediana']:.3f} -> {b_nuevo['bpc_mediana']:.3f}")

        if base.get("bloque_codigo_json") and nuevo.get("bloque_codigo_json"):
            b_base, b_nuevo = base["bloque_codigo_json"], nuevo["bloque_codigo_json"]
            print(f"\n  CÓDIGO-JSON: ppl mediana {b_base['ppl_mediana']:.1f} -> {b_nuevo['ppl_mediana']:.1f} | "
                  f"bits/char mediana {b_base['bpc_mediana']:.3f} -> {b_nuevo['bpc_mediana']:.3f}")

        # Comparativa de likelihood (si se evaluó)
        if all(r.get("likelihood") for r in resultados):
            print("\n  LIKELIHOOD (capacidades en el modelo base):")
            escenarios = list(BANCOS_LIKELIHOOD.keys()) + ["_global"]
            cab = f"  {'escenario':<22}" + "".join(f"{r['etiqueta']:>16}" for r in resultados)
            print(cab)
            print("  " + "-" * (18 + 16 * len(resultados)))
            for esc in escenarios:
                nombre = "GLOBAL" if esc == "_global" else esc
                fila = f"  {nombre:<22}"
                for r in resultados:
                    ok, n, pct = r["likelihood"][esc]
                    fila += f"{ok}/{n} ({pct:.0f}%)".rjust(16)
                print(fila)
            print("\n  (Si el modelo con código puntúa más alto en entity_tracking y "
                  "razonamiento,\n   ahí tienes evidencia —en tu modelo, en español— de que el "
                  "código ayudó.)")


if __name__ == "__main__":
    main()