#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chat_web_app.py
================
App web (Flask, sin Ollama/GGUF) para conversar directamente en PyTorch con
cualquier checkpoint SFT de Alfa1.2 -- pensada como la "prueba de realidad"
tras 13 experimentos de benchmark: el score global lleva varias rondas
oscilando dentro del ruido, y conversar de verdad revela cosas (coherencia
en turnos largos, bucles de repeticion, deriva de idioma) que el benchmark
no mide.

Reutiliza la logica ya validada de chat_sft_console.py (generacion con
kv_cache + penalizacion de repeticion estilo CTRL), pero:
  - Multi-turno real: el cliente manda el historial completo, el server
    reconstruye <USER>u1<ASSISTANT>a1<USER>u2<ASSISTANT>a2... (mismo formato
    que ven los documentos multiturno en entrenamiento tras el fix del
    2026-09-06 en convertir_corpus_instrucciones_a_txt.py).
  - Selector de checkpoint en la UI (v10/v11/v12/v13...), para comparar
    versiones conversando, no solo por el numero de benchmark.
  - Parametros de generacion (temperatura, top_k, repeticion) ajustables
    desde la UI.
  - CORTE EN <FIN> UNICAMENTE, no en "\\n\\n" -- ese "\\n\\n" fue exactamente
    el bug de truncado que escondio 10 experimentos de benchmark
    (ver ejecutar_benchmark_sft.py). No se repite aqui.
  - Logging: cada turno se anade a logs/chat_log_<fecha>.jsonl con el
    historial completo, el prompt crudo, la respuesta SIN limpiar y la
    limpia, los parametros de generacion, si paro en <FIN>, y latencia --
    para poder auditar despues sin tener que reproducir la conversacion.

Uso:
    python chat_web_app.py
    (abre http://127.0.0.1:5000)
"""
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory
from transformers import AutoModelForCausalLM, AutoTokenizer

from alfa1.config_sft import (
    N_EMBD, N_HEAD, N_LAYER, N_KV_HEAD, FFN_HIDDEN, TIE_WEIGHTS, ROPE_THETA,
    BLOCK_SIZE, DEVICE, TOKENIZER_STATE_PATH,
    TOOL_TAG, TOOL_END_TAG, TURN_TAG, FIN_TAG,
)
from alfa1.model import SanchoGPT

# Modo agente (2026-09-13): la ruta plana (max_new_tokens, corte en <FIN>)
# NO ejecuta herramientas -- verificado en vivo con v30 que ante una pregunta
# ReAct encadena dos Action seguidas inventando el dato de la segunda (nunca
# llega a Final Answer). Se reutilizan las MISMAS herramientas/instrucciones
# que entrenan el dataset, para que la Observation inyectada aqui sea
# idéntica en formato a la que vio en entreno.
sys.path.insert(0, r"D:\Generar_SFT\agente_react")
try:
    # 2026-09-26: generar_dataset_react.py ya no existe en agente_react/ (se
    # perdio/renombro en algun momento entre v39 y ahora, sin relacion con
    # este cambio). Solo lo usa ejecutar_ciclo_agente() para checkpoints
    # ANTERIORES a v39 (fuera de CHECKPOINTS_V39, p.ej. v38); el resto de la
    # app -- chat normal, modo agente v39+, y el nuevo checkpoint HF -- no lo
    # necesita, asi que se hace opcional en vez de tumbar el arranque entero.
    from generar_dataset_react import INSTRUCCIONES as INSTRUCCIONES_AGENTE, MAX_CICLOS as MAX_CICLOS_AGENTE, RE_ACTION, RE_FINAL
except ImportError as _exc:
    print(f"[chat_web_app] AVISO: generar_dataset_react no disponible ({_exc}); "
          "el modo agente de checkpoints anteriores a v39 (p.ej. v38) no funcionara.")
    INSTRUCCIONES_AGENTE = MAX_CICLOS_AGENTE = RE_ACTION = RE_FINAL = None
from herramientas_react import HERRAMIENTAS
from alfa1.tokenizer_bpe import BPETokenizer

# v39 (2026-09-21): el preambulo se importa del generador del corpus, NO se
# copia aqui. Si se reescribiera a mano, cualquier divergencia (una coma, el
# indentado del JSON Schema) pondria al modelo fuera de la distribucion que
# vio en entreno sin que nada fallase de forma visible.
from generar_corpus_v39 import PREAMBULO_AGENTE as PREAMBULO_AGENTE_V39

# Checkpoints entrenados con el formato estructurado v39. Lista explicita en
# vez de adivinar por el nombre del fichero: el formato del corpus no se
# puede deducir del .pt, y equivocarse es silencioso (el modelo respondería
# en prosa sin llamar a ninguna herramienta).
CHECKPOINTS_V39 = {
    "alpha_1_sft_release_v39.pt",
    "alpha_1_sft_release_v39_best.pt",
    # v40 y v41 heredan el formato estructurado de v39 (releases construidas
    # con --react-v39). Faltaban aqui: sin esta entrada se evaluaban con el
    # bucle ReAct antiguo y no llamaban a ninguna herramienta.
    "alpha_1_sft_release_v40.pt",
    "alpha_1_sft_release_v40_best.pt",
    "alpha_1_sft_release_v41.pt",
    "alpha_1_sft_release_v41_best.pt",
}

AQUI = Path(__file__).resolve().parent
CHECKPOINTS_DIR = AQUI / "checkpoints"
LOGS_DIR = AQUI / "logs_chat_web"
LOGS_DIR.mkdir(exist_ok=True)

# Checkpoints que se ofrecen en el selector. Se filtra a los que existan de
# verdad en disco al arrancar (algunos v5-v9 se borraron el 2026-09-06 por
# falta de espacio). El _best de cada version es el que se usa siempre para
# comparar (mismo criterio que en los benchmarks). El primero de la lista es
# el que se carga por defecto al arrancar.
#
# 2026-09-07: v16 es el checkpoint de REFERENCIA (cierre del SFT tras 16
# experimentos). Cierra el agujero de seguridad (explosivos/malware, jailbreak
# -> 20/20) sin la regresion de identidad de v15 ("desarrollado por OpenAI",
# familia inventada). Benchmark ~67% (≈ v14 66.2%), val loss 1.1883. Ver
# Benchmark_SFT/resultados/scoring_v14_v15_v16.md. v15 se deja en la lista
# solo para poder reproducir la regresion si hiciera falta.
# El release "release" se movio fuera de checkpoints/ a D:\ (junto al base y
# al annealing); la lista acepta rutas absolutas. v24 (2026-09-11) sustituye
# a v23: corpus +3.613 ejemplos (multiturno_estructurado, restricciones_
# formato, roleplay) dirigidos a agujeros de cobertura confirmados en el
# control con Qwen2.5-0.5B (ver Benchmark_SFT/mtbench/SCORING_mtbench_v2_
# vs_v1.md). Mejor val loss de toda la serie (1.1372 vs 1.1824 de v23),
# respuestas ~20-25% mas largas/completas en EN sin vacias, seguridad
# 10/10 intacta, tablas de formato arregladas. Sin regresiones detectadas
# en el benchmark de 316 ni en MT-Bench bilingue. Ver
# Benchmark_SFT/resultados/ (tablas v24 vs v23 vs qwen).
#
# v23 (2026-09-08) sustituyo a v16: +4 identidad (cobertura de sondas en
# ES), id_en10 arreglado, seguridad 20/20, sin regresiones. Ver
# scoring_v16_vs_v23.md.
# v30 (2026-09-13) ADOPTADO, sustituye a v24: agente ReAct de 4 estados
# (KNOW/SEARCH/TOOL/ABSTAIN, ver D:/Generar_SFT/agente_react/), val loss
# 1.1182, sin regresion en el benchmark de 316 (seguridad 10/10, identidad
# 20/20, JSON 12/12). v24 se deja en la lista para comparar/revertir.
# v39 (2026-09-21): mismo corpus que v38 con el bloque de agente en formato
# estructurado (<tool>/<turno>). val loss 1.2288 vs 1.2282 de v38 -- el
# cambio de formato no cuesta nada. En modo agente usa otro bucle
# (ejecutar_ciclo_agente_v39, ver CHECKPOINTS_V39).
# v30 y v24 se quitan de la lista: los ficheros se borraron el 2026-09-18 y
# seleccionarlos daba FileNotFoundError. v31 es la referencia vigente.
# El PRIMERO de la lista es el que se carga por defecto al arrancar. Se pone
# aqui el checkpoint que se esta evaluando: dejarlo en uno antiguo hizo que una
# sesion entera de evaluacion manual (2026-09-22, 18 turnos) midiera v39
# creyendo que medía v41, porque el selector arranca en el primero.
CANDIDATOS_CHECKPOINT = [
    r"C:\Users\alegaz\Desktop\PROYECTO SANCHO\MODELOS\Alfa1_05B_v2\checkpoints\alpha_1_sft_release_v41_best.pt",
    r"C:\Users\alegaz\Desktop\PROYECTO SANCHO\MODELOS\Alfa1_05B_v2\checkpoints\alpha_1_sft_release_v40_best.pt",
    r"C:\Users\alegaz\Desktop\PROYECTO SANCHO\MODELOS\Alfa1_05B_v2\checkpoints\alpha_1_sft_release_v39_best.pt",
    r"D:\alpha_1_05B_it_v31.pt",
    r"C:\Users\alegaz\Desktop\PROYECTO SANCHO\MODELOS\Alfa1_05B_v2\checkpoints\alpha_1_sft_release_v38_best.pt",
    # 2026-09-26: Alfa1-G-4B-it (Gemma-3 4B + LoRA sobre corpus v43, ver
    # TABLA_PAPER.md Tabla 4). NO es un checkpoint SanchoGPT: es una carpeta
    # con un modelo HF completo (merge del adaptador LoRA), tokenizer HF
    # propio y los MISMOS tokens especiales <USER>/<ASSISTANT>/<FIN> que el
    # resto (los añadió sft_full_qwen.py). Se detecta por ser un directorio,
    # no un .pt, y usa cargar_modelo_hf()/generar_hf() en vez de la ruta
    # SanchoGPT. Puesto al final para no tocar el checkpoint por defecto.
    r"C:\Users\alegaz\Desktop\PROYECTO SANCHO\MODELOS\gemma3_4b_base\sft_v43_out\best",
]

_tokenizer = None
TOOL_END_ID = None  # id de </tool>; lo fija cargar_tokenizer()
_modelos_cache = {}  # nombre_fichero -> modelo cargado (cache, no se descarga)
_modelos_hf_cache = {}  # nombre_ruta -> (model, tokenizer, fin_id) para checkpoints HF (ver GEMMA_4B_PATH)


def _es_checkpoint_hf(nombre_checkpoint):
    """Distingue un checkpoint SanchoGPT (.pt) de un modelo HF guardado como
    carpeta (config.json + safetensors) -- es la unica senal que hace falta,
    porque todos los .pt de Alfa1 son ficheros y todos los HF son directorios."""
    return _ruta_ckpt(nombre_checkpoint).is_dir()


def cargar_modelo_hf(nombre_checkpoint):
    if nombre_checkpoint in _modelos_hf_cache:
        return _modelos_hf_cache[nombre_checkpoint]
    ruta = _ruta_ckpt(nombre_checkpoint)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {ruta}")
    print(f"[chat_web_app] Cargando (HF) {nombre_checkpoint} ...")
    tok = AutoTokenizer.from_pretrained(str(ruta))
    model = AutoModelForCausalLM.from_pretrained(str(ruta), dtype=torch.bfloat16)
    model.to(DEVICE)
    model.eval()
    fin_id = tok.convert_tokens_to_ids("<FIN>")
    _modelos_hf_cache[nombre_checkpoint] = (model, tok, fin_id)
    print(f"[chat_web_app] {nombre_checkpoint} listo en {DEVICE} (HF).")
    return model, tok, fin_id


def _limpiar_respuesta_hf(texto):
    """Misma logica que limpiar_respuesta() pero sobre texto ya decodificado
    (el tokenizer HF no expone skip_special igual que BPETokenizer)."""
    texto = texto.split("<FIN>", 1)[0]
    for corte in ("<USER>", "<ASSISTANT>"):
        if corte in texto:
            texto = texto.split(corte, 1)[0]
    return texto.strip()


@torch.no_grad()
def generar_hf(model, tok, fin_id, historial, mensaje, gen_kwargs):
    """Equivalente de _generar_un_turno() para el checkpoint HF: mismo formato
    de prompt (<USER>.. <ASSISTANT>..), generacion con la API de transformers
    en vez del bucle kv_cache a mano de SanchoGPT. Solo chat plano -- el modo
    agente para este checkpoint no esta implementado todavia (ver api_chat)."""
    partes = []
    for turno in historial:
        tag = "<USER>" if turno["role"] == "user" else "<ASSISTANT>"
        partes.append(f"{tag}{turno['content']}")
    partes.append(f"<USER>{mensaje}<ASSISTANT>")
    prompt = "".join(partes)

    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(DEVICE)
    n_prompt = enc["input_ids"].shape[1]
    salida = model.generate(
        **enc,
        max_new_tokens=gen_kwargs["max_new_tokens"],
        do_sample=gen_kwargs["temperature"] > 0,
        temperature=max(gen_kwargs["temperature"], 1e-5),
        top_k=gen_kwargs["top_k"] if gen_kwargs["top_k"] > 0 else None,
        repetition_penalty=gen_kwargs["repetition_penalty"],
        eos_token_id=fin_id,
        pad_token_id=tok.pad_token_id,
    )
    generados = salida[0, n_prompt:]
    paro_en_fin = bool(len(generados) and int(generados[-1]) == fin_id)
    texto = tok.decode(generados, skip_special_tokens=False)
    return _limpiar_respuesta_hf(texto), len(generados), paro_en_fin


def cargar_tokenizer():
    global _tokenizer, TOOL_END_ID
    if _tokenizer is None:
        with open(TOKENIZER_STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        _tokenizer = BPETokenizer()
        _tokenizer.load_state(state)
        TOOL_END_ID = _tokenizer.encode(TOOL_END_TAG)[0]
    return _tokenizer


def cargar_modelo(nombre_checkpoint):
    if nombre_checkpoint in _modelos_cache:
        return _modelos_cache[nombre_checkpoint]

    tokenizer = cargar_tokenizer()
    # Acepta rutas absolutas (el v16 "release" vive en D:\alpha_1_05B_it_v16.pt,
    # fuera de checkpoints/) o nombres de fichero relativos a CHECKPOINTS_DIR.
    p = Path(nombre_checkpoint)
    ruta = p if p.is_absolute() else CHECKPOINTS_DIR / nombre_checkpoint
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {ruta}")

    print(f"[chat_web_app] Cargando {nombre_checkpoint} ...")
    model = SanchoGPT(
        vocab_size=tokenizer.vocab_size(),
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=0.0,
        device=DEVICE,
        n_kv_head=N_KV_HEAD,
        tie_weights=TIE_WEIGHTS,
        rope_theta=ROPE_THETA,
        ffn_hidden=FFN_HIDDEN,
    )
    ckpt = torch.load(ruta, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    if DEVICE == "cuda":
        # Sin esto, scaled_dot_product_attention no encuentra kernel en fp32
        # con el backend CUDNN_ATTENTION forzado en model.py para prefill en
        # CUDA (ver alfa1/model.py linea ~190) -- mismo fix que ya usa
        # Benchmark_SFT/ejecutar_benchmark_sft.py.
        model.to(torch.bfloat16)
    model.eval()

    _modelos_cache[nombre_checkpoint] = model
    print(f"[chat_web_app] {nombre_checkpoint} listo en {DEVICE}.")
    return model


# ============================================================
# Generacion: identica a chat_sft_console.py (kv_cache + penalizacion CTRL),
# reproducida aqui para no importar un script pensado para consola/input().
# ============================================================
def aplicar_penalizacion_repeticion(logits, generados_recientes, penalty):
    if penalty == 1.0 or not generados_recientes:
        return logits
    # Vectorizado (2026-09-20): el bucle Python anterior indexaba logits[0, tid]
    # escalar a escalar, y cada acceso a un tensor en GPU forzaba una
    # sincronizacion GPU<->CPU por cada id del historial y por cada token
    # generado (~0.25s/token, GPU al 0%). Esta version hace la misma operacion
    # matematica en una sola pasada sobre GPU (gather + where + scatter).
    ids = torch.tensor(sorted(set(generados_recientes)), device=logits.device, dtype=torch.long)
    vals = logits[0, ids]
    vals = torch.where(vals > 0, vals / penalty, vals * penalty)
    logits[0, ids] = vals
    return logits


@torch.no_grad()
def generar(model, idx, max_new_tokens, temperature, top_k, fin_id,
            repetition_penalty, repetition_window, paradas_extra=None):
    # paradas_extra (v39): ids adicionales que cortan la generacion. Se usa
    # para parar en </tool>, de modo que la herramienta la ejecuta el harness
    # y el modelo NUNCA escribe el resultado. Sin esto el modelo continua y
    # se inventa la Observation (comprobado: "El planeta rojo es Jupiter").
    idx_cond = idx[:, -model.block_size:]
    pos_offset = 0
    logits, _, kv_caches = model(idx_cond, use_cache=True, pos_offset=pos_offset)
    pos_offset = idx_cond.shape[1]
    logits = logits[:, -1, :].clone()

    generados = []
    for _ in range(max_new_tokens):
        historial = generados[-repetition_window:] if repetition_window > 0 else generados
        logits = aplicar_penalizacion_repeticion(logits, historial, repetition_penalty)

        logits = logits / max(temperature, 1e-5)
        if top_k is not None and top_k > 0:
            values, indices = torch.topk(logits, min(top_k, logits.shape[-1]))
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(1, indices, values)
            logits = filtered
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)
        generados.append(idx_next.item())

        if fin_id is not None and idx_next.item() == fin_id:
            break
        if paradas_extra and idx_next.item() in paradas_extra:
            break

        if pos_offset >= model.block_size:
            kv_caches = [
                (k[:, :, -(model.block_size - 1):, :], v[:, :, -(model.block_size - 1):, :])
                for (k, v) in kv_caches
            ]
            pos_offset = model.block_size - 1

        logits, _, kv_caches = model(idx_next, use_cache=True, kv_caches=kv_caches, pos_offset=pos_offset)
        pos_offset += 1
        logits = logits[:, -1, :].clone()

    return generados


def construir_prompt_multiturno(tokenizer, historial, mensaje_nuevo):
    """historial: lista de {"role": "user"|"assistant", "content": str},
    en el MISMO orden que ve el modelo en los documentos multiturno de
    entrenamiento (turnos concatenados, un solo <FIN> al final)."""
    partes = []
    for turno in historial:
        tag = tokenizer.user_token if turno["role"] == "user" else tokenizer.assistant_token
        partes.append(f"{tag}{turno['content']}")
    partes.append(f"{tokenizer.user_token}{mensaje_nuevo}{tokenizer.assistant_token}")
    return "".join(partes)


def limpiar_respuesta(tokenizer, respuesta_ids):
    """Corta SOLO en <FIN> o en un <USER>/<ASSISTANT> espurio -- NUNCA en
    '\\n\\n'. Ese corte fue el bug real que trunco 69/316 items del
    benchmark durante 10 experimentos (general_email/opinion al 100%
    truncados) antes de detectarse el 2026-09-06. No reintroducirlo aqui."""
    if tokenizer.fin_id in respuesta_ids:
        respuesta_ids = respuesta_ids[:respuesta_ids.index(tokenizer.fin_id)]
    respuesta = tokenizer.decode(respuesta_ids, skip_special=True).strip()
    for corte in (tokenizer.user_token, tokenizer.assistant_token, "<USER>", "<ASSISTANT>"):
        if corte in respuesta:
            respuesta = respuesta.split(corte, 1)[0].strip()
    return respuesta


def _generar_un_turno(model, tokenizer, historial, mensaje, gen_kwargs):
    """Genera un unico turno de assistant (misma logica que api_chat plano) y
    devuelve (texto_limpio, n_tokens). No hace parsing ReAct -- eso lo hace
    la funcion que llama."""
    prompt = construir_prompt_multiturno(tokenizer, historial, mensaje)
    input_ids = tokenizer.encode(prompt, normalize_chat=False)
    max_new_tokens = gen_kwargs["max_new_tokens"]
    if len(input_ids) >= BLOCK_SIZE - max_new_tokens:
        input_ids = input_ids[len(input_ids) - (BLOCK_SIZE - max_new_tokens):]
    context = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    ids_generados = generar(model, context, fin_id=tokenizer.fin_id, **gen_kwargs)
    return limpiar_respuesta(tokenizer, ids_generados), len(ids_generados)


def _resumir_contexto_previo(historial_conversacion, max_turnos=3):
    """Convierte los ultimos turnos de chat normal en un bloque de texto plano
    para incrustar DENTRO de la Pregunta del ciclo agente (nunca como turnos
    "user"/"assistant" separados: el SFT de ReAct solo vio un unico turno
    "user" con INSTRUCCIONES+Pregunta, asi que anadir turnos previos de verdad
    seria una estructura fuera de distribucion). Devuelve "" si no hay nada."""
    if not historial_conversacion:
        return ""
    turnos = historial_conversacion[-(max_turnos * 2):]
    lineas = []
    for turno in turnos:
        rol = "Usuario" if turno.get("role") == "user" else "Tu"
        lineas.append(f"{rol}: {turno.get('content', '')}")
    return "\n".join(lineas)


def ejecutar_ciclo_agente(model, tokenizer, pregunta_usuario, gen_kwargs, historial_conversacion=None):
    """Bucle ReAct real: genera un turno, si es Action ejecuta la herramienta
    DE VERDAD (nunca la escribe el LLM) e inyecta la Observation real como
    turno "user" antes de generar el siguiente ciclo -- misma mecanica que
    generar_dataset_react.ejecutar_react() y evaluar_agente.py, reimplementada
    aqui sobre generar() local en vez de una llamada HTTP a vLLM/Ollama.
    Si se pasa historial_conversacion (turnos previos del chat normal), se
    resume como texto plano ANTES de "Pregunta:" para dar contexto factual
    (p.ej. un presupuesto mencionado antes) sin romper el formato de un unico
    turno "user" que vio el SFT (ver _resumir_contexto_previo).
    Devuelve (texto_para_mostrar, traza_completa, tokens_totales)."""
    if INSTRUCCIONES_AGENTE is None:
        raise RuntimeError("Modo agente no disponible para este checkpoint: falta generar_dataset_react.py "
                          "(ver aviso en el arranque del servidor).")
    contexto_previo = _resumir_contexto_previo(historial_conversacion)
    bloque_contexto = f"Contexto de la conversación previa:\n{contexto_previo}\n\n" if contexto_previo else ""
    historial = [{"role": "user", "content": f"{INSTRUCCIONES_AGENTE}\n\n{bloque_contexto}Pregunta: {pregunta_usuario}"}]
    traza = []
    tokens_totales = 0
    mensaje_pendiente = historial[0]["content"]
    historial_previo = []

    for ciclo in range(MAX_CICLOS_AGENTE):
        texto, n_tok = _generar_un_turno(model, tokenizer, historial_previo, mensaje_pendiente, gen_kwargs)
        tokens_totales += n_tok

        m_final = RE_FINAL.search(texto)
        m_action = RE_ACTION.search(texto)

        if m_final and (not m_action or m_final.start() < m_action.start()):
            traza.append({"tipo": "assistant", "texto": texto.strip()})
            break

        if not m_action:
            traza.append({"tipo": "assistant", "texto": texto.strip()})
            traza.append({"tipo": "error", "texto": "(no generó Action ni Final Answer parseable)"})
            break

        herramienta = m_action.group(1).lower()
        argumento = m_action.group(2).strip()
        recorte = texto[:m_action.end()].strip()  # descarta cualquier cosa tras la Action (ver hallazgo v30)
        traza.append({"tipo": "assistant", "texto": recorte})

        if herramienta not in HERRAMIENTAS:
            traza.append({"tipo": "error", "texto": f"(herramienta desconocida: {herramienta})"})
            break

        observacion = HERRAMIENTAS[herramienta](argumento)
        traza.append({"tipo": "observation", "texto": observacion})

        historial_previo = historial_previo + [
            {"role": "user", "content": mensaje_pendiente},
            {"role": "assistant", "content": recorte},
        ]
        mensaje_pendiente = f"Observation: {observacion}"
    else:
        traza.append({"tipo": "error", "texto": f"(superó {MAX_CICLOS_AGENTE} ciclos sin Final Answer)"})

    texto_mostrar = "\n\n".join(
        f"[Observation: {t['texto']}]" if t["tipo"] == "observation"
        else f"[{t['texto']}]" if t["tipo"] == "error"
        else t["texto"]
        for t in traza
    )
    return texto_mostrar, traza, tokens_totales


def _partir_por_assistant(ids, tokenizer):
    """Parte la secuencia generada por el token <ASSISTANT>.

    v40 introdujo dos tramos seguidos (razonamiento y respuesta). Devuelve
    una lista de tramos de ids, sin el separador; con el formato de v39 (un
    solo tramo) devuelve la lista entera tal cual, asi que sirve para ambos.
    """
    sep = tokenizer.encode(tokenizer.assistant_token)[0]
    tramos, actual = [], []
    for tid in ids:
        if int(tid) == sep:
            tramos.append(actual)
            actual = []
        else:
            actual.append(tid)
    tramos.append(actual)
    return [t for t in tramos if t]


def ejecutar_ciclo_agente_v39(model, tokenizer, pregunta_usuario, gen_kwargs,
                              historial_conversacion=None):
    """Bucle de agente para checkpoints v39 (formato estructurado).

    Diferencias reales con ejecutar_ciclo_agente (ReAct en texto libre):
      - la generacion PARA en </tool>; la herramienta la ejecuta el harness.
        Con "Action: buscar[...]" habia que recortar por regex lo que el
        modelo siguiera escribiendo; aqui el corte es un token unico.
      - el resultado vuelve en un turno <turno>, no disfrazado de <USER>
        Observation:, que es lo que confundia un resultado con una pregunta.
      - la llamada es JSON: si no parsea, se ve y se corta, en vez de
        ejecutar una herramienta con un argumento a medias.
    """
    gen = dict(gen_kwargs)
    gen.pop("paradas_extra", None)
    contexto_previo = _resumir_contexto_previo(historial_conversacion)
    bloque = f"Contexto de la conversación previa:\n{contexto_previo}\n\n" if contexto_previo else ""
    prompt = (tokenizer.user_token + PREAMBULO_AGENTE_V39 + bloque + pregunta_usuario
              + tokenizer.assistant_token)

    traza = []
    tokens_totales = 0
    for _ in range(MAX_CICLOS_AGENTE):
        ids = torch.tensor([tokenizer.encode(prompt, normalize_chat=False)],
                           dtype=torch.long, device=DEVICE)
        generados = generar(model, ids, fin_id=tokenizer.fin_id,
                            paradas_extra={TOOL_END_ID}, **gen)
        tokens_totales += len(generados)
        texto = tokenizer.decode(generados)

        if TOOL_TAG not in texto:
            # v40/v41 separan razonamiento y respuesta en DOS tramos
            # <ASSISTANT> seguidos. Hay que partir por el TOKEN, antes de
            # decodificar: tokenizer.decode() traduce <ASSISTANT> a "Sancho:",
            # asi que decodificar de golpe pegaba los dos tramos con un
            # "Sancho:" en medio ("...del lenguaje.Sancho:La corriente..."),
            # que parecia un fallo del modelo y no lo era.
            for parte in _partir_por_assistant(generados, tokenizer):
                limpio = tokenizer.decode(parte).replace(FIN_TAG, "").strip()
                if limpio:
                    traza.append({"tipo": "assistant", "texto": limpio})
            break

        cuerpo = texto.split(TOOL_TAG, 1)[1].split(TOOL_END_TAG)[0]
        try:
            llamada = json.loads(cuerpo)
            nombre = llamada["name"]
            argumentos = llamada["arguments"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            traza.append({"tipo": "error", "texto": f"(llamada no parseable: {exc}) {cuerpo[:120]}"})
            break
        if nombre not in HERRAMIENTAS:
            traza.append({"tipo": "error", "texto": f"(herramienta desconocida: {nombre})"})
            break
        if not isinstance(argumentos, dict) or len(argumentos) != 1:
            traza.append({"tipo": "error", "texto": f"(argumentos invalidos: {argumentos!r})"})
            break

        llamada_txt = json.dumps(llamada, ensure_ascii=False)
        traza.append({"tipo": "assistant", "texto": f"{TOOL_TAG}{llamada_txt}{TOOL_END_TAG}"})
        resultado = HERRAMIENTAS[nombre](str(next(iter(argumentos.values()))))
        traza.append({"tipo": "observation", "texto": resultado})

        prompt += (TOOL_TAG + llamada_txt + TOOL_END_TAG
                   + TURN_TAG + str(resultado) + tokenizer.assistant_token)
    else:
        traza.append({"tipo": "error", "texto": f"(superó {MAX_CICLOS_AGENTE} ciclos sin respuesta)"})

    texto_mostrar = "\n\n".join(
        f"[Resultado: {t['texto']}]" if t["tipo"] == "observation"
        else f"[{t['texto']}]" if t["tipo"] == "error"
        else t["texto"]
        for t in traza
    )
    return texto_mostrar, traza, tokens_totales


def registrar_log(payload):
    fecha = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ruta = LOGS_DIR / f"chat_log_{fecha}.jsonl"
    with ruta.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ============================================================
# Flask
# ============================================================
app = Flask(__name__, static_folder=None)


@app.route("/")
def index():
    return send_from_directory(AQUI / "chat_web_static", "index.html")


def _ruta_ckpt(c):
    p = Path(c)
    return p if p.is_absolute() else CHECKPOINTS_DIR / c


@app.route("/api/checkpoints")
def api_checkpoints():
    disponibles = [c for c in CANDIDATOS_CHECKPOINT if _ruta_ckpt(c).exists()]
    return jsonify({"checkpoints": disponibles})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True)
    historial = data.get("historial", [])
    mensaje = (data.get("mensaje") or "").strip()
    checkpoint = data.get("checkpoint") or CANDIDATOS_CHECKPOINT[0]
    params = data.get("params", {}) or {}
    # no_registrar: usado por replay_log_conversaciones.py -- sin esto, cada
    # replay reproduce logs_chat_web/ contra esta misma API y sus respuestas
    # se auto-registran en el log, duplicando sesiones en el siguiente replay
    # (encontrado 2026-09-14: 54 sesiones reales -> 108 tras un par de
    # replays, contaminando la metrica de colapsos).
    no_registrar = bool(data.get("no_registrar", False))

    if not mensaje:
        return jsonify({"error": "mensaje vacio"}), 400

    temperature = float(params.get("temperature", 0.30))
    top_k = int(params.get("top_k", 30))
    max_new_tokens = int(params.get("max_new_tokens", 300))
    repetition_penalty = float(params.get("repetition_penalty", 1.30))
    repetition_window = int(params.get("repetition_window", 60))
    # MAX_HISTORY_TURNS=2 ya estaba en alfa1/config_chat.py (USE_HISTORY=False
    # por defecto) -- lo ignore al construir esta app y mande el historial
    # COMPLETO sin limite. Confirmado con conversaciones reales (2026-09-06):
    # a partir de ~800 tokens acumulados el modelo deja de condicionar en el
    # turno nuevo y colapsa a una respuesta generica identica ("Que tal?")
    # sea cual sea la pregunta -- el corpus SFT no tiene NINGUN ejemplo de
    # cadenas de mas de 2 turnos, asi que mas alla de eso el modelo esta
    # fuera de su distribucion de entrenamiento. Mientras no exista ese
    # corpus, la mitigacion real es no mandarle mas contexto del que sabe
    # manejar.
    max_turnos_historial = int(params.get("max_history_turns", 2))
    if max_turnos_historial >= 0:
        historial = historial[-(max_turnos_historial * 2):] if max_turnos_historial > 0 else []

    modo_agente = bool(data.get("modo_agente", False))
    gen_kwargs = dict(max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
                       repetition_penalty=repetition_penalty, repetition_window=repetition_window)

    # Alfa1-G-4B-it (y cualquier futuro checkpoint HF): rama aparte, ver
    # cargar_modelo_hf/generar_hf. Sin modo agente todavia -- se avisa en vez
    # de fingir un resultado.
    if _es_checkpoint_hf(checkpoint):
        if modo_agente:
            return jsonify({"error": "Modo agente aún no soportado para este checkpoint (solo chat normal)."}), 400
        try:
            model, tok_hf, fin_id = cargar_modelo_hf(checkpoint)
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 400
        t0 = time.time()
        respuesta_limpia, n_generados, paro_en_fin = generar_hf(
            model, tok_hf, fin_id, historial, mensaje, gen_kwargs)
        dt = time.time() - t0
        if not no_registrar:
            registrar_log({
                "id": str(uuid.uuid4()), "timestamp": datetime.now(timezone.utc).isoformat(),
                "checkpoint": checkpoint, "historial_previo": historial,
                "mensaje_usuario": mensaje, "respuesta_tokens_generados": n_generados,
                "paro_en_fin": paro_en_fin, "respuesta_limpia": respuesta_limpia,
                "parametros": {"temperature": temperature, "top_k": top_k,
                              "max_new_tokens": max_new_tokens,
                              "repetition_penalty": repetition_penalty},
                "segundos": round(dt, 2),
            })
        return jsonify({"respuesta": respuesta_limpia, "paro_en_fin": paro_en_fin,
                        "tokens_generados": n_generados, "segundos": round(dt, 2)})

    try:
        tokenizer = cargar_tokenizer()
        model = cargar_modelo(checkpoint)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400

    t0 = time.time()

    if modo_agente:
        # Modo agente: NO encadena turnos ReAct reales con la conversacion
        # normal previa (el SFT de ReAct solo vio un turno "user" con
        # INSTRUCCIONES+Pregunta), pero SI le pasa esos turnos resumidos como
        # texto dentro de la Pregunta para no perder datos mencionados antes
        # (p.ej. un presupuesto) -- ver ejecutar_ciclo_agente / hallazgo v31.
        # El checkpoint decide el formato: v39 usa <tool>/<turno>, el resto
        # el ReAct en texto libre. Usar el bucle equivocado no da error, solo
        # respuestas sin herramienta, asi que la lista es explicita.
        ciclo = (ejecutar_ciclo_agente_v39 if Path(checkpoint).name in CHECKPOINTS_V39
                 else ejecutar_ciclo_agente)
        try:
            respuesta_limpia, traza, tokens_generados = ciclo(
                model, tokenizer, mensaje, gen_kwargs, historial_conversacion=historial)
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 400
        dt = time.time() - t0
        paro_en_fin = None

        if not no_registrar:
            registrar_log({
                "id": str(uuid.uuid4()), "timestamp": datetime.now(timezone.utc).isoformat(),
                "checkpoint": checkpoint, "modo_agente": True,
                "mensaje_usuario": mensaje, "traza_agente": traza,
                "respuesta_limpia": respuesta_limpia, "parametros": gen_kwargs,
                "segundos": round(dt, 2),
            })
        return jsonify({
            "respuesta": respuesta_limpia, "traza_agente": traza,
            "tokens_generados": tokens_generados, "segundos": round(dt, 2),
        })

    prompt = construir_prompt_multiturno(tokenizer, historial, mensaje)
    input_ids = tokenizer.encode(prompt, normalize_chat=False)

    # Si el historial + prompt se pasa del contexto, se recorta por delante
    # (se pierden turnos mas antiguos) para no petar model.block_size. Esto
    # es un tope de SEGURIDAD (evita crashear); el tope real de CALIDAD es
    # max_turnos_historial de arriba, que actua mucho antes.
    if len(input_ids) >= BLOCK_SIZE - max_new_tokens:
        exceso = len(input_ids) - (BLOCK_SIZE - max_new_tokens)
        input_ids = input_ids[exceso:]

    context = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    respuesta_ids = generar(
        model, context,
        max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
        fin_id=tokenizer.fin_id,
        repetition_penalty=repetition_penalty, repetition_window=repetition_window,
    )
    dt = time.time() - t0

    paro_en_fin = tokenizer.fin_id in respuesta_ids
    respuesta_cruda = tokenizer.decode(respuesta_ids, skip_special=False)
    respuesta_limpia = limpiar_respuesta(tokenizer, respuesta_ids)

    if not no_registrar:
        registrar_log({
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checkpoint": checkpoint,
            "historial_previo": historial,
            "mensaje_usuario": mensaje,
            "prompt_tokens": len(input_ids),
            "respuesta_tokens_generados": len(respuesta_ids),
            "paro_en_fin": paro_en_fin,
            "respuesta_cruda": respuesta_cruda,
            "respuesta_limpia": respuesta_limpia,
            "parametros": {
                "temperature": temperature, "top_k": top_k,
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": repetition_penalty,
                "repetition_window": repetition_window,
            },
            "segundos": round(dt, 2),
        })

    return jsonify({
        "respuesta": respuesta_limpia,
        "paro_en_fin": paro_en_fin,
        "tokens_generados": len(respuesta_ids),
        "segundos": round(dt, 2),
    })


if __name__ == "__main__":
    print("Cargando tokenizer y checkpoint por defecto...")
    cargar_tokenizer()
    try:
        cargar_modelo(CANDIDATOS_CHECKPOINT[0])
    except FileNotFoundError as e:
        print(f"AVISO: {e} -- se cargara bajo demanda desde la UI.")
    # 0.0.0.0: accesible desde otros dispositivos de la misma red WiFi
    # (movil), no solo desde esta maquina. Requiere que el firewall de
    # Windows permita el puerto (ver regla creada en PowerShell).
    app.run(host="0.0.0.0", port=5000, debug=False)
