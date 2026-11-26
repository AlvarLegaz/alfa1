"""
chat_lora_console.py — Prueba el modelo SFT-LoRA (fusionado) directamente en
PyTorch, en consola, SIN pasar por la exportación a GGUF/LM Studio.

Objetivo: aislar si los problemas de generación (texto sin sentido, no para
en <FIN>) vienen del entrenamiento/LoRA, o de algo en el pipeline de
exportación (chat template, tokenizer.json, metadatos GGUF, etc.)

Uso:
    python chat_lora_console.py
    (edita LORA_ADAPTER_DIR abajo para probar epoch0, epoch1, epoch2...)
"""

import torch
import torch.nn.functional as F
from peft import PeftModel

from alfa1.config_sft_lora import cfg
from alfa1.model import SanchoGPT
from alfa1.tokenizer_bpe import BPETokenizer

try:
    from alfa1.config_chat import (
        MAX_NEW_TOKENS_CHAT,
        TEMPERATURE_CHAT,
        TOP_K_CHAT,
        ASSISTANT_NAME,
        REPETITION_PENALTY,
        REPETITION_WINDOW,
    )
except ImportError:
    # Valores ajustados tras el diagnostico de eval_estabilidad_fin.py:
    # el modelo necesita mas margen de tokens (150) y menos temperatura/top_k
    # para reducir divagacion en respuestas abiertas (listas de N items).
    MAX_NEW_TOKENS_CHAT = 500
    TEMPERATURE_CHAT = 0.30
    TOP_K_CHAT = 30
    ASSISTANT_NAME = "Nada"
    REPETITION_PENALTY = 1.15  # 1.0 = sin penalización; bug corregido: ya solo penaliza tokens generados, no el prompt
    REPETITION_WINDOW = 60  # cuántos tokens recientes se penalizan (0 = toda la secuencia)

LORA_ADAPTER_DIR = "Alfa1_sft_lora_out/epoch2"  # cambia aquí para probar otra epoch
USE_LORA = True  # True = probar el LoRA fusionado; False = solo el checkpoint base
DEVICE = cfg.device


def cargar_tokenizer(path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)
    tokenizer = BPETokenizer()
    tokenizer.load_state(state)
    return tokenizer


def cargar_modelo_lora():
    base_model = SanchoGPT(
        vocab_size=cfg.vocab_size,
        block_size=cfg.block_size,
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        dropout=0.0,
        device="cpu",
        n_kv_head=cfg.n_kv_head,
        tie_weights=cfg.weight_tying,
        ffn_hidden=cfg.ffn_hidden,
    )

    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    base_model.load_state_dict(state_dict, strict=True)

    if USE_LORA:
        print(f"Cargando adaptador LoRA de: {LORA_ADAPTER_DIR}")
        peft_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
        merged_model = peft_model.merge_and_unload()
    else:
        print("Probando el checkpoint BASE, sin LoRA (diagnóstico).")
        merged_model = base_model

    merged_model.to(DEVICE)
    merged_model.eval()
    return merged_model


def aplicar_penalizacion_repeticion(logits, generados_recientes, penalty):
    """Estilo CTRL (Keskar et al.): para cada token ya generado, si su logit
    es positivo se divide por `penalty` (lo hace menos probable), si es
    negativo se multiplica (también lo empuja hacia abajo)."""
    if penalty == 1.0 or not generados_recientes:
        return logits
    for tid in set(generados_recientes):
        val = logits[0, tid]
        if val > 0:
            logits[0, tid] = val / penalty
        else:
            logits[0, tid] = val * penalty
    return logits


@torch.no_grad()
def generate_con_penalizacion(model, idx, max_new_tokens, temperature, top_k,
                               fin_id, repetition_penalty=1.0, repetition_window=0):
    """Reimplementa SanchoGPT.generate() (mismo manejo de kv_cache/RoPE),
    añadiendo penalización de repetición antes de muestrear. No toca
    model.py para no afectar a otros scripts (entrenamiento, exportación).

    La penalización solo se aplica sobre tokens GENERADOS por el modelo,
    nunca sobre los tokens del prompt del usuario (bug corregido: antes
    `historial` incluía idx[0] completo, penalizando tambien palabras que
    el propio usuario habia escrito en su pregunta)."""
    idx_cond = idx[:, -model.block_size:]
    pos_offset = 0
    logits, _, kv_caches = model(idx_cond, use_cache=True, pos_offset=pos_offset)
    pos_offset = idx_cond.shape[1]
    logits = logits[:, -1, :].clone()

    generados = []  # solo tokens producidos por el modelo, no el prompt

    for _ in range(max_new_tokens):
        historial = generados
        if repetition_window > 0:
            historial = historial[-repetition_window:]
        logits = aplicar_penalizacion_repeticion(logits, historial, repetition_penalty)

        logits = logits / temperature
        if top_k is not None:
            values, indices = torch.topk(logits, top_k)
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(1, indices, values)
            logits = filtered
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)
        generados.append(idx_next.item())

        if fin_id is not None and idx_next.item() == fin_id:
            break

        if pos_offset >= model.block_size:
            kv_caches = [
                (k[:, :, -(model.block_size - 1):, :], v[:, :, -(model.block_size - 1):, :])
                for (k, v) in kv_caches
            ]
            pos_offset = model.block_size - 1

        logits, _, kv_caches = model(
            idx_next, use_cache=True, kv_caches=kv_caches, pos_offset=pos_offset,
        )
        pos_offset += 1
        logits = logits[:, -1, :].clone()

    return idx


def construir_prompt(tokenizer, mensaje):
    """Mismo formato exacto que el entrenamiento: sin espacios extra."""
    return f"{tokenizer.user_token}{mensaje}{tokenizer.assistant_token}"


def limpiar_respuesta(tokenizer, respuesta_ids):
    while respuesta_ids and respuesta_ids[0] == tokenizer.fin_id:
        respuesta_ids = respuesta_ids[1:]
    if tokenizer.fin_id in respuesta_ids:
        respuesta_ids = respuesta_ids[:respuesta_ids.index(tokenizer.fin_id)]

    respuesta = tokenizer.decode(respuesta_ids, skip_special=True).strip()

    cortes = [
        "\n\n", tokenizer.user_token, tokenizer.assistant_token,
        tokenizer.fin_token, "<USER>", "<ASSISTANT>", "<FIN>",
    ]
    for corte in cortes:
        if corte in respuesta:
            respuesta = respuesta.split(corte, 1)[0].strip()
    return respuesta


def responder(model, tokenizer, mensaje):
    prompt = construir_prompt(tokenizer, mensaje)
    input_ids = tokenizer.encode(prompt, normalize_chat=False)

    print(f"[debug] tokens de prompt: {len(input_ids)}")

    context = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        output = generate_con_penalizacion(
            model,
            context,
            max_new_tokens=MAX_NEW_TOKENS_CHAT,
            temperature=TEMPERATURE_CHAT,
            top_k=TOP_K_CHAT,
            fin_id=tokenizer.fin_id,
            repetition_penalty=REPETITION_PENALTY,
            repetition_window=REPETITION_WINDOW,
        )

    respuesta_ids = output[0].tolist()[len(input_ids):]
    n_generados = len(respuesta_ids)
    paro_en_fin = tokenizer.fin_id in respuesta_ids
    print(f"[debug] tokens generados: {n_generados} | ¿paró en <FIN>?: {paro_en_fin}")

    return limpiar_respuesta(tokenizer, respuesta_ids)


def main():
    tokenizer = cargar_tokenizer(cfg.tokenizer_path)
    model = cargar_modelo_lora()

    print(f"\n{ASSISTANT_NAME} listo (consola, sin GGUF). Escribe 'salir' para terminar.\n")
    while True:
        mensaje = input("Tú: ").strip()
        if mensaje.lower() in ("salir", "exit", "quit"):
            break
        if not mensaje:
            continue
        respuesta = responder(model, tokenizer, mensaje)
        print(f"{ASSISTANT_NAME}: {respuesta}\n")


if __name__ == "__main__":
    main()