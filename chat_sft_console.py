"""
chat_sft_console.py — Prueba el modelo FULL SFT (sin LoRA) directamente en
PyTorch, en consola, SIN pasar por la exportación a GGUF/LM Studio.

Adaptado de chat_lora_console.py: como el full SFT ya tiene los pesos
entrenados dentro del checkpoint, NO hay adaptador PEFT que fusionar --
se carga alpha_1_sft_full_best.pt directamente.

Objetivo: probar el modelo entrenado con train_sft_bpe.py (el best guardado
por early stopping) y ver si para en <FIN>, si divaga en respuestas abiertas,
si mantiene coherencia, etc.

Uso:
    python chat_sft_console.py
    (edita SFT_CHECKPOINT abajo para probar otro checkpoint: _best o el final)
"""

import json

import torch
import torch.nn.functional as F

from alfa1.config_sft import (
    N_EMBD, N_HEAD, N_LAYER, N_KV_HEAD, FFN_HIDDEN, TIE_WEIGHTS, ROPE_THETA,
    BLOCK_SIZE, DEVICE, TOKENIZER_STATE_PATH,
)
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
    # Mismos valores por defecto que en chat_lora_console.py.
    MAX_NEW_TOKENS_CHAT = 500
    TEMPERATURE_CHAT = 0.30
    TOP_K_CHAT = 30
    ASSISTANT_NAME = "Nada"
    REPETITION_PENALTY = 1.15   # 1.0 = sin penalización
    REPETITION_WINDOW = 60      # cuántos tokens recientes se penalizan (0 = todos)

# Checkpoint del full SFT. Por defecto el best (el que dejo el early stopping).
# Cambia a "...alpha_1_sft_full.pt" para probar el checkpoint final.
SFT_CHECKPOINT = r"C:\Users\alegaz\Desktop\PROYECTO SANCHO\MODELOS\Alfa1_05B\checkpoints\alpha_1_sft_full_best.pt"


def cargar_tokenizer(path):
    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)
    tokenizer = BPETokenizer()
    tokenizer.load_state(state)
    return tokenizer


def cargar_modelo_sft(tokenizer):
    model = SanchoGPT(
        vocab_size=tokenizer.vocab_size(),
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=0.0,
        device="cpu",
        n_kv_head=N_KV_HEAD,
        tie_weights=TIE_WEIGHTS,
        rope_theta=ROPE_THETA,
        ffn_hidden=FFN_HIDDEN,
    )

    print(f"Cargando checkpoint full SFT: {SFT_CHECKPOINT}")
    ckpt = torch.load(SFT_CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))

    # torch.compile antepone "_orig_mod." a las claves; si el checkpoint se
    # guardo desde el modelo compilado, lo quitamos para que case con el
    # SanchoGPT sin compilar. (train_sft_bpe.py guarda desde model.state_dict()
    # que puede o no estar compilado segun COMPILE_MODEL.)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def aplicar_penalizacion_repeticion(logits, generados_recientes, penalty):
    """Estilo CTRL (Keskar et al.): para cada token ya generado, si su logit
    es positivo se divide por `penalty`, si es negativo se multiplica."""
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
    añadiendo penalización de repetición antes de muestrear. La penalización
    solo se aplica sobre tokens GENERADOS por el modelo, nunca sobre el prompt."""
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
    tokenizer = cargar_tokenizer(TOKENIZER_STATE_PATH)
    model = cargar_modelo_sft(tokenizer)

    print(f"\n{ASSISTANT_NAME} listo (full SFT, consola, sin GGUF). Escribe 'salir' para terminar.\n")
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
