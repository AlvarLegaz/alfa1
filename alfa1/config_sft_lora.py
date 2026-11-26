"""
config_sft_combinado.py
==========================

Config para la corrida COMBINADA: Alpaca + json_variado + domotico, todo
mezclado en la misma carpeta de datos, entrenado en una sola pasada sobre
el checkpoint BASE (sin fusionar nada previamente -- ni Alpaca ni JSON
variado estan ya fusionados aqui, todo se aprende junto en este LoRA).

Por que asi y no encadenando fusiones: cada merge_and_unload() introduce
una pequeña aproximacion numerica; entrenar todo junto en una pasada evita
acumularlas, y con shuffle=True a nivel de ejemplo (ver SFTDataset en
sft_lora_combinado.py) la mezcla es correcta sin tener que intercalar nada
a mano.

Capacidad de LoRA ampliada (r=32 + MLP) respecto a la corrida domotica en
solitario que colapso -- alli el problema era demasiada capacidad para
~750 ejemplos; aqui hay ~40k+ ejemplos combinados, la misma capacidad ya
no esta sobredimensionada para el volumen de datos.
"""

import os as _os

# Rutas RELATIVAS AL DIRECTORIO DESDE EL QUE LANZAS PYTHON (Alfa1_05B),
# no a donde vive este fichero -- este config puede estar dentro de la
# carpeta alfa1\ (paquete), pero checkpoints\ y Alfa1_sft_data\ estan un
# nivel arriba, directamente en Alfa1_05B\. Como siempre lanzas los
# scripts desde ahi, las rutas relativas simples son mas fiables aqui que
# calcular a partir de __file__.

# TODO ADAPTAR si el checkpoint base no esta exactamente en esta ruta
CHECKPOINT_PATH = _os.path.join("checkpoints", "alpha_1_pretrain_05B_best_iter52000.pt")
TOKENIZER_PATH = _os.path.join("checkpoints", "tokenizer_state.json")

# Arquitectura -- confirmada en los logs de este mismo checkpoint en este
# entorno (mismos valores que en EDGE, mismo checkpoint).
_arch = {
    "vocab_size": 48000,
    "block_size": 512,
    "n_embd": 1536,
    "n_head": 24,
    "n_kv_head": 6,
    "n_layer": 18,
    "ffn_hidden": 4096,
    "weight_tying": True,
}


class Config:
    ckpt_path = CHECKPOINT_PATH
    tokenizer_path = TOKENIZER_PATH

    vocab_size = _arch["vocab_size"]
    block_size = _arch["block_size"]
    n_embd = _arch["n_embd"]
    n_head = _arch["n_head"]
    n_kv_head = _arch["n_kv_head"]
    n_layer = _arch["n_layer"]
    ffn_hidden = _arch["ffn_hidden"]
    weight_tying = _arch["weight_tying"]

    # -----------------------------------------------------------------
    # Datos -- carpeta con TODOS los .txt mezclados: alpaca.txt,
    # json_variado.txt, y dataset_domotico.txt (este ultimo repetido
    # varias veces dentro de la carpeta -- ver nota mas abajo -- para que
    # no se diluya frente a los ~40k de Alpaca).
    # -----------------------------------------------------------------
    sft_data_dir = "Alfa1_sft_data"  # ya contiene alpaca.txt, json_variado.txt, dataset_domotico.txt
    val_ratio = 0.05  # con ~40k+ ejemplos, 5% ya da un val set grande de sobra

    # -----------------------------------------------------------------
    # Cuantizacion
    # -----------------------------------------------------------------
    use_4bit = False

    # -----------------------------------------------------------------
    # LoRA -- capacidad ampliada (r=32 + MLP), justificada por el volumen
    # de datos combinado (a diferencia de la corrida domotica en solitario
    # con ~750 ejemplos, donde esta misma capacidad colapso).
    # -----------------------------------------------------------------
    lora_r = 32
    lora_alpha = 64
    lora_dropout = 0.05
    lora_bias = "none"
    lora_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    # -----------------------------------------------------------------
    # Entrenamiento -- ajusta batch_size/grad_accum segun la VRAM real de
    # la 5090 (32GB); estos valores son un punto de partida conservador,
    # puedes subir batch_size si ves margen de sobra en nvidia-smi.
    # -----------------------------------------------------------------
    device = "cuda"
    seed = 42
    batch_size = 16
    grad_accum_steps = 3          # batch efectivo 48
    epochs = 3                    
    learning_rate = 2e-4
    min_lr = 2e-5
    weight_decay = 0.01
    warmup_steps = 100            # mas pasos de warmup, dataset mucho mas grande que antes
    max_grad_norm = 1.0
    mixed_precision_dtype = "float16"

    # -----------------------------------------------------------------
    # Logging / checkpoints
    # -----------------------------------------------------------------
    log_every = 20
    save_every_n_steps = 500      # con un dataset grande, guarda tambien a mitad de epoca, no solo al final
    save_every_epoch = True
    output_dir = "Alfa1_combinado_lora_out"


cfg = Config()

print(f"[config_sft_combinado] ckpt_path={cfg.ckpt_path}")
print(f"[config_sft_combinado] sft_data_dir={cfg.sft_data_dir}")
print(f"[config_sft_combinado] LoRA: r={cfg.lora_r}, alpha={cfg.lora_alpha}, "
      f"target_modules={cfg.lora_target_modules}")