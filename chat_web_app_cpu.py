# -*- coding: utf-8 -*-
"""Lanzador de chat_web_app.py forzando CPU de verdad. CUDA_VISIBLE_DEVICES=""
no bastaba (crash de la sesion 2026-09-18: el servidor cargo en cuda igual
mientras Llama entrenaba, provocando un CUDA error que tumbo el entreno).
Aqui se parchea torch.cuda.is_available() ANTES de importar chat_web_app,
que es lo que lee alfa1/config_pretrain.py para fijar DEVICE -- sin
ambiguedad de variables de entorno."""
import torch
torch.cuda.is_available = lambda: False

import chat_web_app  # noqa: E402  (ejecuta el bloque __main__ al importar? no)

if __name__ == "__main__":
    chat_web_app.cargar_tokenizer()
    try:
        chat_web_app.cargar_modelo(chat_web_app.CANDIDATOS_CHECKPOINT[0])
    except FileNotFoundError as e:
        print(f"AVISO: {e} -- se cargara bajo demanda desde la UI.")
    chat_web_app.app.run(host="0.0.0.0", port=5000, debug=False)
