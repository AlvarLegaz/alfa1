# Alfa 1 (0.5B)

Código de entrenamiento de **Alfa 1**, un modelo de lenguaje de 0,5B parámetros en español e inglés, desarrollado por **Minaya Research** (antes Minaya AI).

Los pesos están publicados en Hugging Face (Apache 2.0, safetensors y GGUF):

- Base: [alvar-legaz/Alfa-1-0.5B](https://huggingface.co/alvar-legaz/Alfa-1-0.5B)
- Instruct: [alvar-legaz/Alfa-1-0.5B-Instruct](https://huggingface.co/alvar-legaz/Alfa-1-0.5B-Instruct)

## Contenido

| Ruta | Qué es |
|---|---|
| `alfa1/` | Modelo, tokenizer BPE y configuraciones (pretrain, annealing, SFT, LoRA, DPO, GRPO) |
| `train_pretrain_v2.py`, `train_annealing.py`, `train_sft_bpe.py` | Entrenamiento: preentreno, annealing y SFT |
| `adaptador_hf.py`, `exportar_release_hf.py`, `exportar_gguf.py` | Conversión a Hugging Face (Llama) y a GGUF |
| `chat_*.py`, `chat_web_static/` | Chat por consola y web local |
| `benchmark_*.py`, `generar_juez_*.py` | Evaluación (BCC-ES, benchmarks estándar, juez LLM) |
| `datos/prompts_*.json` | Prompts de control |

## Notas

- Las rutas de datos y checkpoints de `alfa1/config_*.py` son las de la máquina de entrenamiento original. Ajústalas a tu entorno.
- No se incluyen los datos de entrenamiento, los checkpoints ni el banco BCC-ES (`datos/bcc_es_*.json`), que necesita `benchmark_bcc_es.py`.

## Licencia

Apache 2.0 (ver `LICENSE`).
