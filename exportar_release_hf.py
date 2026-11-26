#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exportar_release_hf.py  (2026-11-22)
====================================
Cierre de la epoca Alfa 1: prepara en local los dos modelos finales para
publicarlos en Hugging Face (NO sube nada).

  base      -> checkpoints_annealing/alpha_1_annealing_05B.pt
               (fin del annealing, it 14.976, best_val_loss 1,1322; el "Alfa 1" del paper)
  instruct  -> checkpoints_sft/alpha_1_sft_release_v43_best.pt
               (corpus v43, 110.061 ejemplos; el "Alfa 1 Instruct" del paper)

Para cada uno deja en gguf_export/release/:
  <Nombre>/          repo HF (LlamaForCausalLM, safetensors bf16 + tokenizer)
  <Nombre>-GGUF/     <Nombre>-F16.gguf, -Q8_0.gguf, -Q4_K_M.gguf

Reutiliza las funciones de exportar_gguf.py (mismo camino ya verificado:
SanchoGPT.export_to_hf -> convert_hf_to_gguf.py del llama.cpp parcheado
con pre=qwen2). Solo cambia rutas, nombres y la plantilla de chat (acepta
ademas un mensaje "system", que se antepone al primer turno de usuario,
igual que en el SFT de agente). No usa GPU.

Uso:
    python exportar_release_hf.py --modelo base
    python exportar_release_hf.py --modelo instruct
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

import exportar_gguf as eg
from alfa1.tokenizer_bpe import BPETokenizer

AQUI = Path(__file__).resolve().parent
# La ruta original del tokenizador (Generar_corpus\corpus_80Bt\...) ya no existe;
# esta copia es identica (md5 de tokenizer_json = el de hf_alpha1_it_v32b).
TOK_STATE = AQUI / "checkpoints_sft" / "tokenizer_state.json"
REL = AQUI / "gguf_export" / "release"
QUANTIZE = AQUI / "gguf_export" / "llamacpp_bin" / "llama-quantize.exe"

MODELOS = {
    "base": {
        "ckpt": AQUI / "checkpoints_annealing" / "alpha_1_annealing_05B.pt",
        "nombre": "Alfa-1-0.5B",
        "eos_token": "<|endofstory|>",
        "chat": False,
    },
    "instruct": {
        "ckpt": AQUI / "checkpoints_sft" / "alpha_1_sft_release_v43_best.pt",
        "nombre": "Alfa-1-0.5B-Instruct",
        "eos_token": "<FIN>",
        "chat": True,
    },
}

# Formato del SFT: <USER>texto<ASSISTANT>respuesta<FIN>. Las instrucciones de
# sistema (y la lista de herramientas en modo agente) iban DENTRO del primer
# <USER>, seguidas de una linea en blanco: el "system" se renderiza asi.
CHAT_TEMPLATE = (
    "{% set ns = namespace(sys='') %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}{% set ns.sys = message['content'] + '\n\n' %}"
    "{% elif message['role'] == 'user' %}<USER>{{ ns.sys }}{{ message['content'] }}{% set ns.sys = '' %}"
    "{% elif message['role'] == 'assistant' %}<ASSISTANT>{{ message['content'] }}<FIN>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<ASSISTANT>{% endif %}"
)


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=list(MODELOS), required=True)
    ap.add_argument("--sin-gguf", action="store_true")
    args = ap.parse_args()
    m = MODELOS[args.modelo]
    hf_dir = REL / m["nombre"]
    gguf_dir = REL / f"{m['nombre']}-GGUF"
    hf_dir.mkdir(parents=True, exist_ok=True)
    gguf_dir.mkdir(parents=True, exist_ok=True)

    print(f"1/4 tokenizer <- {TOK_STATE}")
    tok = BPETokenizer()
    tok.load_state(json.loads(TOK_STATE.read_text(encoding="utf-8")))

    print(f"2/4 modelo <- {m['ckpt']}")
    modelo = eg.cargar_modelo(m["ckpt"])

    print(f"3/4 repo HF -> {hf_dir}")
    modelo.export_to_hf(str(hf_dir), dtype=torch.bfloat16)
    del modelo
    eg.CHAT_TEMPLATE = CHAT_TEMPLATE
    ids = eg.escribir_metadatos(hf_dir, tok, m["eos_token"], m["chat"])
    print(f"    especiales: {ids}  EOS={m['eos_token']}")
    if args.sin_gguf:
        return

    print("4/4 GGUF")
    conv = eg.LLAMACPP / "convert_hf_to_gguf.py"
    f16 = gguf_dir / f"{m['nombre']}-F16.gguf"
    run([sys.executable, conv, hf_dir, "--outfile", f16, "--outtype", "f16"])
    run([sys.executable, conv, hf_dir, "--outfile", gguf_dir / f"{m['nombre']}-Q8_0.gguf", "--outtype", "q8_0"])
    run([QUANTIZE, f16, gguf_dir / f"{m['nombre']}-Q4_K_M.gguf", "Q4_K_M"])
    for g in sorted(gguf_dir.glob("*.gguf")):
        print(f"    {g.name}: {g.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
