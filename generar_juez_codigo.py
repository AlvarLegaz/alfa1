#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generar_juez_codigo.py
========================
Igual que generar_juez_es_en.py pero para los 15 prompts de codigo Python
(datos/prompts_codigo_python.json), con normalize_chat=False (el codigo no
debe pasar por la normalizacion de chat) y muestreo (no greedy).
"""
import argparse
import gc
import json
from pathlib import Path

import torch

from benchmark_bcc_es import cargar_modelo, generar

AQUI = Path(__file__).resolve().parent
DATOS_DIR = AQUI / "datos"

from generar_juez_es_en import MODELOS  # reutiliza la misma lista de 7 modelos


def cargar_prompts():
    py = json.load(open(DATOS_DIR / "prompts_codigo_python.json", encoding="utf-8"))
    return [{"idioma": "codigo_python", "prompt": p} for p in py]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260912)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = cargar_prompts()
    print(f"{len(prompts)} prompts de codigo")

    resultados = []
    if args.out.exists():
        resultados = json.load(open(args.out, encoding="utf-8"))
        ya_hechos = {(r["modelo"], r["prompt"]) for r in resultados}
        print(f"Reanudando: {len(resultados)} resultados ya existentes")
    else:
        ya_hechos = set()

    for nombre, path, tipo in MODELOS:
        pendientes = [p for p in prompts if (nombre, p["prompt"]) not in ya_hechos]
        if not pendientes:
            print(f"[{nombre}] ya completo, se salta")
            continue

        print(f"\n=== Cargando {nombre} ({path}) ===")
        torch.manual_seed(args.seed)
        if tipo == "nativo":
            model, tokenizer, config, _ck = cargar_modelo(
                path, device, "alfa1.model", "SanchoGPT", "alfa1.tokenizer_bpe", "BPETokenizer")
        else:
            model, tokenizer, config, _ck = cargar_modelo(
                path, device, "adaptador_hf", "ModeloHF", "adaptador_hf", "TokenizerHF")

        for i, item in enumerate(pendientes, 1):
            texto, dt, n_gen, tok_s, _pos = generar(
                model, tokenizer, item["prompt"], device,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                top_k=args.top_k, deterministic=False, normalize_chat=False,
            )
            resultados.append({
                "modelo": nombre, "idioma": "codigo_python", "prompt": item["prompt"],
                "completado": texto, "tokens_generados": n_gen,
            })
            if i % 5 == 0 or i == len(pendientes):
                print(f"  [{nombre}] {i}/{len(pendientes)}")
            with args.out.open("w", encoding="utf-8") as f:
                json.dump(resultados, f, ensure_ascii=False, indent=2)

        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n[OK] {len(resultados)} resultados -> {args.out}")


if __name__ == "__main__":
    main()
