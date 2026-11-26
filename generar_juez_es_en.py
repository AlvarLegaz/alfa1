#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generar_juez_es_en.py
======================
Genera, para cada modelo de la Tabla de resultados del paper, las 30
continuaciones en espanol y las 30 en ingles de datos/prompts_{es,en}.json
(las mismas que usa benchmark_bcc_es.py), con MUESTREO (no greedy: temp 0.8,
top_k 40, igual que la condicion "muestreo" de la valoracion cualitativa
existente) para que el juez LLM vea al modelo en su mejor condicion medida.

Salida: un unico JSON con todos los resultados, cada registro con un campo
"modelo" que identifica quien lo genero.

Uso: python generar_juez_es_en.py --out juez_out/es_en_muestreo.json
"""
import argparse
import gc
import json
from pathlib import Path

import torch

from benchmark_bcc_es import cargar_modelo, generar

AQUI = Path(__file__).resolve().parent
DATOS_DIR = AQUI / "datos"

ALFA1_ANNEALING = r"C:\Users\alegaz\Desktop\PROYECTO SANCHO\MODELOS\Alfa1_05B_v2\checkpoints_midtrain\alpha_1_annealing_05B.pt"
QWEN25_BASE = r"C:\Users\alegaz\Desktop\PROYECTO SANCHO\MODELOS\qwen2.5_0.5b_base\qwen2.5-0.5b-base"

MODELOS = [
    ("alfa1", ALFA1_ANNEALING, "nativo"),
    ("qwen15_05b", "Qwen/Qwen1.5-0.5B", "hf"),
    ("qwen25_05b", QWEN25_BASE, "hf"),
    ("qwen3_06b", "Qwen/Qwen3-0.6B", "hf"),
    ("pythia410m", "EleutherAI/pythia-410m", "hf"),
    ("smollm2_360m", "HuggingFaceTB/SmolLM2-360M", "hf"),
    ("salamandra_2b", "BSC-LT/salamandra-2b", "hf"),
]


def cargar_prompts():
    es = json.load(open(DATOS_DIR / "prompts_es.json", encoding="utf-8"))
    en = json.load(open(DATOS_DIR / "prompts_en.json", encoding="utf-8"))
    items = [{"idioma": "es", "prompt": p} for p in es]
    items += [{"idioma": "en", "prompt": p} for p in en]
    return items


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
    print(f"{len(prompts)} prompts ({sum(1 for p in prompts if p['idioma']=='es')} es / "
          f"{sum(1 for p in prompts if p['idioma']=='en')} en)")

    resultados = []
    if args.out.exists():
        resultados = json.load(open(args.out, encoding="utf-8"))
        ya_hechos = {(r["modelo"], r["idioma"], r["prompt"]) for r in resultados}
        print(f"Reanudando: {len(resultados)} resultados ya existentes")
    else:
        ya_hechos = set()

    for nombre, path, tipo in MODELOS:
        pendientes = [p for p in prompts if (nombre, p["idioma"], p["prompt"]) not in ya_hechos]
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
                top_k=args.top_k, deterministic=False, normalize_chat=True,
            )
            resultados.append({
                "modelo": nombre, "idioma": item["idioma"], "prompt": item["prompt"],
                "completado": texto, "tokens_generados": n_gen,
            })
            if i % 10 == 0 or i == len(pendientes):
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
