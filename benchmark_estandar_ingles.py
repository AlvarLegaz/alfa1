#!/usr/bin/env python3
"""
benchmark_estandar_ingles.py
-----------------------------
Bateria estandar de sentido comun/razonamiento en ingles (estilo Open LLM
Leaderboard / GPT4All-eval, la misma que reporta TinyLlama en su EVAL.md):
HellaSwag, PIQA, WinoGrande, ARC-Easy, ARC-Challenge, BoolQ, OpenBookQA.

Reutiliza SIN DUPLICAR la carga de modelo/tokenizer y el scorer de
log-prob-por-token de benchmark_bcc_es.py (cargar_modelo,
logprob_continuacion) -- mismo protocolo de evaluacion (eleccion forzada por
log-prob media por token, length-normalized) que ya se usa para BCC-ES y
para las competencias, asi que los numeros son comparables entre si dentro
de este proyecto.

Por defecto evalua una MUESTRA (--n-muestras, default 200 por benchmark) en
vez del split de validacion completo -- HellaSwag por ejemplo tiene 10K
ejemplos, y a la velocidad de un modelo de 0.5B en CPU/GPU compartida con el
entrenamiento eso tardaria horas. Con muestra aleatoria fija (--seed) el
resultado es reproducible y ya da una estimacion razonable del intervalo de
confianza a 200 ejemplos (~±7 puntos con 95% de confianza para un
porcentaje cercano al 50%).
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_bcc_es import cargar_modelo, logprob_continuacion  # noqa: E402

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402


def _elige_opcion(model, tokenizer, contexto, opciones, device):
    """Devuelve el indice de la opcion con mayor log-prob media por token."""
    lps = [logprob_continuacion(model, tokenizer, contexto, op, device)[0]
           for op in opciones]
    return max(range(len(lps)), key=lambda i: lps[i]), lps


@torch.no_grad()
def eval_hellaswag(model, tokenizer, device, n, rng):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    idxs = rng.sample(range(len(ds)), min(n, len(ds)))
    ok = 0
    for i in idxs:
        ex = ds[i]
        ctx = ex["ctx"]
        opciones = [" " + e for e in ex["endings"]]
        label = int(ex["label"])
        pred, _ = _elige_opcion(model, tokenizer, ctx, opciones, device)
        ok += int(pred == label)
    return ok, len(idxs)


@torch.no_grad()
def eval_piqa(model, tokenizer, device, n, rng):
    ds = load_dataset("gimmaru/piqa", split="validation")
    idxs = rng.sample(range(len(ds)), min(n, len(ds)))
    ok = 0
    for i in idxs:
        ex = ds[i]
        ctx = f"Question: {ex['goal']}\nAnswer:"
        opciones = [" " + ex["sol1"], " " + ex["sol2"]]
        label = int(ex["label"])
        pred, _ = _elige_opcion(model, tokenizer, ctx, opciones, device)
        ok += int(pred == label)
    return ok, len(idxs)


@torch.no_grad()
def eval_winogrande(model, tokenizer, device, n, rng):
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
    idxs = rng.sample(range(len(ds)), min(n, len(ds)))
    ok = 0
    usados = 0
    for i in idxs:
        ex = ds[i]
        sentence = ex["sentence"]
        if "_" not in sentence:
            continue
        antes, despues = sentence.split("_", 1)
        opciones_ctx = [antes + ex["option1"], antes + ex["option2"]]
        label = int(ex["answer"]) - 1  # "1"/"2" -> 0/1
        lps = [logprob_continuacion(model, tokenizer, c, despues, device)[0]
               for c in opciones_ctx]
        pred = max(range(len(lps)), key=lambda k: lps[k])
        ok += int(pred == label)
        usados += 1
    return ok, usados


@torch.no_grad()
def eval_arc(model, tokenizer, device, n, rng, subset):
    ds = load_dataset("allenai/ai2_arc", subset, split="test")
    idxs = rng.sample(range(len(ds)), min(n, len(ds)))
    ok = 0
    usados = 0
    for i in idxs:
        ex = ds[i]
        labels = ex["choices"]["label"]
        textos = ex["choices"]["text"]
        if ex["answerKey"] not in labels:
            continue
        label = labels.index(ex["answerKey"])
        ctx = f"Question: {ex['question']}\nAnswer:"
        opciones = [" " + t for t in textos]
        pred, _ = _elige_opcion(model, tokenizer, ctx, opciones, device)
        ok += int(pred == label)
        usados += 1
    return ok, usados


@torch.no_grad()
def eval_boolq(model, tokenizer, device, n, rng):
    ds = load_dataset("google/boolq", split="validation")
    idxs = rng.sample(range(len(ds)), min(n, len(ds)))
    ok = 0
    for i in idxs:
        ex = ds[i]
        ctx = f"{ex['passage']}\nQuestion: {ex['question']}?\nAnswer:"
        opciones = [" no", " yes"]
        label = int(ex["answer"])  # True->1 (yes), False->0 (no)
        pred, _ = _elige_opcion(model, tokenizer, ctx, opciones, device)
        ok += int(pred == label)
    return ok, len(idxs)


@torch.no_grad()
def eval_openbookqa(model, tokenizer, device, n, rng):
    ds = load_dataset("allenai/openbookqa", "main", split="test")
    idxs = rng.sample(range(len(ds)), min(n, len(ds)))
    ok = 0
    usados = 0
    for i in idxs:
        ex = ds[i]
        labels = ex["choices"]["label"]
        textos = ex["choices"]["text"]
        if ex["answerKey"] not in labels:
            continue
        label = labels.index(ex["answerKey"])
        ctx = f"Question: {ex['question_stem']}\nAnswer:"
        opciones = [" " + t for t in textos]
        pred, _ = _elige_opcion(model, tokenizer, ctx, opciones, device)
        ok += int(pred == label)
        usados += 1
    return ok, usados


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-modulo", default="alfa1.model")
    ap.add_argument("--model-clase", default="SanchoGPT")
    ap.add_argument("--tokenizer-modulo", default="alfa1.tokenizer_bpe")
    ap.add_argument("--tokenizer-clase", default="BPETokenizer")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-muestras", type=int, default=200,
                     help="Ejemplos por benchmark (muestra, no el split completo)")
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--solo", nargs="*", default=None,
                     help="Subconjunto de benchmarks a correr (por defecto todos): "
                          "hellaswag piqa winogrande arc_easy arc_challenge boolq openbookqa")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cargando checkpoint {args.checkpoint} en {device}...")
    model, tokenizer, config, checkpoint = cargar_modelo(
        args.checkpoint, device, args.model_modulo, args.model_clase,
        args.tokenizer_modulo, args.tokenizer_clase,
    )
    it = checkpoint.get("iteracion", "?")
    print(f"Checkpoint cargado (iteracion={it}). Muestra por benchmark: {args.n_muestras}\n")

    benchmarks = {
        "hellaswag": lambda: eval_hellaswag(model, tokenizer, device, args.n_muestras, random.Random(args.seed)),
        "piqa": lambda: eval_piqa(model, tokenizer, device, args.n_muestras, random.Random(args.seed)),
        "winogrande": lambda: eval_winogrande(model, tokenizer, device, args.n_muestras, random.Random(args.seed)),
        "arc_easy": lambda: eval_arc(model, tokenizer, device, args.n_muestras, random.Random(args.seed), "ARC-Easy"),
        "arc_challenge": lambda: eval_arc(model, tokenizer, device, args.n_muestras, random.Random(args.seed), "ARC-Challenge"),
        "boolq": lambda: eval_boolq(model, tokenizer, device, args.n_muestras, random.Random(args.seed)),
        "openbookqa": lambda: eval_openbookqa(model, tokenizer, device, args.n_muestras, random.Random(args.seed)),
    }
    a_correr = args.solo if args.solo else list(benchmarks.keys())

    resultados = {}
    print(f"{'benchmark':<16}{'aciertos':>10}{'%':>8}")
    print("-" * 34)
    for nombre in a_correr:
        try:
            ok, n = benchmarks[nombre]()
        except Exception as e:
            print(f"{nombre:<16} ERROR: {e!r}")
            continue
        pct = ok / n * 100 if n else float("nan")
        resultados[nombre] = (ok, n, pct)
        print(f"{nombre:<16}{ok:>4}/{n:<5}{pct:>7.1f}%")

    if resultados:
        media = sum(r[2] for r in resultados.values()) / len(resultados)
        print("-" * 34)
        print(f"{'MEDIA':<16}{'':>10}{media:>7.1f}%")


if __name__ == "__main__":
    main()
