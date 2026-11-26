#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exportar_gguf.py
================
Exporta a GGUF (para Ollama / llama.cpp) los dos checkpoints full-finetune:

  --modelo base      -> checkpoints_midtrain/alpha_1_annealing_05B_best.pt
                        (EOS = <|endofstory|>, sin chat template)
  --modelo instruct  -> D:\\alpha_1_05B_it_v16.pt
                        (EOS = <FIN>, con chat template <USER>/<ASSISTANT>/<FIN>
                         y los tokens de conversacion marcados como especiales)

Clave: el <FIN> del instruct se mapea a EOS en el propio GGUF
(tokenize.ggml.eos_token_id), asi que Ollama corta la generacion sin
necesidad de PARAMETER stop ni adaptaciones. Igual se deja el Modelfile con
stop "<FIN>"/"<USER>" como cinturon y tirantes.

Pipeline: SanchoGPT.export_to_hf() -> repo HF LlamaForCausalLM -> se le
anaden tokenizer.json / tokenizer_config.json / generation_config.json con
los IDs reales -> convert_hf_to_gguf.py.

No usa GPU. Uso:
    python exportar_gguf.py --modelo instruct --outtype q8_0
    python exportar_gguf.py --modelo base     --outtype q8_0
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from alfa1.model import SanchoGPT
from alfa1.tokenizer_bpe import BPETokenizer
from alfa1.config_annealing import (
    N_EMBD, N_HEAD, N_LAYER, N_KV_HEAD, FFN_HIDDEN, TIE_WEIGHTS, ROPE_THETA,
    BLOCK_SIZE, BPE_VOCAB_SIZE,
)

AQUI = Path(__file__).resolve().parent
TOK_STATE = Path(r"C:\Users\alegaz\Desktop\PROYECTO SANCHO\Generar_corpus"
                 r"\corpus_80Bt\bin_alfa1\tokenizer_state.json")
LLAMACPP = Path(r"C:\Users\alegaz\Desktop\PROYECTO SANCHO\MODELOS\Alfa1_05B\llama.cpp")
OUT_DIR = AQUI / "gguf_export"

MODELOS = {
    # "instruct" = referencia vigente. ADOPTADO v30 (2026-09-13): agente ReAct
    # de 4 estados (KNOW/SEARCH/TOOL/ABSTAIN) sin regresion en el benchmark de
    # 316 (seguridad 10/10, identidad 20/20, JSON 12/12) ni en el scoring
    # objetivo de general_calculo/general_factual (32 items verificables:
    # v24 21/32, v30 22/32, 1 regresion aislada en gc01 descartada como ruido
    # de muestreo tras 12 repeticiones limpias). v24 se conserva intacto en
    # Ollama como "alpha1-v24" y en D:\alpha_1_05B_it_v24.pt por si hace falta
    # revertir. Historial anterior (v16-v26) en Benchmark_SFT/resultados/.
    "instruct": {
        "ckpt": Path(r"D:\alpha_1_05B_it_v30.pt"),
        "hf": OUT_DIR / "hf_alpha1_it_v30",
        "gguf_stem": "alpha1-05B-it-v30",
        "eos_token": "<FIN>",
        "chat": True,
    },
    "base": {
        "ckpt": AQUI / "checkpoints_midtrain" / "alpha_1_annealing_05B.pt",
        "hf": OUT_DIR / "hf_alpha1_base",
        "gguf_stem": "alpha1-05B-base",
        "eos_token": "<|endofstory|>",
        "chat": False,
    },
    # Exp.27 (2026-09-12/13): v26 + agente_react_es (806 ejemplos ReAct),
    # candidato a evaluar contra el banco held-out react_agente_es_eval.jsonl
    # via evaluar_agente.py --backend ollama. NO sustituye a "instruct" (v24,
    # referencia vigente) hasta confirmar mejora real.
    "v27": {
        "ckpt": AQUI / "checkpoints" / "alpha_1_sft_alpaca_plus_v27_best.pt",
        "hf": OUT_DIR / "hf_alpha1_it_v27",
        "gguf_stem": "alpha1-05B-it-v27",
        "eos_token": "<FIN>",
        "chat": True,
    },
    # Exp.28 (2026-09-13): v27 con la cadena buscar+calcular reforzada (17->~200)
    # y prompt adelgazado. Mismo banco held-out para comparar v27 vs v28.
    "v28": {
        "ckpt": AQUI / "checkpoints" / "alpha_1_sft_alpaca_plus_v28_best.pt",
        "hf": OUT_DIR / "hf_alpha1_it_v28",
        "gguf_stem": "alpha1-05B-it-v28",
        "eos_token": "<FIN>",
        "chat": True,
    },
    # Exp.29 (2026-09-13): v28 + cadena-abstención (búsqueda fallida -> abstenerse
    # en vez de fabricar un año). Mismo banco held-out.
    "v29": {
        "ckpt": AQUI / "checkpoints" / "alpha_1_sft_alpaca_plus_v29_best.pt",
        "hf": OUT_DIR / "hf_alpha1_it_v29",
        "gguf_stem": "alpha1-05B-it-v29",
        "eos_token": "<FIN>",
        "chat": True,
    },
    # Exp.30 (2026-09-13): v29 + responder_directo (conocimiento canónico sin
    # herramienta). Evaluado con banco_calibracion_busqueda.jsonl. ADOPTADO
    # como referencia ("instruct", arriba).
    "v30": {
        "ckpt": AQUI / "checkpoints" / "alpha_1_sft_alpaca_plus_v30_best.pt",
        "hf": OUT_DIR / "hf_alpha1_it_v30",
        "gguf_stem": "alpha1-05B-it-v30",
        "eos_token": "<FIN>",
        "chat": True,
    },
    # Exp.31 (2026-09-13): v30 + contraste_know (afila el límite KNOW/SEARCH
    # tras hallazgos reales de alucinación en chat_web_app) + sin_herramienta
    # (tareas de redacción/procedimiento sin forzar Action).
    "v31": {
        "ckpt": AQUI / "checkpoints" / "alpha_1_sft_alpaca_plus_v31_best.pt",
        "hf": OUT_DIR / "hf_alpha1_it_v31",
        "gguf_stem": "alpha1-05B-it-v31",
        "eos_token": "<FIN>",
        "chat": True,
    },
    # Exp.32 (2026-09-14): v30 + extraccion_real (20 ej., Observation real
    # con el ano no en primera posicion) + pregunta_test (7 ej., MCQ sobre
    # texto) + cuento_protagonista (15 ej., fidelidad de protagonista). Ver
    # memoria [[dataset-v32-nuevas-categorias]].
    "v32": {
        "ckpt": AQUI / "checkpoints" / "alpha_1_sft_alpaca_plus_v32_best.pt",
        "hf": OUT_DIR / "hf_alpha1_it_v32",
        "gguf_stem": "alpha1-05B-it-v32",
        "eos_token": "<FIN>",
        "chat": True,
    },
    # Exp.32b (2026-09-14): v32 SIN extraccion_real (regresionaba KNOW/SEARCH,
    # ver memoria). Solo pregunta_test + cuento_protagonista sobre v30.
    "v32b": {
        "ckpt": AQUI / "checkpoints" / "alpha_1_sft_alpaca_plus_v32b_best.pt",
        "hf": OUT_DIR / "hf_alpha1_it_v32b",
        "gguf_stem": "alpha1-05B-it-v32b",
        "eos_token": "<FIN>",
        "chat": True,
    },
}

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}<USER>{{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}<ASSISTANT>{{ message['content'] }}<FIN>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<ASSISTANT>{% endif %}"
)

# Modelfile de Ollama para el instruct. Go template que replica el chat
# format. Con <FIN>=EOS en el GGUF ya cortaria solo; los stop son refuerzo.
MODELFILE_INSTRUCT = '''FROM ./{gguf}

TEMPLATE """{{{{- range $i, $m := .Messages }}}}{{{{- if eq $m.Role "user" }}}}<USER>{{{{ $m.Content }}}}
{{{{ else if eq $m.Role "assistant" }}}}<ASSISTANT>{{{{ $m.Content }}}}<FIN>
{{{{ end }}}}{{{{- end }}}}<ASSISTANT>"""

PARAMETER stop "<FIN>"
PARAMETER stop "<USER>"
PARAMETER temperature 0.3
PARAMETER top_k 30
PARAMETER repeat_penalty 1.3
'''


def cargar_tokenizer():
    tok = BPETokenizer()
    tok.load_state(json.loads(TOK_STATE.read_text(encoding="utf-8")))
    return tok


def cargar_modelo(ckpt: Path) -> SanchoGPT:
    m = SanchoGPT(
        vocab_size=BPE_VOCAB_SIZE, block_size=BLOCK_SIZE, n_embd=N_EMBD,
        n_head=N_HEAD, n_layer=N_LAYER, dropout=0.0, device="cpu",
        n_kv_head=N_KV_HEAD, tie_weights=TIE_WEIGHTS, rope_theta=ROPE_THETA,
        ffn_hidden=FFN_HIDDEN,
    )
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck.get("model", ck))
    sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m


def escribir_metadatos(hf_dir: Path, tok: BPETokenizer, eos_token: str, chat: bool) -> dict:
    (hf_dir / "tokenizer.json").write_text(
        tok.get_state()["tokenizer_json"], encoding="utf-8")

    ids = {s: tok._tok.token_to_id(s) for s in tok.special_tokens}

    tcfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<BOS>",
        "eos_token": eos_token,
        "pad_token": "<PAD>",
        "unk_token": "<UNK>",
        "clean_up_tokenization_spaces": False,
        "model_max_length": BLOCK_SIZE,
        "add_bos_token": False,
        "add_eos_token": False,
        "added_tokens_decoder": {
            str(i): {
                "content": s, "special": True, "lstrip": False, "rstrip": False,
                "normalized": False, "single_word": False,
            } for s, i in ids.items()
        },
    }
    if chat:
        tcfg["chat_template"] = CHAT_TEMPLATE
        tcfg["additional_special_tokens"] = [
            "<USER>", "<ASSISTANT>", "<FIN>", "<tool>", "</tool>", "<turno>",
            "<|endofstory|>", "<EOS>",
        ]
    (hf_dir / "tokenizer_config.json").write_text(
        json.dumps(tcfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # parchear config.json con los IDs reales
    cfgp = hf_dir / "config.json"
    cfg = json.loads(cfgp.read_text(encoding="utf-8"))
    cfg["bos_token_id"] = ids["<BOS>"]
    cfg["eos_token_id"] = ids[eos_token]
    cfg["pad_token_id"] = ids["<PAD>"]
    cfgp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    (hf_dir / "generation_config.json").write_text(json.dumps({
        "bos_token_id": ids["<BOS>"],
        "eos_token_id": ids[eos_token],
        "pad_token_id": ids["<PAD>"],
    }, indent=2), encoding="utf-8")

    return ids


def convertir_gguf(hf_dir: Path, gguf_stem: str, outtype: str) -> Path:
    conv = LLAMACPP / "convert_hf_to_gguf.py"
    if not conv.exists():
        raise SystemExit(f"No existe {conv}. Ajusta LLAMACPP.")
    out = OUT_DIR / f"{gguf_stem}-{outtype}.gguf"
    cmd = [sys.executable, str(conv), str(hf_dir),
           "--outfile", str(out), "--outtype", outtype]
    print("  ", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=list(MODELOS), required=True)
    ap.add_argument("--outtype", default="q8_0",
                     choices=["q8_0", "f16", "bf16", "f32"])
    ap.add_argument("--solo-hf", action="store_true",
                     help="Solo exportar el repo HF, sin convertir a GGUF.")
    args = ap.parse_args()

    m = MODELOS[args.modelo]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"1/4 tokenizer ({TOK_STATE.name})")
    tok = cargar_tokenizer()

    print(f"2/4 modelo <- {m['ckpt']}")
    if not Path(m["ckpt"]).exists():
        raise SystemExit(f"No existe el checkpoint {m['ckpt']}")
    modelo = cargar_modelo(Path(m["ckpt"]))

    print(f"3/4 export HF -> {m['hf']}")
    m["hf"].mkdir(parents=True, exist_ok=True)
    modelo.export_to_hf(str(m["hf"]), dtype=torch.bfloat16)
    ids = escribir_metadatos(m["hf"], tok, m["eos_token"], m["chat"])
    print(f"    tokens especiales: {ids}")
    print(f"    EOS = {m['eos_token']} (id {ids[m['eos_token']]})  BOS id {ids['<BOS>']}  PAD id {ids['<PAD>']}")

    if args.solo_hf:
        print("(--solo-hf) hecho.")
        return

    print(f"4/4 GGUF ({args.outtype})")
    gguf = convertir_gguf(m["hf"], m["gguf_stem"], args.outtype)
    print(f"\nGGUF: {gguf}  ({gguf.stat().st_size / 1e6:.0f} MB)")

    if m["chat"]:
        mf = OUT_DIR / f"Modelfile.{m['gguf_stem']}"
        mf.write_text(MODELFILE_INSTRUCT.format(gguf=gguf.name), encoding="utf-8")
        print(f"Modelfile Ollama: {mf}")
        print(f"  ollama create {m['gguf_stem']} -f {mf.name}   (desde {OUT_DIR})")
    else:
        print(f"  ollama:  echo 'FROM ./{gguf.name}' > Modelfile && ollama create {m['gguf_stem']} -f Modelfile")


if __name__ == "__main__":
    main()
