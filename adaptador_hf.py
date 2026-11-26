"""
adaptador_hf.py
---------------
Adaptador para poder pasar Qwen2.5-0.5B (o cualquier modelo causal de
HuggingFace) por benchmark_bcc_es.py sin tocar el resto del pipeline.

benchmark_bcc_es.py, para tus propios checkpoints, espera:
  - model(x)       -> logits                    (o tupla con logits en [0])
  - model(x, y)     -> (logits, loss)             (loss media por token, en nats)
  - model.generate(context, max_new_tokens, temperature, top_k, deterministic, fin_id)
  - tokenizer.encode(texto, normalize_chat=...) -> list[int]
  - tokenizer.decode(ids, skip_special=...)     -> str
  - tokenizer.load_state(...)                    (no aplica aquí -> no-op)

Este adaptador envuelve un modelo/tokenizer de HuggingFace para exponer
exactamente esa interfaz. cargar_modelo() en benchmark_bcc_es.py detecta
este caso por duck typing (ClaseModelo.desde_huggingface existe) y trata
'path' como un model_id o ruta local del modelo HF, no como un .pt propio.

Requiere: pip install transformers --break-system-packages
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModeloHF:
    def __init__(self, model, device, config_benchmark):
        self._model = model
        self._device = device
        # config con las claves que evaluar_checkpoint() imprime/usa
        # (block_size, vocab_size) -- no son hiperparámetros reales de
        # arquitectura del modelo HF, solo lo que el resto del script
        # necesita leer de 'config'.
        self.config_benchmark = config_benchmark

    @classmethod
    def desde_huggingface(cls, model_id, device):
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        model.to(device)
        model.eval()
        tok = AutoTokenizer.from_pretrained(model_id)
        config_benchmark = {
            "block_size": getattr(model.config, "max_position_embeddings", 32768),
            "vocab_size": len(tok),
            "dropout": 0.0,
        }
        return cls(model, device, config_benchmark)

    def to(self, device):
        self._model.to(device)
        self._device = device
        return self

    def eval(self):
        self._model.eval()
        return self

    def parameters(self):
        return self._model.parameters()

    @torch.no_grad()
    def __call__(self, x, y=None):
        salida = self._model(input_ids=x)
        logits = salida.logits
        if y is None:
            return logits
        # misma convención que el resto del script: loss media por token, en nats
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, context, max_new_tokens, temperature, top_k,
                 deterministic=True, fin_id=None):
        # fin_id lo pasa SIEMPRE generar() en benchmark_bcc_es.py como
        # keyword (ID de fin de documento). Antes esta firma no lo aceptaba y
        # cualquier test que generase texto (GSM8K, HumanEval) moria con
        # TypeError nada mas empezar; los que puntuan por verosimilitud
        # (MMLU, HellaSwag, los dos bancos de BCC-ES) no pasan por aqui y no
        # estaban afectados.
        # Aqui fin_id equivale al EOS del modelo HF: si se recibe se usa como
        # stop token real en vez de generar siempre hasta max_new_tokens.
        kwargs = dict(max_new_tokens=max_new_tokens)
        if fin_id is not None:
            kwargs["eos_token_id"] = fin_id
        if deterministic:
            kwargs["do_sample"] = False
        else:
            kwargs["do_sample"] = True
            kwargs["temperature"] = max(temperature, 1e-4)
            kwargs["top_k"] = top_k
        return self._model.generate(context, **kwargs)


class TokenizerHF:
    def __init__(self, model_id):
        self._tok = AutoTokenizer.from_pretrained(model_id)
        # generar() en benchmark_bcc_es.py busca 'story_id' o 'eos_id' en el
        # tokenizer para saber donde parar la generacion. Sin esto stop_id
        # quedaba en None (no rompia nada: los usos posteriores lo protegen
        # con 'is not None'), pero la generacion nunca paraba en EOS y corria
        # siempre hasta max_new_tokens.
        # OJO: para pythia y SmolLM2 el eos_token_id es 0, que es falsy --
        # cualquier comprobacion tipo 'if fin_id:' desactivaria el corte en
        # silencio justo para esos modelos. Usar siempre 'is not None'.
        self.eos_id = self._tok.eos_token_id

    def load_state(self, state):
        # no aplica: el tokenizer HF ya se carga desde el hub/ruta, no
        # desde un 'tokenizer_state' embebido en un checkpoint propio.
        pass

    def encode(self, texto, normalize_chat=False):
        return self._tok.encode(texto, add_special_tokens=False)

    def decode(self, ids, skip_special=True):
        return self._tok.decode(ids, skip_special_tokens=skip_special)
