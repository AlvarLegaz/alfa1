# model_v2.py
"""
ProyectoSancho v2 - arquitectura tipo Llama/Qwen.

Mejoras sobre la v1 (SanchoGPT clasico con wpe + LayerNorm + GELU):
  - RoPE (Rotary Position Embedding) en lugar de position embeddings
    aprendidos. Mejor extrapolacion de longitud y export limpio a HF.
  - RMSNorm (pre-norm) en lugar de LayerNorm. Mas barato, mapea 1:1 a Llama.
  - SwiGLU (gate/up/down) en lugar de FFN GELU 4x. Mejor calidad por FLOP.
  - GQA ya estaba en v1; se mantiene (n_kv_head <= n_head).
  - QKV SEPARADO y SIN bias (esquema Llama / Qwen3), para que el state_dict
    exporte por copia directa a LlamaForCausalLM (cero friccion con llama.cpp).

DISENO CLAVE: el naming de submodulos imita a HF
(self_attn.q_proj/k_proj/v_proj/o_proj, mlp.gate_proj/up_proj/down_proj,
input_layernorm, post_attention_layernorm, model.layers.N, model.norm, lm_head).
Asi el exportador a HF es un renombrado mecanico, no un mapeo a mano.

COMPATIBILIDAD: la clase publica se sigue llamando SanchoGPT y conserva la
firma de v1 (vocab_size, block_size, n_embd, n_head, n_layer, dropout, device,
n_kv_head, tie_weights) mas extras opcionales (rope_theta, ffn_hidden).
Es drop-in en train_pretrain_bpe.py: solo cambia el import.

NOTA sobre el contrato de forward(): devuelve (logits, loss, new_caches),
igual que el model.py v1 que subiste. Tu train_pretrain_bpe.py hace
`_, loss = model(X, Y)`, que desempaqueta solo 2 valores y PETARIA. Ese bug
ya esta en tu v1. Para v2 conviene cambiar esas dos lineas a:
    logits, loss, _ = model(X, Y)
(o usar el helper model.loss_fwd(X, Y) que anado abajo, que devuelve solo loss).
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


# ======================================================================
# Utilidades de dimensionado
# ======================================================================

def llama_intermediate_size(n_embd, multiple_of=256, ffn_dim_multiplier=None):
    """Tamano del hidden de SwiGLU al estilo Llama.

    SwiGLU tiene 3 matrices (gate, up, down) frente a las 2 de un FFN GELU,
    asi que para igualar parametros se usa hidden ~ (8/3)*n_embd en lugar de
    4*n_embd, redondeado al alza al multiplo `multiple_of` (mejor para tensor
    cores). ffn_dim_multiplier permite afinarlo (Llama2-70B usa 1.3).
    """
    hidden = int(8 * n_embd / 3)
    if ffn_dim_multiplier is not None:
        hidden = int(ffn_dim_multiplier * hidden)
    hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
    return hidden


# ======================================================================
# RoPE
# ======================================================================

class RotaryEmbedding(nn.Module):
    """Genera cos/sin para RoPE y los cachea hasta max_seq_len.

    Convencion de partido de Llama/HF: la dimension head_dim se parte en dos
    MITADES (no en pares intercalados como en el paper GPT-NeoX original).
    Esto es lo que espera LlamaForCausalLM, asi que lo replicamos para que el
    export sea valido sin re-permutar nada.
    """

    def __init__(self, head_dim, max_seq_len, base=10000.0, device=None):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim debe ser par para RoPE"
        self.head_dim = head_dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len, device)

    def _build_cache(self, seq_len, device):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=torch.float32,
                         device=device if device is not None else self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq.to(t.device))  # (seq_len, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)             # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len, device, dtype):
        if seq_len > self.max_seq_len_cached:
            self._build_cache(seq_len, device)
        return (self.cos_cached[:seq_len].to(device=device, dtype=dtype),
                self.sin_cached[:seq_len].to(device=device, dtype=dtype))


def rotate_half(x):
    """Parte por la mitad y rota: (-x2, x1). Convencion HF/Llama."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q, k, cos, sin, pos_offset=0):
    """Aplica RoPE a q,k de forma (B, n_head, T, head_dim).

    cos/sin llegan como (T_total, head_dim); seleccionamos la ventana
    [pos_offset : pos_offset+T] para soportar decoding incremental.
    """
    T = q.shape[-2]
    cos = cos[pos_offset:pos_offset + T].unsqueeze(0).unsqueeze(0)  # (1,1,T,hd)
    sin = sin[pos_offset:pos_offset + T].unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# ======================================================================
# RMSNorm
# ======================================================================

class RMSNorm(nn.Module):
    """RMSNorm sin centrado ni bias. Mapea 1:1 a LlamaRMSNorm."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculo en fp32 para estabilidad, luego volvemos al dtype original.
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm.to(dtype)) * self.weight


# ======================================================================
# Atencion con GQA + RoPE
# ======================================================================

class CausalSelfAttention(nn.Module):
    """Q/K/V SEPARADOS, sin bias, GQA, RoPE, SDPA (flash en H100).

    Naming HF: q_proj, k_proj, v_proj, o_proj.
    """

    def __init__(self, n_embd, n_head, n_kv_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        assert n_head % n_kv_head == 0
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head
        self.n_embd = n_embd
        self.attn_dropout_p = dropout

        q_dim = n_head * self.head_dim
        kv_dim = n_kv_head * self.head_dim
        self.q_proj = nn.Linear(n_embd, q_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, kv_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, n_embd, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin, kv_cache=None, use_cache=False, pos_offset=0):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # RoPE ANTES del cache (las claves se guardan ya rotadas, como en HF).
        q, k = apply_rotary(q, k, cos, sin, pos_offset=pos_offset)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat((k_prev, k), dim=2)
            v = torch.cat((v_prev, v), dim=2)

        new_cache = (k, v) if use_cache else None

        if self.n_kv_head != self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # is_causal solo si q y k tienen la misma longitud (prefill/train).
        # En decoding incremental (T_q=1) no hace falta mascara.
        is_causal = kv_cache is None

        # RTX 5090 / PyTorch 2.11: benchmark real de Alfa1 mostró que
        # CUDNN_ATTENTION es ~1.44x más rápido que el dispatcher AUTO en este
        # tramo (B=10, T=1024, H=24, D=64, BF16), sin aumento medible de VRAM.
        #
        # Se fuerza SOLO cuando no hay KV-cache (training / prefill). Para
        # decoding incremental con kv_cache se conserva AUTO, que es más
        # compatible con longitudes Q/K distintas.
        if q.is_cuda and kv_cache is None and T > 1:
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.attn_dropout_p if self.training else 0.0,
                    is_causal=is_causal,
                )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                is_causal=is_causal,
            )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)
        out = self.resid_dropout(out)
        return out, new_cache


# ======================================================================
# SwiGLU MLP
# ======================================================================

class FeedForward(nn.Module):
    """SwiGLU: down( silu(gate(x)) * up(x) ). Naming HF: gate/up/down_proj."""

    def __init__(self, n_embd, hidden, dropout):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, hidden, bias=False)
        self.up_proj = nn.Linear(n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return self.dropout(x)


# ======================================================================
# Bloque transformer (pre-norm, naming HF)
# ======================================================================

class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, hidden, dropout, rms_eps):
        super().__init__()
        self.input_layernorm = RMSNorm(n_embd, eps=rms_eps)
        self.self_attn = CausalSelfAttention(n_embd, n_head, n_kv_head, dropout)
        self.post_attention_layernorm = RMSNorm(n_embd, eps=rms_eps)
        self.mlp = FeedForward(n_embd, hidden, dropout)

    def forward(self, x, cos, sin, kv_cache=None, use_cache=False, pos_offset=0):
        attn_out, new_cache = self.self_attn(
            self.input_layernorm(x), cos, sin,
            kv_cache=kv_cache, use_cache=use_cache, pos_offset=pos_offset,
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_cache


# ======================================================================
# Modelo (clase publica SanchoGPT - drop-in con la firma v1)
# ======================================================================

class SanchoGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        n_embd,
        n_head,
        n_layer,
        dropout,
        device,
        n_kv_head=None,
        tie_weights=True,
        rope_theta=10000.0,
        ffn_hidden=None,          # None -> llama_intermediate_size(n_embd)
        ffn_multiple_of=256,
        rms_eps=1e-5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.device = device
        self.tie_weights = tie_weights
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.rope_theta = rope_theta
        self.rms_eps = rms_eps

        if n_kv_head is None:
            n_kv_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head

        if ffn_hidden is None:
            ffn_hidden = llama_intermediate_size(n_embd, multiple_of=ffn_multiple_of)
        self.ffn_hidden = ffn_hidden

        # --- Embeddings (sin wpe: la posicion la pone RoPE) ---
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.rotary = RotaryEmbedding(self.head_dim, block_size, base=rope_theta, device=device)

        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, n_kv_head, ffn_hidden, dropout, rms_eps)
            for _ in range(n_layer)
        ])

        self.ln_f = RMSNorm(n_embd, eps=rms_eps)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        if self.tie_weights:
            self.lm_head.weight = self.token_embedding_table.weight

        self.apply(self._init_weights)

        # Init escalada de las proyecciones residuales de salida (o_proj y
        # down_proj), dividida por sqrt(2*n_layer) para estabilizar la varianza
        # en la rama residual (truco GPT-2, valido tambien aqui).
        for pn, p in self.named_parameters():
            if pn.endswith("o_proj.weight") or pn.endswith("down_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_caches=None, use_cache=False, pos_offset=0):
        B, T = idx.shape
        x = self.drop(self.token_embedding_table(idx))

        # cos/sin para el rango absoluto [0, pos_offset+T); apply_rotary recorta.
        cos, sin = self.rotary(pos_offset + T, idx.device, x.dtype)

        new_caches = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, layer_new_cache = block(
                x, cos, sin,
                kv_cache=layer_cache, use_cache=use_cache, pos_offset=pos_offset,
            )
            if use_cache:
                new_caches.append(layer_new_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            Bx, Tx, Cx = logits.shape
            loss = F.cross_entropy(
                logits.reshape(Bx * Tx, Cx),
                targets.reshape(Bx * Tx),
                ignore_index=-100,
            )
        return logits, loss, new_caches

    def loss_fwd(self, idx, targets):
        """Helper de entrenamiento: devuelve SOLO la loss.

        Evita el bug de desempaquetado `_, loss = model(X, Y)` (forward
        devuelve 3 valores). Uso en el bucle:  loss = model.loss_fwd(X, Y)
        """
        _, loss, _ = self(idx, targets)
        return loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.3, top_k=None,
                 deterministic=False, fin_id=None):
        idx_cond = idx[:, -self.block_size:]
        pos_offset = 0
        logits, _, kv_caches = self(idx_cond, use_cache=True, pos_offset=pos_offset)
        pos_offset = idx_cond.shape[1]
        logits = logits[:, -1, :]

        for _ in range(max_new_tokens):
            if deterministic:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    values, indices = torch.topk(logits, top_k)
                    filtered = torch.full_like(logits, float("-inf"))
                    filtered.scatter_(1, indices, values)
                    logits = filtered
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

            if fin_id is not None and idx_next.shape[0] == 1 and idx_next.item() == fin_id:
                break

            if pos_offset >= self.block_size:
                kv_caches = [
                    (k[:, :, -(self.block_size - 1):, :],
                     v[:, :, -(self.block_size - 1):, :])
                    for (k, v) in kv_caches
                ]
                pos_offset = self.block_size - 1

            logits, _, kv_caches = self(
                idx_next, use_cache=True, kv_caches=kv_caches, pos_offset=pos_offset,
            )
            pos_offset += 1
            logits = logits[:, -1, :]

        return idx

    # ------------------------------------------------------------------
    # Export a HF LlamaForCausalLM
    # ------------------------------------------------------------------
    def export_to_hf(self, out_dir, tokenizer_state=None, dtype=torch.bfloat16):
        """Exporta los pesos a un repo HF cargable como LlamaForCausalLM.

        Renombrado mecanico (gracias al naming alineado). Como tie_weights
        comparte embedding y lm_head, escribimos tie_word_embeddings=True y
        omitimos lm_head.weight (Llama lo reconstruye por tying).

        Requiere `transformers` instalado para escribir el config; si no esta,
        igual deja el state_dict renombrado en {out_dir}/pytorch_model.bin.
        """
        import os
        import json
        os.makedirs(out_dir, exist_ok=True)

        sd = self.state_dict()
        new_sd = {}

        # Embeddings y norma final / cabeza
        new_sd["model.embed_tokens.weight"] = sd["token_embedding_table.weight"]
        new_sd["model.norm.weight"] = sd["ln_f.weight"]
        if not self.tie_weights:
            new_sd["lm_head.weight"] = sd["lm_head.weight"]

        # Capas: nuestros nombres ya coinciden con los de Llama salvo el prefijo
        # "blocks.N." -> "model.layers.N.".
        for i in range(self.n_layer):
            src = f"blocks.{i}."
            dst = f"model.layers.{i}."
            for sub in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ]:
                new_sd[dst + sub] = sd[src + sub]

        new_sd = {k: v.to(dtype).contiguous() for k, v in new_sd.items()}

        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "vocab_size": self.vocab_size,
            "hidden_size": self.n_embd,
            "intermediate_size": self.ffn_hidden,
            "num_hidden_layers": self.n_layer,
            "num_attention_heads": self.n_head,
            "num_key_value_heads": self.n_kv_head,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.block_size,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.rms_eps,
            "tie_word_embeddings": bool(self.tie_weights),
            "hidden_act": "silu",
            "attention_bias": False,
            "mlp_bias": False,
            "torch_dtype": str(dtype).replace("torch.", ""),
        }

        try:
            from transformers import LlamaConfig, LlamaForCausalLM
            cfg = LlamaConfig(**{k: v for k, v in config.items()
                                 if k not in ("architectures", "torch_dtype")})
            hf_model = LlamaForCausalLM(cfg).to(dtype)
            missing, unexpected = hf_model.load_state_dict(new_sd, strict=False)
            # Con tying, lm_head.weight aparece en "missing" pero se reconstruye.
            hf_model.save_pretrained(out_dir)
            print(f"[export] LlamaForCausalLM guardado en {out_dir}")
            print(f"[export] missing={list(missing)} unexpected={list(unexpected)}")
        except ImportError:
            torch.save(new_sd, os.path.join(out_dir, "pytorch_model.bin"))
            with open(os.path.join(out_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)
            print(f"[export] transformers no disponible: dejado state_dict + "
                  f"config.json en {out_dir}")

        if tokenizer_state is not None:
            with open(os.path.join(out_dir, "tokenizer_state.json"), "w",
                      encoding="utf-8") as f:
                json.dump(tokenizer_state, f, ensure_ascii=False, indent=2)

        return out_dir
