# benchmark_sdpa_backends.py
"""
Benchmark Alfa1 - backend SDPA para la implementacion ACTUAL de GQA.

Compara, con las shapes reales:
  - AUTO (dispatcher de PyTorch)
  - MATH
  - EFFICIENT_ATTENTION
  - CUDNN_ATTENTION
  - FLASH_ATTENTION

Mantiene exactamente la estrategia actual:
  K/V 6 cabezas -> repeat_interleave -> 24 cabezas -> SDPA

Objetivo:
  descubrir si PyTorch AUTO ya elige el backend mas rapido o si conviene
  forzar cuDNN/Efficient de forma explicita.

NO carga ni modifica checkpoints.
NO modifica model.py.
NO avanza el entrenamiento.

Ejecutar con el entrenamiento principal PARADO:
    python benchmark_sdpa_backends.py
"""

import argparse
import gc
import sys
import warnings

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception as e:
    raise RuntimeError(
        "Tu PyTorch no expone torch.nn.attention.sdpa_kernel/SDPBackend"
    ) from e


def gib(n):
    return n / (1024 ** 3)


def sync():
    torch.cuda.synchronize()


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    sync()


def gpu_info():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible.")

    props = torch.cuda.get_device_properties(0)
    free_b, total_b = torch.cuda.mem_get_info()

    print("=" * 92)
    print("BENCHMARK ALFA1 - BACKENDS SDPA (GQA ACTUAL CON repeat_interleave)")
    print("=" * 92)
    print(f"GPU: {props.name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"VRAM total: {gib(total_b):.2f} GiB")
    print(f"VRAM libre: {gib(free_b):.2f} GiB")
    print()

    # Estados globales
    for name, fn in [
        ("flash_sdp_enabled", getattr(torch.backends.cuda, "flash_sdp_enabled", None)),
        ("mem_efficient_sdp_enabled", getattr(torch.backends.cuda, "mem_efficient_sdp_enabled", None)),
        ("math_sdp_enabled", getattr(torch.backends.cuda, "math_sdp_enabled", None)),
        ("cudnn_sdp_enabled", getattr(torch.backends.cuda, "cudnn_sdp_enabled", None)),
        ("is_flash_attention_available", getattr(torch.backends.cuda, "is_flash_attention_available", None)),
    ]:
        if fn is not None:
            try:
                print(f"{name}: {fn()}")
            except Exception as e:
                print(f"{name}: ERROR {e}")
    print()

    if free_b < 6 * 1024**3:
        print("[ABORTADO] Menos de 6 GiB libres. Para el entrenamiento primero.")
        sys.exit(2)


def make_qkv(B, Hq, Hkv, T, D, seed=1337):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    q = torch.randn(B, Hq, T, D, dtype=torch.bfloat16, device="cuda", generator=g)
    k = torch.randn(B, Hkv, T, D, dtype=torch.bfloat16, device="cuda", generator=g)
    v = torch.randn(B, Hkv, T, D, dtype=torch.bfloat16, device="cuda", generator=g)
    q.mul_(0.1)
    k.mul_(0.1)
    v.mul_(0.1)
    return q.contiguous(), k.contiguous(), v.contiguous()


def expand_kv(q, k, v):
    rep = q.shape[1] // k.shape[1]
    return (
        k.repeat_interleave(rep, dim=1),
        v.repeat_interleave(rep, dim=1),
    )


def sdpa_auto(q, k, v):
    k, v = expand_kv(q, k, v)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
    )


def make_forced(backend):
    def fn(q, k, v):
        k, v = expand_kv(q, k, v)
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
            )
    return fn


def benchmark(name, fn, args, compile_it=False):
    cleanup()
    q0, k0, v0 = make_qkv(
        args.batch, args.hq, args.hkv, args.seq, args.head_dim, seed=20260816
    )

    runner = fn
    suffix = ""
    if compile_it:
        runner = torch.compile(fn, mode="default")
        suffix = " [compile]"

    # warmup
    try:
        for _ in range(args.warmup):
            q = q0.detach().clone().requires_grad_(True)
            k = k0.detach().clone().requires_grad_(True)
            v = v0.detach().clone().requires_grad_(True)
            out = runner(q, k, v)
            loss = out.float().square().mean()
            loss.backward()
            del q, k, v, out, loss
        sync()
    except Exception as e:
        cleanup()
        return {
            "name": name + suffix,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
        }

    cleanup()
    base_alloc = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    times = []
    try:
        for _ in range(args.repeats):
            q = q0.detach().clone().requires_grad_(True)
            k = k0.detach().clone().requires_grad_(True)
            v = v0.detach().clone().requires_grad_(True)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            out = runner(q, k, v)
            loss = out.float().square().mean()
            loss.backward()

            end.record()
            sync()
            times.append(start.elapsed_time(end))
            del q, k, v, out, loss
    except Exception as e:
        cleanup()
        return {
            "name": name + suffix,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
        }

    peak_alloc = torch.cuda.max_memory_allocated()
    med_ms = float(torch.tensor(times).median())
    mean_ms = float(torch.tensor(times).mean())
    tokens = args.batch * args.seq

    result = {
        "name": name + suffix,
        "ok": True,
        "median_ms": med_ms,
        "mean_ms": mean_ms,
        "tok_s": tokens / (med_ms / 1000.0),
        "peak_gib": gib(peak_alloc),
        "extra_gib": gib(max(0, peak_alloc - base_alloc)),
    }

    del q0, k0, v0, runner
    cleanup()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--hq", type=int, default=24)
    p.add_argument("--hkv", type=int, default=6)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args()

    gpu_info()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Shape real Alfa1:")
    print(
        f"B={args.batch}, T={args.seq}, Hq={args.hq}, Hkv={args.hkv}, "
        f"D={args.head_dim}, BF16, causal=True"
    )
    print("K/V se expanden 6->24 igual que en model.py.")
    print()

    backends = [
        ("AUTO", sdpa_auto),
        ("MATH", make_forced(SDPBackend.MATH)),
        ("EFFICIENT", make_forced(SDPBackend.EFFICIENT_ATTENTION)),
        ("CUDNN", make_forced(SDPBackend.CUDNN_ATTENTION)),
        ("FLASH", make_forced(SDPBackend.FLASH_ATTENTION)),
    ]

    print("-" * 92)
    print("1) EAGER")
    print("-" * 92)
    results = []
    for name, fn in backends:
        print(f"[BENCH] {name}...")
        results.append(benchmark(name, fn, args, compile_it=False))

    # Compilamos AUTO y todo backend forced que haya funcionado en eager.
    if not args.no_compile:
        print()
        print("-" * 92)
        print("2) torch.compile(mode='default')")
        print("-" * 92)
        for (name, fn), eager_result in zip(backends, results[:len(backends)]):
            if eager_result["ok"]:
                print(f"[BENCH] {name} [compile]...")
                results.append(benchmark(name, fn, args, compile_it=True))

    print()
    print(f"{'Ruta':<28} {'mediana ms':>12} {'tok/s':>14} {'VRAM pico':>12} {'extra pico':>12}")
    print("-" * 84)
    for r in results:
        if r["ok"]:
            print(
                f"{r['name']:<28} "
                f"{r['median_ms']:>12.3f} "
                f"{r['tok_s']:>14,.0f} "
                f"{r['peak_gib']:>11.2f}G "
                f"{r['extra_gib']:>11.2f}G"
            )

    failed = [r for r in results if not r["ok"]]
    if failed:
        print()
        print("No disponibles/fallidos:")
        for r in failed:
            # Solo primera linea para que no inunde la salida.
            err = r["error"].splitlines()[0]
            print(f"  - {r['name']}: {err}")

    # Decision sobre compiled si existe; si no, eager.
    good = [r for r in results if r["ok"]]
    compiled_good = [r for r in good if "[compile]" in r["name"]]
    pool = compiled_good if compiled_good else [r for r in good if "[compile]" not in r["name"]]

    auto_name = "AUTO [compile]" if compiled_good else "AUTO"
    auto = next((r for r in pool if r["name"] == auto_name), None)
    forced = [r for r in pool if not r["name"].startswith("AUTO")]

    print()
    print("=" * 92)
    print("DECISION")
    print("=" * 92)

    if auto is None or not forced:
        print("No hay comparacion suficiente. Pegame la salida.")
    else:
        best = min(forced, key=lambda r: r["median_ms"])
        speedup = auto["median_ms"] / best["median_ms"]
        time_red = 100.0 * (1.0 - best["median_ms"] / auto["median_ms"])
        mem_delta = best["peak_gib"] - auto["peak_gib"]

        print(f"AUTO: {auto['median_ms']:.3f} ms")
        print(f"Mejor backend forzado: {best['name']} = {best['median_ms']:.3f} ms")
        print(f"Speedup vs AUTO: {speedup:.3f}x")
        print(f"Cambio de tiempo: {time_red:+.1f}%")
        print(f"Cambio VRAM pico: {mem_delta:+.2f} GiB")
        print()

        if speedup >= 1.03 and mem_delta <= 0.10:
            print("HAY MEJORA: merece probar este backend forzado dentro del model.py")
            print("y medir tok/s del entrenamiento completo desde el mismo resume.pt.")
        else:
            print("AUTO YA ES OPTIMO/PRÁCTICAMENTE OPTIMO.")
            print("No tocaría SDPA; el siguiente paso sería perfilar el modelo completo.")

    print()
    print("Este benchmark NO ha cargado ni escrito checkpoints.")


if __name__ == "__main__":
    warnings.simplefilter("default")
    main()
