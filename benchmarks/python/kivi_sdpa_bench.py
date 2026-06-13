import argparse
import math
import time

import mlx.core as mx

N_warmup = 5
N_iter_bench = 40


def bench(f, *args, iters=N_iter_bench, **kwargs):
    for _ in range(N_warmup):
        mx.eval(f(*args, **kwargs))

    tic = time.perf_counter_ns()
    for _ in range(iters):
        mx.eval(f(*args, **kwargs))
    toc = time.perf_counter_ns()
    return (toc - tic) * 1e-6 / iters


def prepare_inputs(B, qL, kL, D, qH, kH, dtype):
    scale = 1.0 / math.sqrt(D)
    q = mx.random.normal((B, qH, qL, D)).astype(dtype)
    k = mx.random.normal((B, kH, kL, D)).astype(dtype)
    v = mx.random.normal((B, kH, kL, D)).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v, scale


def qk_reference(q, kq, ks, kb, scale, key_group_size, bits):
    k = mx.dequantize(kq, ks, kb, group_size=key_group_size, bits=bits)
    if q.shape[1] != k.shape[1]:
        k = mx.repeat(k, q.shape[1] // k.shape[1], axis=1)
    return mx.matmul(q, k) * scale


def fused_qk(q, kq, ks, kb, scale, key_group_size, bits):
    return mx.fast.kivi_fused_dequantized_matmul(
        q,
        kq,
        ks,
        kb,
        scale=scale,
        key_group_size=key_group_size,
        bits=bits,
    )


def attn_reference(
    q,
    kq,
    ks,
    kb,
    vq,
    vs,
    vb,
    scale,
    mask,
    key_group_size,
    value_group_size,
    bits,
):
    k = mx.dequantize(kq, ks, kb, group_size=key_group_size, bits=bits)
    k = mx.swapaxes(k, -1, -2)
    v = mx.dequantize(vq, vs, vb, group_size=value_group_size, bits=bits)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def kivi_attn(
    q,
    kq,
    ks,
    kb,
    vq,
    vs,
    vb,
    scale,
    mask,
    key_group_size,
    value_group_size,
    bits,
):
    return mx.fast.kivi_scaled_dot_product_attention(
        q,
        kq,
        ks,
        kb,
        vq,
        vs,
        vb,
        scale=scale,
        mask=mask,
        key_group_size=key_group_size,
        value_group_size=value_group_size,
        bits=bits,
    )


def run_shape(
    name,
    B,
    qL,
    kL,
    D,
    qH,
    kH,
    dtype,
    mask,
    key_group_size,
    value_group_size,
    bits,
    iters,
):
    q, k, v, scale = prepare_inputs(B, qL, kL, D, qH, kH, dtype)
    kq, ks, kb, vq, vs, vb = mx.fast.kivi_quantize_kv(
        k,
        v,
        key_group_size=key_group_size,
        value_group_size=value_group_size,
        bits=bits,
    )
    mx.eval(kq, ks, kb, vq, vs, vb)

    qk_ref = qk_reference(q, kq, ks, kb, scale, key_group_size, bits)
    qk_out = fused_qk(q, kq, ks, kb, scale, key_group_size, bits)
    attn_ref = attn_reference(
        q,
        kq,
        ks,
        kb,
        vq,
        vs,
        vb,
        scale,
        mask,
        key_group_size,
        value_group_size,
        bits,
    )
    attn_out = kivi_attn(
        q,
        kq,
        ks,
        kb,
        vq,
        vs,
        vb,
        scale,
        mask,
        key_group_size,
        value_group_size,
        bits,
    )
    mx.eval(qk_ref, qk_out, attn_ref, attn_out)

    qk_abs = mx.max(mx.abs(qk_ref - qk_out)).item()
    attn_abs = mx.max(mx.abs(attn_ref - attn_out)).item()
    qk_ref_ms = bench(
        qk_reference,
        q,
        kq,
        ks,
        kb,
        scale,
        key_group_size,
        bits,
        iters=iters,
    )
    qk_fused_ms = bench(
        fused_qk,
        q,
        kq,
        ks,
        kb,
        scale,
        key_group_size,
        bits,
        iters=iters,
    )
    attn_ref_ms = bench(
        attn_reference,
        q,
        kq,
        ks,
        kb,
        vq,
        vs,
        vb,
        scale,
        mask,
        key_group_size,
        value_group_size,
        bits,
        iters=iters,
    )
    attn_kivi_ms = bench(
        kivi_attn,
        q,
        kq,
        ks,
        kb,
        vq,
        vs,
        vb,
        scale,
        mask,
        key_group_size,
        value_group_size,
        bits,
        iters=iters,
    )

    print(
        f"{name:8s} B={B:2d} qL={qL:5d} kL={kL:5d} D={D:3d} "
        f"qH={qH:3d} kH={kH:3d} dtype={dtype} mask={mask} bits={bits} "
        f"qk_abs={qk_abs:.3e} attn_abs={attn_abs:.3e} "
        f"qk_ref={qk_ref_ms:.3f}ms qk_fused={qk_fused_ms:.3f}ms "
        f"attn_ref={attn_ref_ms:.3f}ms attn_kivi={attn_kivi_ms:.3f}ms"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KIVI SDPA benchmarks")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--key-group-size", type=int, default=64)
    parser.add_argument("--value-group-size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=N_iter_bench)
    parser.add_argument("--mask", choices=["none", "causal"], default="causal")
    parser.add_argument("--case", choices=["all", "prefill", "decode"], default="all")
    args = parser.parse_args()

    mx.random.seed(7)
    mask = None if args.mask == "none" else args.mask
    dtype = mx.float16

    shapes = []
    if args.case in ("all", "prefill"):
        shapes.append(("prefill", 1, 128, 128, 64, 8, 4))
    if args.case in ("all", "decode"):
        shapes.append(("decode", 1, 1, 256, 64, 8, 4))

    for shape in shapes:
        run_shape(
            *shape,
            dtype,
            mask,
            args.key_group_size,
            args.value_group_size,
            args.bits,
            args.iters,
        )
