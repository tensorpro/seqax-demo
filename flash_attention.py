from functools import partial
from time import time
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

import jax_extra
from jax_extra import save_for_backward
from shardlib import kernelops, shardtypes
from shardlib.kernelops import Carry, pallas_call, sequentially
from shardlib.shardtypes import bf16, f32

shardtypes.register_with_typeguard()


@shardtypes.pytree_dataclass
class AttentionConfig:
    q_block_fwd: int = 32
    k_block_fwd: int = 32
    q_block_bwd: int = 32
    k_block_bwd: int = 32


MASK_VALUE = -1e10


@shardtypes.scope
def attention(
    q: bf16[b"B QL G H D"],
    k: bf16[b"B KL H D"],
    v: bf16[b"B KL H D"],
    config: AttentionConfig = AttentionConfig(),
) -> bf16[b"B QL G H D"]:
    @jax.custom_gradient
    def attention_fn(q, k, v):
        O, L = _attn_fwd(q, k, v, config)
        q, k, v, O, L = map(save_for_backward, [q, k, v, O, L])

        def grad_fn(dO):
            D = jnp.sum(jnp.float32(dO) * jnp.float32(O), axis=-1, keepdims=True)
            dq = q_bwd(q, k, v, D, dO, L, config)
            dk, dv = kv_bwd(q, k, v, D, dO, L, config)
            return dq, dk, dv

        return O, grad_fn

    return attention_fn(q, k, v)


@shardtypes.scope
def _attn_fwd(
    q: bf16[b"B QL G H D"], k: bf16[b"B KL H D"], v: bf16[b"B KL H D"], config: AttentionConfig
) -> tuple[bf16[b"B QL G H D"], f32[b"B QL G H 1"]]:
    q_block_size = config.q_block_fwd
    k_block_size = config.k_block_fwd

    @pallas_call(["QL -> q_block", "KL -> *", "D -> *"], q_block=q_block_size, k_block=k_block_size)
    def kernel(
        q_ref: b"B [QL] G H [D]",
        k_ref: b"B [KL] H [D]",
        v_ref: b"B [KL] H [D]",
        *,
        output_ref: bf16[b"B [QL] G H [D]"],
        L_ref: f32[b"B [QL] G H [1]"],
    ):
        q = q_ref[:]
        min_qpos = kernelops.get_index("q_block") * q_block_size
        qpos = min_qpos + jnp.arange(q_block_size)
        max_qpos = min_qpos + q_block_size - 1

        @sequentially(
            "KL -> k_block", Carry=["q_block 1", "q_block 1", "q_block D"], k_block=k_block_size, q_block=q_block_size
        )
        def kv_loop(k_: b"[KL] D", v_: b"[KL] D", carry: Carry) -> Carry:
            min_kpos = kernelops.get_index("k_block") * k_block_size
            kpos = min_kpos + jnp.arange(k_block_size)

            def run():
                nonlocal k_, v_
                k, v = k_[:], v_[:]
                prev_max, prev_sum, prev_acc = carry
                logits = pl.dot(q, k, trans_b=True)
                mask = qpos[:, jnp.newaxis] >= kpos[jnp.newaxis, :]
                logits = jnp.where(mask, logits, MASK_VALUE)
                local_max = logits.max(axis=-1, keepdims=True)
                new_max = jnp.maximum(prev_max, local_max)
                exp_logits = jnp.exp(logits - new_max)
                correction = jnp.exp(prev_max - new_max)
                new_sum = prev_sum * correction + jnp.sum(exp_logits, axis=-1, keepdims=True)
                new_acc = prev_acc * correction + pl.dot(jnp.bfloat16(exp_logits), v)
                return new_max, new_sum, new_acc

            def skip():
                return carry

            return jax.lax.cond(max_qpos >= min_kpos, run, skip)

        d_head = q.shape[-1]
        row_max, row_sum, unnormalized_o = kv_loop(
            k_ref,
            v_ref,
            (
                jnp.zeros((q_block_size, 1), jnp.float32) - jnp.inf,  # max(logits)
                jnp.zeros((q_block_size, 1), jnp.float32),  # sum(exp(logits - max(logits)))
                jnp.zeros((q_block_size, d_head), jnp.float32),
            ),
        )
        output_ref[:] = jnp.bfloat16(unnormalized_o / row_sum)
        L_ref[:] = row_max + jnp.log(row_sum)

    return kernel(q, k, v)


def shared_bwd(q, k, v, L, D, dO, qpos, kpos):
    logits = pl.dot(q, k, trans_b=True)
    mask = qpos[:, jnp.newaxis] >= kpos[jnp.newaxis, :]
    logits = jnp.where(mask, logits, MASK_VALUE)
    P = jnp.exp(logits - L)
    dP = pl.dot(dO, v, trans_b=True)
    dlogits = P * (dP - D)
    return P, dlogits


@shardtypes.scope
def q_bwd(
    q: bf16[b"B QL G H D"],
    k: bf16[b"B KL H D"],
    v: bf16[b"B KL H D"],
    D: f32[b"B QL G H 1"],
    dO: bf16[b"B QL G H D"],
    L: f32[b"B QL G H 1"],
    config: AttentionConfig,
) -> bf16[b"B QL G H D"]:
    q_block_size = config.q_block_bwd
    k_block_size = config.k_block_bwd

    @pallas_call(["QL -> q_block", "KL -> *", "D -> *", "1 -> *"], k_block=k_block_size, q_block=q_block_size)
    def kernel(
        q_ref: b"B [QL] G H [D]",
        k_ref: b"B [KL] H [D]",
        v_ref: b"B [KL] H [D]",
        L_ref: b"B [QL] G H [1]",
        D_ref: b"B [QL] G H [1]",
        dO_ref: b"B [QL] G H [D]",
        *,
        dQ_ref: bf16[b"B [QL] G H [D]"],
    ):
        q = q_ref[:]
        L = L_ref[:]
        D = D_ref[:]
        dO = dO_ref[:]
        min_qpos = kernelops.get_index("q_block") * q_block_size
        qpos = min_qpos + jnp.arange(q_block_size)
        max_qpos = min_qpos + q_block_size - 1

        @sequentially("KL -> k_block", q_block=q_block_size, k_block=k_block_size, Carry="q_block D")
        def kv_loop(k: b"[KL] D", v: b"[KL] D", carry: Carry) -> Carry:
            min_kpos = kernelops.get_index("k_block") * k_block_size
            kpos = min_kpos + jnp.arange(k_block_size)

            def run():
                nonlocal k, v
                k, v = k[:], v[:]
                dQ = carry
                _, dlogits = shared_bwd(q, k, v, L, D, dO, qpos, kpos)
                dQ = dQ + pl.dot(jnp.bfloat16(dlogits), k)
                return dQ

            def skip():
                return carry

            return jax.lax.cond(max_qpos >= min_kpos, run, skip)

        d_head = q.shape[-1]
        dQ = kv_loop(
            k_ref,
            v_ref,
            jnp.zeros((q_block_size, d_head), jnp.float32),
        )
        dQ_ref[:] = jnp.bfloat16(dQ)

    dq = kernel(q, k, v, L, D, dO)
    return dq


@shardtypes.scope
def kv_bwd(
    q: bf16[b"B QL G H D"],
    k: bf16[b"B KL H D"],
    v: bf16[b"B KL H D"],
    D: f32[b"B QL G H 1"],
    dO: bf16[b"B QL G H D"],
    L: f32[b"B QL G H 1"],
    config: AttentionConfig,
) -> Tuple[bf16[b"B QL G H D"], bf16[b"B QL G H D"]]:
    q_block_size = config.q_block_bwd
    k_block_size = config.k_block_bwd

    @pallas_call(["QL -> *", "KL -> k_block", "D -> *", "1 -> *"], k_block=k_block_size, q_block=q_block_size)
    def kernel(
        q_ref: b"B [QL] G H [D]",
        k_ref: b"B [KL] H [D]",
        v_ref: b"B [KL] H [D]",
        L_ref: b"B [QL] G H [1]",
        D_ref: b"B [QL] G H [1]",
        dO_ref: b"B [QL] G H [D]",
        *,
        dK_ref: bf16[b"B [KL] G H [D]"],
        dV_ref: bf16[b"B [KL] G H [D]"],
    ):
        k = k_ref[:]
        v = v_ref[:]
        min_kpos = kernelops.get_index("k_block") * k_block_size
        kpos = min_kpos + jnp.arange(k_block_size)

        @sequentially("QL -> q_block", q_block=q_block_size, k_block=k_block_size, Carry=["k_block D", "k_block D"])
        def q_loop(q_: b"[QL] D", L_: b"[QL] 1", D_: b"[QL] 1", dO_: b"[QL] D", carry: Carry) -> Carry:
            min_qpos = kernelops.get_index("q_block") * q_block_size
            qpos = min_qpos + jnp.arange(q_block_size)
            max_qpos = min_qpos + q_block_size - 1

            def run():
                nonlocal q_, L_, D_, dO
                q, L, D, dO = q_[:], L_[:], D_[:], dO_[:]
                dK, dV = carry
                P, dlogits = shared_bwd(q, k, v, L, D, dO, qpos, kpos)
                dV = dV + pl.dot(jnp.bfloat16(P), dO, trans_a=True)
                dK = dK + pl.dot(jnp.bfloat16(dlogits), q, trans_a=True)
                return dK, dV

            def skip():
                return carry

            return jax.lax.cond(max_qpos >= min_kpos, run, skip)

        d_head = q.shape[-1]
        dk, dv = q_loop(
            q_ref,
            L_ref,
            D_ref,
            dO_ref,
            (jnp.zeros((k_block_size, d_head), jnp.float32), jnp.zeros((k_block_size, d_head), jnp.float32)),
        )
        dK_ref[:] = jnp.bfloat16(dk)
        dV_ref[:] = jnp.bfloat16(dv)

    dgk, dgv = kernel(q, k, v, L, D, dO)
    return dgk.sum(-3), dgv.sum(-3)


##### Test and benchmark utilities below #####

import einops
from jax.experimental import mesh_utils
from jax.sharding import Mesh


@jax.jit
@partial(shardtypes.typed_shard_map, check_rep=False)
def jax_attn_fwd(
    q: bf16[b"B QL G H D"], k: bf16[b"B KL H D"], v: bf16[b"B KL H D"]
) -> tuple[bf16[b"B QL G H D"], f32[b"B QL G H 1"]]:
    logits = jnp.einsum("bqghd,bkhd->bqkgh", q, k, preferred_element_type=jnp.float32)
    mask = jnp.arange(q.shape[1])[:, None] >= jnp.arange(k.shape[1])[None, :]
    masked_logits = jnp.where(mask[None, :, :, None, None], logits, MASK_VALUE)
    probs = jnp.bfloat16(jax.nn.softmax(masked_logits, axis=2))
    qkv = jnp.einsum("bqkgh,bkhd->bqghd", probs, v)
    max_logits = jnp.max(masked_logits, axis=2, keepdims=True)
    logsumexp = jnp.log(jnp.sum(jnp.exp(masked_logits - max_logits), axis=2, keepdims=True)) + max_logits
    return qkv, einops.rearrange(logsumexp, "b q 1 g h -> b q g h 1")


@jax.jit
@partial(shardtypes.typed_shard_map, check_rep=False)
def jax_attn_fwd_and_bwd(
    q: bf16[b"B QL G H D"], k: bf16[b"B KL H D"], v: bf16[b"B KL H D"]
) -> tuple[bf16[b"B QL G H D"], bf16[b"B KL H D"], bf16[b"B KL H D"]]:
    dq, dk, dv = jax.grad(lambda q, k, v: jnp.sum(jax_attn_fwd(q, k, v)[0]), argnums=(0, 1, 2))(q, k, v)
    return dq, dk, dv


@jax.jit
@partial(shardtypes.typed_shard_map, check_rep=False)
def pallas_attn_fwd(
    q: bf16[b"B QL G H D"], k: bf16[b"B KL H D"], v: bf16[b"B KL H D"]
) -> tuple[bf16[b"B QL G H D"], f32[b"B QL G H 1"]]:
    O, L = _attn_fwd(q, k, v, AttentionConfig())
    return O, L


@jax.jit
@partial(shardtypes.typed_shard_map, check_rep=False)
def pallas_attn_fwd_and_bwd(
    q: bf16[b"B QL G H D"], k: bf16[b"B KL H D"], v: bf16[b"B KL H D"]
) -> tuple[bf16[b"B QL G H D"], bf16[b"B KL H D"], bf16[b"B KL H D"]]:
    dq, dk, dv = jax.grad(lambda q, k, v: jnp.sum(attention(q, k, v)), argnums=(0, 1, 2))(q, k, v)
    return dq, dk, dv


def init_random(name, shape):
    # NOTE: dq_ref and dq, dk_ref and dk, do not match if q, k are initialized with jax.random.normal
    return jax.random.uniform(jax_extra.fold_in_str(jax.random.key(0), name), shape, dtype=jnp.bfloat16) - 0.5


def test_flash_attention():
    if jax.default_backend() != "gpu":
        raise RuntimeError("Requires GPU.")
    with Mesh(mesh_utils.create_device_mesh([1], jax.devices()), ("d")):
        B, L, G, H, D = 4, 8192, 1, 8, 128
        q = init_random("q", (B, L, G, H, D))
        k = init_random("k", (B, L, H, D))
        v = init_random("v", (B, L, H, D))

        jax_fwd = jax.jit(jax_attn_fwd).lower(q, k, v).compile()
        jax_fwd_and_bwd = jax.jit(jax_attn_fwd_and_bwd).lower(q, k, v).compile()
        pal_fwd = jax.jit(pallas_attn_fwd).lower(q, k, v).compile()
        pal_fwd_and_bwd = jax.jit(pallas_attn_fwd_and_bwd).lower(q, k, v).compile()

        correctness_steps = 3
        for w in range(correctness_steps):
            q = init_random(f"q_{w}", q.shape)
            k = init_random(f"k_{w}", k.shape)
            v = init_random(f"v_{w}", v.shape)

            O_ref, logsumexp_ref = jax_fwd(q, k, v)
            dq_ref, dk_ref, dv_ref = jax_fwd_and_bwd(q, k, v)
            O, logsumexp = pal_fwd(q, k, v)
            dq, dk, dv = pal_fwd_and_bwd(q, k, v)

            jax_outputs = (O_ref, logsumexp_ref, dq_ref, dk_ref, dv_ref)
            pal_outputs = (O, logsumexp, dq, dk, dv)
            jax.block_until_ready(jax_outputs)
            jax.block_until_ready(pal_outputs)

            names = ("O", "logsumexp", "dq", "dk", "dv")
            assert_allclose = partial(np.testing.assert_allclose, rtol=1e-2, atol=1e-2)
            for jax_out, pal_out, name in zip(jax_outputs, pal_outputs, names):
                assert_allclose(jnp.float32(jax_out), jnp.float32(pal_out), err_msg=f"{name} does not match")


if __name__ == "__main__":
    from tqdm import tqdm

    if jax.default_backend() != "gpu":
        raise RuntimeError("This script requires jax on GPU.")

    print("Running correctness test...")
    test_flash_attention()
    print("Correctness test passed!")

    print("Running performance benchmark...")
    with Mesh(mesh_utils.create_device_mesh([1], jax.devices()), ("d")):
        B, L, G, H, D = 4, 8192, 1, 8, 128
        q = init_random("q", (B, L, G, H, D))
        k = init_random("k", (B, L, H, D))
        v = init_random("v", (B, L, H, D))

        jax_fwd = jax.jit(jax_attn_fwd).lower(q, k, v).compile()
        jax_fwd_and_bwd = jax.jit(jax_attn_fwd_and_bwd).lower(q, k, v).compile()
        pal_fwd = jax.jit(pallas_attn_fwd).lower(q, k, v).compile()
        pal_fwd_and_bwd = jax.jit(pallas_attn_fwd_and_bwd).lower(q, k, v).compile()

        warmup_steps = 3
        for w in tqdm(range(warmup_steps), desc="warmup"):
            q = init_random(f"q_{w}", q.shape)
            k = init_random(f"k_{w}", k.shape)
            v = init_random(f"v_{w}", v.shape)
            jax.block_until_ready(jax_fwd(q, k, v))
            jax.block_until_ready(jax_fwd_and_bwd(q, k, v))
            jax.block_until_ready(pal_fwd(q, k, v))
            jax.block_until_ready(pal_fwd_and_bwd(q, k, v))

        def benchmark(fn, q, k, v):
            t0 = time()
            jax.block_until_ready(fn(q, k, v))
            t1 = time()
            return t1 - t0

        benchmark_steps = 30
        fns = [jax_fwd, pal_fwd, jax_fwd_and_bwd, pal_fwd_and_bwd]
        fn_names = ["jax_fwd", "pal_fwd", "jax_fwd_and_bwd", "pal_fwd_and_bwd"]
        fn_times = {fn_name: 0 for fn_name in fn_names}
        for fn, fn_name in zip(fns, fn_names):
            for b in tqdm(range(benchmark_steps), desc=f"benchmark {fn_name:15}"):
                q = init_random(f"q_{fn_name}_{b}", q.shape)
                k = init_random(f"k_{fn_name}_{b}", k.shape)
                v = init_random(f"v_{fn_name}_{b}", v.shape)
                fn_times[fn_name] += benchmark(fn, q, k, v)

        def MFU(t, B, QL, KL, G_times_H, D):
            device_FLOPs_per_sec = 1671e12 / 2  # H100 NVL BFLOAT16 Tensor Core FLOPS
            FLOPs = B * G_times_H * (12 * QL * KL * D)
            optimal_t = FLOPs / device_FLOPs_per_sec
            return (optimal_t / t) * 100  # %

        for fn_name in fn_names:
            mean_time = fn_times[fn_name] / benchmark_steps
            mfu = MFU(mean_time, B, L, L, G * H, D)
            print(f"{fn_name:15} | MFU: {mfu:.2f}% | Mean time: {mean_time:.4f}s")

        fwd_multiplier = fn_times["jax_fwd"] / fn_times["pal_fwd"]
        fwd_and_bwd_multiplier = fn_times["jax_fwd_and_bwd"] / fn_times["pal_fwd_and_bwd"]
        print(f"Pallas fwd is {fwd_multiplier:.2f}× faster than JAX fwd")
        print(f"Pallas fwd and bwd is {fwd_and_bwd_multiplier:.2f}× faster than JAX fwd and bwd")
    print("Performance benchmark completed.")
