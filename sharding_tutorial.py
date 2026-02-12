# %% [markdown]
# # Sharding & Tiling Matmuls — A Hands-On Tutorial
#
# Shows that sharded matmuls produce the same results as numpy.
# We simulate 8 devices on CPU so we can print per-device shards.

# %%
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import inspect
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import Mesh

from shardlib import shardops
from shardlib.shardtypes import f32, make_partition_specs, Scope

def typed_shard_map(f, **kwargs):
    """Like shardtypes.typed_shard_map but without typechecked (works in notebooks)."""
    sig = inspect.signature(f)
    def wrapped(*args):
        mesh = jax._src.mesh.thread_resources.env.physical_mesh
        in_specs = tuple(make_partition_specs(p.annotation) for p in sig.parameters.values())
        out_specs = make_partition_specs(sig.return_annotation)
        return jax.experimental.shard_map.shard_map(
            f, in_specs=in_specs, out_specs=out_specs, mesh=mesh, **kwargs
        )(*args)
    return wrapped

print(f"Devices available: {jax.device_count()}")

# %% [markdown]
# ## Part 1: Sharding Non-Contracting Dimensions
#
# `Y = X @ W` contracts over K. Sharding M (rows of X) or N (cols of W) needs
# no communication — each device just computes its own slice.

# %%
# Row-sharded: M/d — each device gets a subset of rows of X
d = 2

with Mesh(np.array(jax.devices()[:d]).reshape(d), ("d",)):
    X = jnp.array([[1, 2],
                    [3, 4],
                    [5, 6],
                    [7, 8]], dtype=jnp.float32)  # (4, 2)
    W = jnp.array([[1, 0, 1, 0],
                    [0, 1, 0, 1]], dtype=jnp.float32)  # (2, 4)

    with Scope():
        def fn(x: f32[b"M/d K"], w: f32[b"K N"]) -> f32[b"M/d N"]:
            print("x =", x)
            print("w =", w)
            result = shardops.einsum_unreduced("M/d K, K N -> M/d N", x, w)
            print("result =", result)
            return result
        sharded = typed_shard_map(fn, check_rep=False)(X, W)

    print("\nsharded result:\n", sharded)
    print("numpy result:\n", np.array(X) @ np.array(W))

# %%
# Column-sharded: N/t — each device gets a subset of columns of W
t = 2

with Mesh(np.array(jax.devices()[:t]).reshape(t), ("t",)):
    X = jnp.array([[1, 2],
                    [3, 4],
                    [5, 6],
                    [7, 8]], dtype=jnp.float32)  # (4, 2)
    W = jnp.array([[1, 0, 1, 0],
                    [0, 1, 0, 1]], dtype=jnp.float32)  # (2, 4)

    with Scope():
        def fn(x: f32[b"M K"], w: f32[b"K N/t"]) -> f32[b"M N/t"]:
            print("x =", x)
            print("w =", w)
            result = shardops.einsum_unreduced("M K, K N/t -> M N/t", x, w)
            print("result =", result)
            return result
        sharded = typed_shard_map(fn, check_rep=False)(X, W)

    print("\nsharded result:\n", sharded)
    print("numpy result:\n", np.array(X) @ np.array(W))

# %% [markdown]
# ## Part 2: Sharding the Contracting Dimension
#
# When K is sharded, each device computes a partial matmul. We need
# `psum` or `psum_scatter` to sum the partial results across devices.

# %%
# Sharded dot product: K/d — each device gets a slice of the vectors
d = 2

with Mesh(np.array(jax.devices()[:d]).reshape(d), ("d",)):
    a = jnp.array([1, 2, 3, 4], dtype=jnp.float32)  # (4,)
    b = jnp.array([2, 1, 2, 1], dtype=jnp.float32)  # (4,)

    with Scope():
        def fn(a: f32[b"K/d"], b: f32[b"K/d"]) -> f32[b"K/d"]:
            print("a =", a)
            print("b =", b)
            partial = jnp.sum(a * b, keepdims=True)
            print("partial (before psum) =", partial)
            result = jax.lax.psum(partial, "d")
            print("result (after psum) =", result)
            return jnp.broadcast_to(result, a.shape)
        sharded = typed_shard_map(fn, check_rep=False)(a, b)

    print("\nsharded result:", sharded[0])
    print("numpy result:", np.dot(np.array(a), np.array(b)))

# %%
# K-sharded matmul: same idea but for matrices
d = 2

with Mesh(np.array(jax.devices()[:d]).reshape(d), ("d",)):
    X2 = jnp.array([[1, 2, 3, 4],
                     [5, 6, 7, 8]], dtype=jnp.float32)  # (2, 4)
    W2 = jnp.array([[1, 0],
                     [0, 1],
                     [1, 0],
                     [0, 1]], dtype=jnp.float32)  # (4, 2)

    with Scope():
        def fn(x: f32[b"M K/d"], w: f32[b"K/d N"]) -> f32[b"M N"]:
            print("x =", x)
            print("w =", w)
            partial = shardops.einsum_unreduced("M K/d, K/d N -> M N", x, w)
            print("partial (before psum) =", partial)
            result = jax.lax.psum(partial, "d")
            print("result (after psum) =", result)
            return result
        sharded = typed_shard_map(fn, check_rep=False)(X2, W2)

    print("\nsharded result:\n", sharded)
    print("numpy result:\n", np.array(X2) @ np.array(W2))

# %% [markdown]
# ## Part 3: Megatron-Style MLP
#
# `y = relu(x @ W1) @ W2` — W1 column-sharded (`F/t`), W2 row-sharded (`F/t`).
# Intermediate `(B, F/t)` feeds directly into W2. Only one `psum` at the end.

# %%
t = 2

with Mesh(np.array(jax.devices()[:t]).reshape(t), ("t",)):
    x_mlp = jnp.array([[1, 1],
                        [1, 1],
                        [1, 1],
                        [1, 1]], dtype=jnp.float32)  # (B=4, M=2)
    W1 = jnp.array([[0.5, 0.5, 0.5, 0.5],
                     [0.5, 0.5, 0.5, 0.5]], dtype=jnp.float32)  # (M=2, F=4)
    W2 = jnp.array([[0.5, 0.5],
                     [0.5, 0.5],
                     [0.5, 0.5],
                     [0.5, 0.5]], dtype=jnp.float32)  # (F=4, M=2)

    with Scope():
        def fn(x: f32[b"B Mdl"], w1: f32[b"Mdl F/t"], w2: f32[b"F/t Mdl"]) -> f32[b"B Mdl"]:
            print("x =", x)
            print("w1 =", w1)
            print("w2 =", w2)
            h = jax.nn.relu(jnp.einsum("bm,mf->bf", x, w1))
            print("h (after layer 1, no comm needed) =", h)
            y_partial = jnp.einsum("bf,fm->bm", h, w2)
            print("y_partial (before psum) =", y_partial)
            result = jax.lax.psum(y_partial, "t")
            print("result (after psum) =", result)
            return result
        sharded = typed_shard_map(fn, check_rep=False)(x_mlp, W1, W2)

    print("\nsharded result:\n", sharded)
    print("numpy result:\n", np.maximum(0, np.array(x_mlp) @ np.array(W1)) @ np.array(W2))

# %% [markdown]
# ## Part 4: Tiled Matmul (no K-split)
#
# Each output tile `Y[i,j] = X[i_rows, :] @ W[:, j_cols]`.

# %%
X_t = jnp.array([[ 1,  2,  3],
                  [ 4,  5,  6],
                  [ 7,  8,  9],
                  [10, 11, 12]], dtype=jnp.float32)  # (4, 3)
W_t = jnp.array([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [1, 1, 0, 0]], dtype=jnp.float32)  # (3, 4)

def tiled_matmul(X, W, m, n):
    M, K = X.shape
    _, N = W.shape
    Y = jnp.zeros((M, N))
    for i in range(M // m):
        for j in range(N // n):
            Y = Y.at[i*m:(i+1)*m, j*n:(j+1)*n].set(
                X[i*m:(i+1)*m, :] @ W[:, j*n:(j+1)*n])
    return Y

print("tiled:\n", tiled_matmul(X_t, W_t, 2, 2))
print("numpy:\n", np.array(X_t) @ np.array(W_t))

# %% [markdown]
# ## Part 5: Tiled Matmul with K-splitting
#
# Split K into chunks, accumulate partial products.
# This maps to `@pallas_call` + `@sequentially` in `kernelops.py`.

# %%
X_t2 = jax.random.normal(jax.random.PRNGKey(0), (8, 6))
W_t2 = jax.random.normal(jax.random.PRNGKey(1), (6, 4))

def tiled_matmul_k(X, W, m, n, k):
    M, K = X.shape
    _, N = W.shape
    Y = jnp.zeros((M, N))
    for i in range(M // m):
        for j in range(N // n):
            acc = jnp.zeros((m, n))
            for kk in range(K // k):
                acc = acc + X[i*m:(i+1)*m, kk*k:(kk+1)*k] @ W[kk*k:(kk+1)*k, j*n:(j+1)*n]
            Y = Y.at[i*m:(i+1)*m, j*n:(j+1)*n].set(acc)
    return Y

result = tiled_matmul_k(X_t2, W_t2, 4, 2, 3)
print("tiled:\n", result)
print("numpy:\n", np.array(X_t2) @ np.array(W_t2))
print(f"match? {jnp.allclose(result, np.array(X_t2) @ np.array(W_t2), atol=1e-4)}")
