"""Functions to concisely express custom Pallas kernels in JAX, "einops"-style.

We provide:
* `pallas_call`, which corresponds to a GPU/TPU kernel invocation. This wraps JAX's
  `pallas_call` with a more concise einops-like syntax.
* `sequentially`, to be used inside a `pallas_call`, which implements a sequential loop
  a single variable. This wraps JAX's `jax.lax.fori_loop`.

For a complete example, here is a simple matmul kernel with a ReLU activation. This is
sized for an A100 GPU, splitting the matmul into in 128x32x128 tiles, running in parallel
over M and N and sequentially over K:

```
@jax.jit
@partial(shardtypes.typed_shard_map, check_rep=False)
def matmul_relu(x: bf16[b"M K"], y: bf16[b"K N"]) -> f32[b"M N"]:
    m = n = 128
    @pallas_call(['M -> m', 'N -> n', 'K -> *'], m = m, n = n)
    def kernel(x_ref: '[M] [K]', y_ref: '[K] [N]', *, dst: f32['[M] [N]']):
        k = 32
        @sequentially("K -> k", k=k, Carry='m n')
        def step(x_ref: 'm [K]', y_ref: '[K] n', acc: Carry):
            x = x_ref[:]
            y = y_ref[:]
            return acc + pl.dot(x, y)

        dst[:] = jax.nn.relu(step(x_ref, y_ref, jnp.zeros((m, n))))

    return kernel(x, y)
```

In the above:
in the `@pallas_call`'s spec, we define how tensors at the callsite are blocked within the kernel.

  * The M axis of the first tensor is split into blocks of size m
  * The N axis of the second tensor is split into blocks of size n
  * The K axis is not split into blocks at the top level

The pallas call runs the kernel on each block in parallel.

* in the `@sequentially`:
  * We sequentially loop over K, iterated in chunks of size k=32.
  * We carry a `m n`-shaped tensor to accumulate the matmul output
"""

import inspect
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from functools import partial, wraps
from typing import Mapping, Optional, Sequence, Set, Tuple, Union

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from shardlib import shardtypes
from shardlib.shardtypes import bf16, f32

shardtypes.register_with_typeguard()

################################## Specification parsing ##################################


@dataclass
class DimSpec:
    """A dimension with optional slicing. For example, 'x/d[m@k]'."""

    shard_dim: shardtypes.DimSpec  # 'x/d'
    slicing: Optional[Sequence[str]]

    @staticmethod
    def parse(spec: str):
        name, pieces = spec.split("->")
        name, pieces = name.strip(), pieces.strip()
        slicing = []
        for piece in pieces.split(" "):
            piece = piece.strip()
            slicing.append(piece)
        return DimSpec(shardtypes.DimSpec.parse(name), slicing)

    def __str__(self):
        if not self.slicing:
            return self.shard_dim.shape
        slicing = " ".join(self.slicing)
        return f"{self.shard_dim} -> {slicing}"


@dataclass
class TensorSpec:
    """A list of dimensions. For example, 'x y z', or '()' for a scalar."""

    dims: Sequence[DimSpec]

    @staticmethod
    def parse(context, spec: str):
        if spec == "()":
            return TensorSpec([])
        dims = []
        pieces = spec.split(" ")
        for piece in pieces:
            if piece.startswith("[") and piece.endswith("]") and piece[1:-1] in context:
                dims.append(context[piece[1:-1]])
            elif piece.startswith("[") and piece.endswith("]"):
                try:
                    int(piece[1:-1])  # try to parse as a literal dimension
                    literal_dim = shardtypes.DimSpec(piece[1:-1], ())
                    dims.append(DimSpec(literal_dim, []))
                except ValueError:
                    dims.append(DimSpec(shardtypes.DimSpec.parse(piece), []))
            else:
                dims.append(DimSpec(shardtypes.DimSpec.parse(piece), []))

        return TensorSpec(dims)

    def __str__(self):
        return f"[{' | '.join(map(str, self.dims))}]"


@dataclass
class TypedTensorSpec:
    """A tensor spec with a dtype. For example, 'f32['M [N]']'."""

    dtype: type
    shape: TensorSpec

    @staticmethod
    def parse(context, annotation):
        dtype = annotation.dtype
        args = annotation.__args__
        name = annotation.__name__
        assert len(args) == 1, (
            f"shape within {name} annotation should be defined within a single string, but got {len(args)} values within {name} brackets"
        )
        assert isinstance(args[0], bytes), (
            f"shape within {name}[...] annotation must be a string, but got {args[0]} of type {type(args[0])}"
        )
        shape = annotation.__args__[0].decode("utf-8")
        shape = TensorSpec.parse(context, shape)
        return TypedTensorSpec(dtype, shape)

    def __str__(self):
        return f"{self.dtype}[{self.shape}]"


def __dimension_types(ts: TensorSpec) -> Tuple[Set[str], Set[str]]:
    kernel_dims = set()
    non_kernel_dims = set()
    for dim in ts.dims:
        if dim.slicing:
            kernel_dims.add(dim.shard_dim.shape)
        if not dim.slicing:
            non_kernel_dims.add(dim.shard_dim.shape)
    return kernel_dims, non_kernel_dims


@dataclass
class ParsedSignature:
    args: Sequence[TensorSpec]
    outputs: Optional[Sequence[TypedTensorSpec]]
    carry_argname: Optional[str]
    kernel_dims: Set[str]
    non_kernel_dims: Set[str]
    arg_names: Sequence[str]


def parse_signature(context, f) -> Tuple[Sequence[TensorSpec], Sequence[TypedTensorSpec]]:
    sig = inspect.signature(f)
    args = []
    arg_names = []
    outputs = []
    carry = False
    kernel_dims = defaultdict(list)
    non_kernel_dims = defaultdict(list)
    for i, (name, p) in enumerate(sig.parameters.items()):
        arg_names.append(name)
        if carry:
            extra_args = list(sig.parameters.keys())[i:]
            raise ValueError(
                f"Carry must be the last argument, but argmuents: {extra_args} are defined after the carry argument"
            )
        if p.annotation == __CarryType:
            args.append(__CarryType)
            carry = True
            continue
        try:
            if p.kind != inspect.Parameter.KEYWORD_ONLY:
                annotation = p.annotation.decode("utf-8") if isinstance(p.annotation, bytes) else str(p.annotation)
                spec = ts = TensorSpec.parse(context, annotation.strip())
                args.append(spec)
            else:
                spec = TypedTensorSpec.parse(context, p.annotation)
                ts = spec.shape
                outputs.append(spec)
            new_kernel_dims, new_non_kernel_dims = __dimension_types(ts)
            for kd in new_kernel_dims:
                kernel_dims[kd].append(name)
            for nkd in new_non_kernel_dims:
                non_kernel_dims[nkd].append(name)
        except Exception as e:
            raise ValueError(f"Could not parse annotation for argument {name}: {e}")
    if carry and len(outputs) != 0:
        raise ValueError("Carry and outputs are mutually exclusive")
    overlapping_dims = set(kernel_dims.keys()).intersection(set(non_kernel_dims.keys()))
    if overlapping_dims:
        errors = []
        for dim in overlapping_dims:
            errors.append(
                f"{dim} is used as a kernel dimension in {kernel_dims[dim]} and as a non-kernel dimension in {non_kernel_dims[dim]}"
            )
        error_string = "\n".join(errors)
        raise ValueError(
            f"Kernel and non-kernel dimensions must be disjoint, but the following dimensions had issues:\n\t{error_string}"
        )
    return ParsedSignature(args, outputs, carry, kernel_dims, non_kernel_dims, arg_names)


def make_context(specs: Sequence[str]) -> Mapping[str, Sequence[DimSpec]]:
    ret = {}
    for spec in specs:
        dim_spec = DimSpec.parse(spec)
        ret[dim_spec.shard_dim.shape] = dim_spec
    return ret


def all_devices_are_cpus() -> bool:
    devices = jax.devices()
    return all(device.device_kind == "cpu" for device in devices)


def positional_wrapper(func):
    signature = inspect.signature(func)
    parameters = signature.parameters

    @wraps(func)
    def wrapper(*args):
        num_required_args = len(
            [
                p
                for p in parameters.values()
                if p.default == inspect.Parameter.empty
                and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)
            ]
        )
        num_keyword_only_args = len([p for p in parameters.values() if p.kind == inspect.Parameter.KEYWORD_ONLY])

        if len(args) != num_required_args + num_keyword_only_args:
            raise TypeError(
                f"{func.__name__}() takes {num_required_args + num_keyword_only_args} positional arguments but {len(args)} were given"
            )

        new_args = args[:num_required_args]
        new_kwargs = {name: arg for name, arg in zip(list(parameters)[num_required_args:], args[num_required_args:])}

        return func(*new_args, **new_kwargs)

    return wrapper


# global variables used to track indices in pallas_call and sequentially
# these are used to allow us to get_index()

__grid_order = ContextVar("kernelops.__grid_order", default=None)
__sequential_index = ContextVar("kernelops.__sequential_index", default={})  # populated by sequentially


def __grid_order_scope(grid_order, f):
    def wrapper(*args, **kwargs):
        global __grid_order
        assert __grid_order.get() is None, "Cannot nest pallas_call"
        __grid_order.set(grid_order)
        out = f(*args, **kwargs)
        __grid_order.set(None)
        return out

    return wrapper


def get_index(loop_var: str) -> int:
    grid_order = __grid_order.get()
    assert grid_order is not None, "Cannot call get_index outside of pallas_call"
    if loop_var in grid_order:
        idx = grid_order.index(loop_var)
        return pl.program_id(idx)
    return __sequential_index.get()[loop_var]


################################## pallas_call ##################################
def pallas_call(spec: Sequence[str], compiler_params={}, **vars):
    """Implements a GPU/TPU kernel using Pallas."""
    ctx = make_context(spec)

    def base_index_map(index_map_data):
        def index_map(*indices):
            result = []
            for d in index_map_data:
                total_index = 0
                for index, scale in d:
                    total_index += indices[index] * scale
                result.append(total_index)
            return tuple(result)

        return index_map

    def get_chunk_size(spec: Union[str, int]) -> int:
        if isinstance(spec, int):
            return spec
        if spec in vars:
            return vars[spec]
        raise ValueError(f"Unknown chunk size: {spec}")

    def kernel_acceptor(kernel):
        parsed = parse_signature(ctx, kernel)
        args, results = parsed.args, parsed.outputs
        args_and_results = args + [result.shape for result in results]

        # non Kernel args
        non_kernel_dims = set()
        for t in args:
            for dim in t.dims:
                if dim.slicing and dim.shard_dim.shape in non_kernel_dims:
                    raise ValueError(f"Inconsistent slicing for dimension: {dim.shard_dim.shape}")
                if not dim.slicing:
                    non_kernel_dims.add(dim.shard_dim.shape)
        # grid order takes the loop variables in the order they appear in the spec.
        grid_order = []
        for dim in non_kernel_dims:
            grid_order.append(dim)
        for t in args_and_results:
            for dim in t.dims:
                if dim.slicing == ["*"] or not dim.slicing:
                    continue
                for s in dim.slicing:
                    if s not in grid_order:
                        grid_order.append(s)

        def grid_order_index(loop_var: str) -> int:
            try:
                return grid_order.index(loop_var)
            except ValueError:
                raise ValueError(f"Unknown loop variable: {loop_var}")

        def args_acceptor(*runtime_args):
            if len(runtime_args) != len(args):
                raise ValueError(f"Expected {len(args)} arguments, got {len(runtime_args)}")

            for name, arg, runtime_arg in zip(parsed.arg_names, args, runtime_args):
                shape_spec = shardtypes.ShapeSpec(dims=[dim.shard_dim for dim in arg.dims])
                # Checks the input type, and also puts any variables into shardtypes scope.
                # We'll use that later to infer result shapes.
                if shape_spec.dims == [] and runtime_arg.shape == ():
                    # handle scalars within jax.jit
                    continue
                try:
                    shardtypes.check(runtime_arg.dtype, shape_spec, runtime_arg)
                except Exception as e:
                    raise ValueError(f"Error in argument {name}: {e}")

            loop_var_range = {}

            def arg_or_result_to_specs(arg_or_result: TensorSpec) -> Tuple[pl.BlockSpec, Sequence[int]]:
                # Given a pallas_call spec ['K -> C', 'n1 -> B b']
                # For a tensor with spec 'M [K] [N]' we form:
                #   shape = [M, K, N]
                #   block_shape = [M, C, b]
                #   index_map_data = [[], [(k, C / C)], [(n1, B / b), (n0, b / b)]]
                #
                # In addition, we store loop bounds for each loop variable, and check consistency if we see them again.
                #   k's bounds: K/C
                #   n1's bounds: N/B
                #   n0's bounds: B/b
                shape = []
                block_shape = []
                index_map_data = []
                for dim in arg_or_result.dims:
                    dim_shape = dim.shard_dim.get_per_shard_shape_from_environment()
                    if dim.slicing and not dim.slicing == ["*"]:
                        dim_block_shape = get_chunk_size(dim.slicing[-1])
                    else:
                        dim_block_shape = dim_shape

                    dim_index_map_data = []
                    index_range = dim_shape
                    for slicing in dim.slicing:
                        if slicing == "*":
                            continue
                        # Calculate loop bounds for this loop variable
                        if index_range % get_chunk_size(slicing) != 0:
                            chunk_size = get_chunk_size(slicing)
                            raise ValueError(
                                f"Chunk size {chunk_size} ({slicing}) does not divide dimension size {index_range}"
                            )
                        slicing_range = index_range // get_chunk_size(slicing)
                        if slicing in loop_var_range:
                            if loop_var_range[slicing] != slicing_range:
                                raise ValueError(
                                    f"Inconsistent loop bounds for loop variable {slicing}: {loop_var_range[slicing]} vs {slicing_range}"
                                )
                        else:
                            loop_var_range[slicing] = slicing_range
                        index_range = slicing

                        # Calculate multiplier for this loop variable
                        dim_index_map_data.append(
                            (
                                grid_order_index(slicing),
                                _div_exact(get_chunk_size(slicing), dim_block_shape),
                            )
                        )
                    if dim.shard_dim.shape in non_kernel_dims:
                        loop_var_range[dim.shard_dim.shape] = dim_shape
                        index_map_data.append([(grid_order_index(dim.shard_dim.shape), 1)])
                        block_shape.append(None)
                    else:
                        block_shape.append(dim_block_shape)
                        index_map_data.append(dim_index_map_data)
                    shape.append(dim_shape)

                return pl.BlockSpec(index_map=base_index_map(index_map_data), block_shape=tuple(block_shape)), tuple(
                    shape
                )

            in_specs = [arg_or_result_to_specs(arg)[0] for arg in args]
            out_specs = []
            out_shapes = []
            for result in results:
                block_spec, shape = arg_or_result_to_specs(result.shape)
                out_specs.append(block_spec)
                out_shapes.append(jax.ShapeDtypeStruct(shape, result.dtype))

            if len(out_shapes) == 1:
                out_shapes = out_shapes[0]
                out_specs = out_specs[0]
            else:
                out_shapes = tuple(out_shapes)
                out_specs = tuple(out_specs)

            grid = []
            for loop_var in grid_order:
                if loop_var in loop_var_range:
                    grid.append(loop_var_range[loop_var])
                else:
                    raise ValueError(f"Could not infer loop bounds for loop variable [{loop_var}]")

            return pl.pallas_call(
                __grid_order_scope(grid_order, positional_wrapper(kernel)),
                interpret=all_devices_are_cpus(),
                compiler_params=compiler_params,
                in_specs=in_specs,
                out_specs=out_specs,
                out_shape=out_shapes,
                grid=grid,
            )(*runtime_args)

        return args_acceptor

    return kernel_acceptor


class ArgKind(Enum):
    INPUT = 1
    CARRY = 2
    OUTPUT = 3


@dataclass
class LoopArrayData:
    slice_index: int
    chunk_size: int

    def slicing_expr(self, ndims: int, loop_iter):
        slicing_expr = [pl.dslice(None)] * ndims
        slicing_expr[self.slice_index] = pl.dslice(loop_iter * self.chunk_size, self.chunk_size)
        return tuple(slicing_expr)


################################## sequentially ##################################
class Carry:
    pass


__CarryType = Carry


class HBMRef:
    """
    The runtime arguments to @sequentially are HBMRefs. This allows for
    read / write access to specific blocks with a ref using [:] slicing.

    These are being converted to pl.load and pl.store, but they can be called with
    arbitrary keyword arguments using a __call__ before a load / store.

    For example, an eviction policy can be specifed on a ref called `x_ref` using
    x_ref(eviction_policy='evict_last')[:].

    Finally, we provide support for an atomic add. Though it is worth considering
    whether your kernel can be reworked to do the accumulation in a later stage
    instead of with an atomic add, as this can be more effecient.
    """

    def __init__(self, ref, shape, index, slicing):
        self.shape = shape
        self.index = index
        self.slicing = slicing
        self.ref = ref
        self.kwargs = {}

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self

    def clear(self, **kwargs):
        self.kwargs = {}

    def __getitem__(self, key):
        if isinstance(key, slice):
            out = self.load(**self.kwargs)
            self.clear()
            return out
        else:
            raise IndexError("Only indexing by `:` is supported")

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            self.store(value, **self.kwargs)
            self.clear()
        else:
            raise IndexError("Only indexing by `:` is supported for refs in @sequentially")

    def load(self, **kwargs):
        return pl.load(self.ref, self.slicing.slicing_expr(len(self.shape), self.index), **kwargs)

    def store(self, value, **kwargs):
        return pl.store(self.ref, self.slicing.slicing_expr(len(self.shape), self.index), value, **kwargs)

    def atomic_add(self, value, **kwargs):
        return pl.atomic_add(self.ref, self.slicing.slicing_expr(len(self.shape), self.index), value)


def sequentially(spec: str, Carry: Union[Sequence[str], str] = [], **vars):
    """Runs a loop sequentially, inside a pallas_call.

    An example call is @sequentially('K -> k', k=32, Carry='m n').
    The spec is similar to a single spec of pallas_call, since we are only iterating over one dimension.
    * all argument tensors must be sliced using the loop variable
    * the shapes of the carries can be set as a string of a tensor shape, IE 'm n` for a single tensor. Or a sequenceof shapes IE ['m n', 'n k'] for a tuple of tensors.
    * If a carry is specified, exactly one of the arguments of the function annotated with @sequentially
    * should be annotated with the type `kernelops.Carry`.
    * the return value of `sequentially` is the carries only

    Note that within the body of an @sequentially function, tensors are kernelops.HBMRefs.
    This is subject to change.
    """
    # TODO: Add support + typechecking for pytreedataclass.
    assert isinstance(Carry, list) or isinstance(Carry, tuple) or isinstance(Carry, str), (
        f"Carry type definition must be of type: str, tuple[str] or list[str]. Got {type(Carry)}."
    )
    ctx = make_context([spec])

    def get_chunk_size(spec: Union[str, int]) -> int:
        if isinstance(spec, int):
            return spec
        if spec in vars:
            return vars[spec]
        raise ValueError(f"Unknown chunk size: {spec}")

    def kernel_acceptor(kernel):
        parsed = parse_signature(ctx, kernel)
        args = parsed.args

        def args_acceptor(*runtime_args):
            # Visit all args and results:
            # * detect loopvar range. Check there's exactly 0 or 1 loop vars.
            # * typecheck (for non-carry values)
            # * form slicing expressions

            loop_var = None
            loop_var_range = None

            def process_arg_or_result(arg_or_result: TensorSpec, runtime_arg) -> Optional[LoopArrayData]:
                nonlocal loop_var
                nonlocal loop_var_range
                result = None
                for dim_index, dim in enumerate(arg_or_result.dims):
                    for s in dim.slicing:
                        chunk_size = get_chunk_size(s)
                        range = _div_exact(dim.shard_dim.get_per_shard_shape_from_environment(), chunk_size)
                        if loop_var is None:
                            loop_var = s
                            loop_var_range = range
                        else:
                            if result is not None:
                                raise ValueError("Only one loop variable is supported")
                            if loop_var != s:
                                raise ValueError(f"Only one loop variable is supported: {loop_var} vs {s}")
                            if loop_var_range != range:
                                raise ValueError(f"Inconsistent loop variable range: {loop_var_range} vs {range}")

                        result = LoopArrayData(dim_index, chunk_size)

                if result is not None:
                    # Checks the input type, and also puts any variables into shardtypes scope.
                    # We'll use that later to infer loop-carry shapes.
                    shape_spec = shardtypes.ShapeSpec(dims=[dim.shard_dim for dim in arg.dims])
                    shardtypes.check(
                        runtime_arg.dtype, shape_spec, jax.ShapeDtypeStruct(runtime_arg.shape, runtime_arg.dtype)
                    )

                return result

            arg_slicing = []
            carried_arg = None
            argnames = [p.name for p in inspect.signature(kernel).parameters.values()]
            for name, arg, runtime_arg in zip(argnames, args, runtime_args):
                if arg == __CarryType:
                    carried_arg = runtime_arg
                    continue
                try:
                    data = process_arg_or_result(arg, runtime_arg)
                except Exception as e:
                    raise ValueError(f"Could not process arg {name}: {e}")
                if data is None:
                    raise ValueError(f"Expected slicing on all input arguments, but {arg} is not sliced")
                arg_slicing.append(data)

            if loop_var is None:
                raise ValueError("Expected slicing on at least one result argument")
            if loop_var_range is None:
                raise ValueError("Could not infer loop variable range")

            # Now generate the loop body. It takes and returns carries.
            def loop_body(loop_iter, input_carries):
                # Load args
                seq_index = __sequential_index.get()
                seq_index[loop_var] = loop_iter

                arg_refs = []
                for runtime_arg, slicing in zip(runtime_args[: len(args)], arg_slicing):
                    arg_refs.append(HBMRef(runtime_arg, runtime_arg.shape, loop_iter, slicing))

                if input_carries is not None:
                    return kernel(*arg_refs, input_carries)
                else:
                    return kernel(*arg_refs)

            if isinstance(Carry, Sequence) and not isinstance(Carry, str) and len(Carry):
                assert isinstance(carried_arg, tuple), (
                    f"Carry must be initialized with a tuple of {len(Carry)} tensors, but Carry is of type {type(Carry)}"
                )
                assert len(carried_arg) == len(Carry), (
                    f"Carry must be a tuple of {len(Carry)} tensor(s) with shapes f{Carry}."
                )

            return jax.lax.fori_loop(0, loop_var_range, loop_body, carried_arg)

        return args_acceptor

    return kernel_acceptor


def _div_exact(a, b):
    if a % b != 0:
        raise ValueError(f"{a} is not divisible by {b}")
    return a // b


################################## test ##################################
if __name__ == "__main__":
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh

    MESH = Mesh(mesh_utils.create_device_mesh([1], jax.devices()[:1]), ("d"))

    with MESH:
        m = n = 128
        k = 32

        @jax.jit
        @partial(shardtypes.typed_shard_map, check_rep=False)
        def matmul_relu(x: bf16[b"M K"], y: bf16[b"K N"]) -> f32[b"M N"]:
            @pallas_call(["M -> m", "N -> n", "K -> *"], m=m, n=n)
            def kernel(x_ref: b"[M] [K]", y_ref: b"[K] [N]", *, dst: f32[b"[M] [N]"]):
                k = 32

                @sequentially("K -> k", k=k, Carry="m n")
                def step(x_ref: b"m [K]", y_ref: b"[K] n", acc: Carry):
                    x = x_ref[:]
                    y = y_ref[:]
                    return acc + pl.dot(x, y)

                dst[:] = jax.nn.relu(step(x_ref, y_ref, jnp.zeros((m, n))))

            return kernel(x, y)

        result = matmul_relu(jnp.ones((128, 256), jnp.bfloat16), jnp.ones((256, 512), jnp.bfloat16))
        print(result.shape)
        print(result)

        @jax.jit
        @shardtypes.scope
        @partial(shardtypes.typed_shard_map, check_rep=False)
        # Using a kernel with conditional execution to make a block diagonal matrix with
        # a shape like that of x. The value on the block diagonal is c.
        def conditional_kernel(x: bf16[b"M N"], c: bf16[b""]) -> f32[b"M N"]:
            @pallas_call(["M -> m", "N -> *"], m=128)
            def kernel(x_ref: b"[M] [N]", c_ref: b"()", *, dst: f32[b"[M] [N]"]):
                dst[:] = jnp.zeros_like(dst[:])
                c = pl.load(c_ref, ())

                @sequentially("N -> n", n=128)
                def step(x: b"m [N]", out: b"m [N]"):
                    def on_diagonal():
                        out[:] = jnp.float32(jnp.broadcast_to(c, out[:].shape))

                    def off_diagonal():
                        pass

                    jax.lax.cond(get_index("m") == get_index("n"), on_diagonal, off_diagonal)

                step(x_ref, dst)

            return kernel(x, c)

        M = N = 2048
        result = conditional_kernel(jnp.ones((M, N), jnp.bfloat16), jnp.bfloat16(1337.0))
        print(result.shape)
        want = jax.scipy.linalg.block_diag(*[jnp.full((128, 128), jnp.bfloat16(1337.0)) for _ in range(M // 128)])
        print(want)
        assert jnp.all(result == want)
