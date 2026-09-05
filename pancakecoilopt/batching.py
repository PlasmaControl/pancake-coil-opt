"""Chunked/batched vmap: evaluate a function over many inputs without ever
holding every input's intermediate activations in memory at once.

physics.py's Biot-Savart kernel is naively vectorized with `jax.vmap`,
computing every source-point (or grid-point) contribution in parallel. For a
modest coil discretization that's fine, but a fine field grid or many
thousand source rings can make the fully-vectorized intermediates exceed
available memory. `vmap_chunked` gives the same result as `jax.vmap`, but
processes the mapped axis `chunk_size` elements at a time.

An earlier version of this project hand-rolled this chunking logic by
borrowing NetKet's `_scanmap` machinery, which reaches into several private
JAX internals (`jax._src.api_util.argnums_partial`,
`jax._src.lax.control_flow.loops._batch_and_remainder`) that have no
stability guarantee across JAX versions. Newer JAX exposes the same
"batch a vmap through scan" behavior directly as the public `batch_size`
argument to `jax.lax.map`, so that's used here instead.
"""

from collections.abc import Callable

import jax


def vmap_chunked(
    f: Callable,
    in_axes: int | tuple[int | None, ...] = 0,
    *,
    chunk_size: int | None = None,
) -> Callable:
    """Like `jax.vmap(f, in_axes=in_axes)`, but bounds peak memory by
    processing the mapped axis `chunk_size` elements at a time (via
    `jax.lax.map`'s `batch_size`) instead of all at once.

    Only `in_axes` entries of 0 (this argument is mapped over) or None (this
    argument is passed through unchanged to every call) are supported —
    the only two cases physics.py's batched functions actually need.
    """
    if isinstance(in_axes, int):
        in_axes = (in_axes,)

    unsupported_axes = set(in_axes) - {0, None}
    if unsupported_axes:
        message = f"in_axes entries must be 0 or None, got {in_axes}"
        raise NotImplementedError(message)

    if chunk_size is None:
        return jax.vmap(f, in_axes=in_axes)

    mapped_positions = [i for i, axis in enumerate(in_axes) if axis == 0]

    def chunked(*args):
        static_args_by_position = {
            i: args[i] for i, axis in enumerate(in_axes) if axis is None
        }

        def f_over_mapped_args(mapped_args: tuple):
            full_args = [static_args_by_position.get(i) for i in range(len(args))]
            for position, value in zip(mapped_positions, mapped_args, strict=True):
                full_args[position] = value
            return f(*full_args)

        mapped_args = tuple(args[i] for i in mapped_positions)
        return jax.lax.map(f_over_mapped_args, mapped_args, batch_size=chunk_size)

    return chunked
