"""Chunked/batched vmap: evaluate a function over many inputs without ever
holding every input's intermediate activations in memory at once.
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
    argument is passed through unchanged to every call) are supported.
    The only two cases physics.py's batched functions actually need.
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
