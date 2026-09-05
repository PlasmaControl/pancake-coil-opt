import jax.numpy as jnp
import pytest

from pancakecoilopt.batching import vmap_chunked


def test_chunked_matches_plain_vmap_for_a_single_argument():
    f = lambda x: x**2 + 1
    xs = jnp.arange(10.0)

    unchunked = vmap_chunked(f, in_axes=0, chunk_size=None)(xs)
    chunked = vmap_chunked(f, in_axes=0, chunk_size=3)(xs)  # 10 doesn't divide by 3

    assert jnp.allclose(chunked, unchunked)
    assert jnp.allclose(chunked, xs**2 + 1)


def test_chunked_matches_plain_vmap_for_multiple_mapped_arguments():
    f = lambda a, b: a * b + jnp.sum(a)
    a = jnp.arange(7.0)
    b = jnp.arange(7.0)[::-1]

    chunked = vmap_chunked(f, in_axes=(0, 0), chunk_size=2)(a, b)
    unchunked = vmap_chunked(f, in_axes=(0, 0), chunk_size=None)(a, b)

    assert jnp.allclose(chunked, unchunked)


def test_chunked_supports_a_broadcast_static_argument():
    # in_axes=None means "pass this argument through unchanged to every
    # call" rather than mapping over its leading axis -- e.g. the source
    # points stay fixed while chunking over target points.
    def f(x, scale):
        return x * scale

    xs = jnp.arange(9.0)
    scale = 3.0

    chunked = vmap_chunked(f, in_axes=(0, None), chunk_size=4)(xs, scale)
    assert jnp.allclose(chunked, xs * scale)


def test_chunk_size_that_evenly_divides_the_input_still_works():
    f = lambda x: x + 1
    xs = jnp.arange(8.0)
    chunked = vmap_chunked(f, in_axes=0, chunk_size=4)(xs)
    assert jnp.allclose(chunked, xs + 1)


def test_chunk_size_larger_than_the_input_still_works():
    f = lambda x: x + 1
    xs = jnp.arange(3.0)
    chunked = vmap_chunked(f, in_axes=0, chunk_size=1000)(xs)
    assert jnp.allclose(chunked, xs + 1)


def test_unsupported_in_axes_value_raises():
    with pytest.raises(NotImplementedError):
        vmap_chunked(lambda x: x, in_axes=1, chunk_size=2)
