"""Coil curve geometry.

A "bent" pancake coil is not a flat spiral of constant radius: its winding
radius is allowed to vary along the coil's height (z), tracing out a curve

    r = f(z) = A + B*z^2 + C*z^4 + D*z^6 + ...

Only even powers of z are used, so the curve is symmetric about z=0 (the
coil's midplane) — the winding radius at +z equals the radius at -z. The
coefficients (A, B, C, ...) are exactly the free parameters an optimizer is
allowed to vary.

Everything here works with that one curve and produces two things needed
downstream:
  1. Its arc length (how much conductor tape is needed to trace it), and the
     inverse of that (given a target tape length, how tall is the coil?).
  2. A cloud of discrete (r, z) "source points" approximating the coil's
     right-half cross-section, each carrying a current density and an area
     element — the raw input the Biot-Savart field calculation sums over.
"""

import functools
import math
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import jax.scipy.integrate
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import NDArray

from . import schemas


@jax.jit
def compute_curve(z: Array, curve_parameters: Array) -> Array:
    """Evaluate r = A + B*z^2 + C*z^4 + ... at the given z values.

    `jnp.polyval` evaluates a polynomial in a single variable, so we hand it
    z^2 instead of z: polyval([D, C, B, A], z^2) = A + B*z^2 + C*z^4 + D*z^6.
    curve_parameters is stored lowest-order-first (A, B, C, D, ...), so it's
    reversed to match polyval's highest-order-first convention.
    """
    return jnp.polyval(curve_parameters[::-1], z**2)


def _as_even_polynomial(curve_parameters: Array) -> Array:
    """Expand (A, B, C, ...) into full polyval-ready coefficients for
    A + B*z^2 + C*z^4 + ..., i.e. one with the odd-power slots filled with
    zero: [..., 0, D, 0, C, 0, B, 0, A] (highest order first).
    """
    poly_coefficients = jnp.zeros(2 * len(curve_parameters) - 1)
    return poly_coefficients.at[::2].set(curve_parameters[::-1])


@jax.jit
def compute_curve_derivative(z: Array, curve_parameters: Array) -> Array:
    """Evaluate dr/dz = 2*B*z + 4*C*z^3 + 6*D*z^5 + ..."""
    derivative_coefficients = jnp.polyder(_as_even_polynomial(curve_parameters))
    return jnp.polyval(derivative_coefficients, z)


@jax.jit
def compute_curve_unit_tangent(z: Array, curve_parameters: Array) -> Array:
    """Unit tangent vector (dr, dz)/|...| to the curve at each z.

    Needed later to measure the angle between the local magnetic field and
    the HTS tape's own direction — critical current drops sharply when the
    field is more perpendicular to the tape.
    """
    dr_over_dz = compute_curve_derivative(z, curve_parameters)
    one_over_norm = jax.lax.rsqrt(1 + dr_over_dz**2)
    return jnp.stack([dr_over_dz * one_over_norm, one_over_norm], axis=-1)


@jax.jit
def compute_area_below_curve(
    z_start: float, z_end: float, curve_parameters: Array
) -> Array:
    """Integral of r=f(z) dz from z_start to z_end (i.e. the area between the
    curve and the z-axis), via the polynomial's antiderivative.
    """
    integral_coefficients = jnp.polyint(_as_even_polynomial(curve_parameters))
    return jnp.polyval(integral_coefficients, z_end) - jnp.polyval(
        integral_coefficients, z_start
    )


@jax.jit
def compute_arc_length(z_top: float, curve_parameters: Array) -> Array:
    """Arc length of r=f(z) from 0 to z_top: integral of sqrt(1 + (dr/dz)^2) dz.

    There's no closed-form antiderivative for a general even polynomial, so
    this integrates numerically (trapezoid rule on 400 points spanning the
    full [0, z_top] range — `endpoint=True` matters here: dropping the last
    sample would silently shrink the integration domain and bias every
    arc length low). Negative z_top gives a negative length, so
    `2 * compute_arc_length(z_top, ...)` matches the total conductor length
    from -z_top to z_top regardless of z_top's sign — used by the Newton
    solve below.
    """
    z_vals = jnp.linspace(0, jnp.abs(z_top), 400, endpoint=True)
    arc_length = jax.scipy.integrate.trapezoid(
        jnp.sqrt(1 + compute_curve_derivative(z_vals, curve_parameters) ** 2),
        z_vals,
    )
    return jnp.where(z_top > 0, arc_length, -arc_length)


def solve_f_of_x_equals_0_with_grad(
    f: Callable[[Array], Array], x0: float, iterations: int = 10
) -> ArrayLike:
    """Newton's method: x_{n+1} = x_n - f(x_n)/f'(x_n), for a fixed number of
    steps (no convergence check — `iterations` is chosen generously instead,
    since this needs to run inside jax.jit/vmap where loop counts must be
    static). f'(x) comes from autodiff (`jax.grad`) rather than a
    hand-derived formula, so `f` can be swapped out freely.
    """
    f = jax.jit(f)
    grad_f = jax.jit(jax.grad(f))

    @jax.jit
    def newton_step(x):
        return x - f(x) / grad_f(x)

    return jax.lax.fori_loop(
        0, iterations, lambda _, x: newton_step(x), jnp.array([x0])
    )[0]


def solve_f_of_x_equals_0_with_finite_differences(
    f: Callable[[Array], Array], x0: float, iterations: int = 10
) -> ArrayLike:
    """Same fixed-iteration Newton's method as `solve_f_of_x_equals_0_with_grad`,
    but with a central-difference derivative instead of autodiff.

    Used for the self-consistent current-density solve (physics.py), where
    `f` involves a `jnp.min` over many field-point evaluations; autodiff
    through that pattern is well-defined in principle but this route has
    proven more numerically robust in practice.
    """
    f = jax.jit(f)

    # Step size ~1e-3 relative to x0. A fixed absolute step (like x0/1e7)
    # underflows float32 precision for the large x0 values this solves for
    # in practice (current densities ~1e7): the central difference then
    # subtracts two nearly-equal floats, and catastrophic cancellation
    # leaves a near-zero or garbage derivative, sending Newton's step to
    # NaN. A relative step avoids that regardless of x0's scale.
    step_size = jnp.abs(x0) * 1e-3 + 1e-12

    @jax.jit
    def grad_f(x):
        return (f(x + step_size) - f(x - step_size)) / (2 * step_size)

    @jax.jit
    def newton_step(x):
        g = grad_f(x)
        # Guard a near-zero derivative so a step can't blow up to NaN.
        g_safe = jnp.where(jnp.abs(g) < 1e-12, 1.0, g)
        return x - f(x) / g_safe

    return jax.lax.fori_loop(
        0, iterations, lambda _, x: newton_step(x), jnp.array([x0])
    )[0]


@jax.jit
def find_z_bound_for_a_given_arc_length(
    arc_length: float, curve_parameters: Array
) -> ArrayLike:
    """Invert compute_arc_length: given a total conductor length, find the z
    such that the curve from -z to z has exactly that length. This is how a
    coil's height is determined from a fixed tape length rather than the
    other way around — an optimizer varying the curve shape (A, B, C, ...)
    should not also need to independently pick a consistent height.
    """

    def f(z: Array) -> Array:
        return 2 * compute_arc_length(z[0], curve_parameters) - arc_length

    return solve_f_of_x_equals_0_with_grad(f, x0=1.0, iterations=10)


@functools.partial(jax.jit, static_argnames=["width", "total_arc_length", "nr", "nz"])
def compute_source_points_at_right_half_for_bent_coil(
    curve_parameters: Array,
    total_arc_length: float,
    width: float,
    current_density: float,
    nr: int,
    nz: int,
) -> tuple[Array, Array, Array, Array]:
    """Discretize the right half (r > 0) of the coil's cross-section into an
    nz-by-nr grid of point sources for the Biot-Savart sum.

    For each of nz heights along the coil, the winding starts at r=f(z) (the
    curve) and extends radially outward by `width` (the total radial build of
    all the stacked tape turns at that height); nr points sample that radial
    span. Each point carries the same current density and an equal share
    `dA` of the total cross-sectional area, so summing (current_density * dA)
    over all points reproduces the coil's total current.

    Returns:
        r_points, z_points: flattened (nr*nz,) coordinates of each source point.
        J_source: current density at each point (currently uniform).
        dA: cross-sectional area represented by one point.
    """
    z_top = find_z_bound_for_a_given_arc_length(total_arc_length, curve_parameters)
    z_source = jnp.linspace(-z_top, z_top, nz)

    # A curve dipping to r <= 0 is unphysical and sends the Biot-Savart
    # elliptic-integral kernel (which divides by r) to NaN. Clamping to a
    # small positive floor keeps the objective finite so an optimizer
    # exploring such shapes gets steered away instead of crashing.
    r_source = jnp.maximum(compute_curve(z_source, curve_parameters), 1e-4)

    delta_r = width / nr
    r_offsets = jnp.arange(nr)
    r_grid = r_source[:, None] + delta_r * r_offsets[None, :]  # (nz, nr)

    r_points = r_grid.flatten()
    z_points = jnp.repeat(z_source, nr)
    J_source = jnp.full_like(r_points, current_density)

    # Total area = area under the outer edge (curve shifted out by `width`)
    # minus area under the inner edge (the bare curve), split evenly among
    # all nr*nz points.
    curve_parameters_inner = curve_parameters.at[0].set(0.0)
    curve_parameters_outer = curve_parameters.at[0].set(width)
    total_area = compute_area_below_curve(
        -z_top, z_top, curve_parameters_outer
    ) - compute_area_below_curve(-z_top, z_top, curve_parameters_inner)
    dA = total_area / (nr * nz)

    return r_points, z_points, J_source, dA


def _expand_to_at_least(
    small_value: float, big_value: float, maximum_small_value: float,
    minimum_big_value: float,
) -> tuple[float, float]:
    """Widen a requested (small_value, big_value) range so it covers at
    least [maximum_small_value, minimum_big_value] — used to guarantee a
    plot/grid range always contains the whole coil: a narrower request is
    silently expanded back out to the coil's own extent, while a wider
    request (going beyond the coil) is honored as-is.
    """
    small_value = min(small_value, maximum_small_value)
    big_value = max(big_value, minimum_big_value)
    return small_value, big_value


def compute_r_z_range(
    pancake_coil: "schemas.PancakeCoil",
    r_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """The (r, z) extent to grid a coil over: by default, symmetric about
    both axes and just big enough to contain the whole coil
    (+/- maximum_outer_radius, +/- half of maximum_height).

    An explicit r_range/z_range can zoom OUT beyond that (e.g. to see the
    field further from the coil), but never IN — a narrower request would
    crop part of the coil out of the grid, so it's silently widened back to
    the coil's own default extent instead of honored as-is.
    """
    default_r_end = pancake_coil.maximum_outer_radius
    default_r_start = -default_r_end
    if r_range is not None:
        r_start, r_end = _expand_to_at_least(
            r_range[0], r_range[1], default_r_start, default_r_end
        )
    else:
        r_start, r_end = default_r_start, default_r_end

    default_z_end = pancake_coil.maximum_height / 2
    default_z_start = -default_z_end
    if z_range is not None:
        z_start, z_end = _expand_to_at_least(
            z_range[0], z_range[1], default_z_start, default_z_end
        )
    else:
        z_start, z_end = default_z_start, default_z_end

    return (r_start, r_end), (z_start, z_end)


def generate_a_grid(
    pancake_coil: "schemas.PancakeCoil",
    nr: int,
    nz: int | None = None,
    r_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
    numpy_or_jax: Literal["numpy", "jax"] = "jax",
) -> tuple[Array, Array] | tuple[NDArray, NDArray]:
    """A mesh grid (R, Z) covering the coil (or an explicit r_range/z_range),
    for field evaluation and plotting. If nz isn't given, it's chosen to
    keep grid cells roughly square given the aspect ratio of the range.
    """
    (r_start, r_end), (z_start, z_end) = compute_r_z_range(
        pancake_coil, r_range, z_range
    )
    if nz is None:
        nz = math.ceil((z_end - z_start) / (r_end - r_start) * nr)

    module = np if numpy_or_jax == "numpy" else jnp
    r = module.linspace(r_start, r_end, nr)
    z = module.linspace(z_start, z_end, nz)
    return module.meshgrid(r, z, indexing="xy")
