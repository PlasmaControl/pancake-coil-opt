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
     element — the raw input the Biot-Savart field calculation (physics.py,
     built next) sums over.
"""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.scipy.integrate
from jax import Array
from jax.typing import ArrayLike


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
