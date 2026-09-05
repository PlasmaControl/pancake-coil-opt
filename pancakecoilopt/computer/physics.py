"""Magnetic field physics: Biot-Savart field from the coil's source points, and
the critical-current-density model for the HTS tape.

The coil is discretized (basics.py) into a cloud of circular current loops:
each source point (r_source, z_source) with current density J_phi represents
a thin ring of current I = J_phi * dA at that (r, z). The total field at any
point is the superposition of every ring's field — an exact closed form
exists for a single current loop, in terms of complete elliptic integrals,
so no discretization is needed on the field-evaluation side (only in how
finely the coil itself is sliced into rings).
"""

from collections.abc import Callable
from typing import Literal, overload

import jax
import jax.numpy as jnp
import jaxellip
from jax import Array
from jax.typing import ArrayLike
from scipy.constants import mu_0, pi

from ..batching import vmap_chunked
from . import basics


@jax.jit
def compute_magnetic_field_at_a_point_from_source_points(
    r_source: Array,
    z_source: Array,
    corresponding_J_phi: Array,
    dA: float,
    r: float,
    z: float,
) -> tuple[Array, Array]:
    """Field (B_r, B_z) at (r, z) from a set of circular current loops, one
    per source point, each carrying current I = J_phi * dA.

    For a single loop of radius a at z=0 carrying current I, the field at
    (r, z) is (Simpson/Jackson's classic result, in terms of complete
    elliptic integrals K and E of parameter k^2 = 4*a*r / ((a+r)^2 + z^2)):

        B_z = mu_0*I / (2*pi*sqrt((a+r)^2+z^2)) *
              [ K(k^2) + (a^2 - r^2 - z^2)/((a-r)^2+z^2) * E(k^2) ]
        B_r = mu_0*I*z / (2*pi*r*sqrt((a+r)^2+z^2)) *
              [ -K(k^2) + (a^2 + r^2 + z^2)/((a-r)^2+z^2) * E(k^2) ]

    This function evaluates that formula for every source loop at once and
    sums the contributions (superposition). `grid_spacing_squared = dA/4` is
    added into every squared-distance term as a softening length: without
    it, a field point that lands exactly on a source ring (a=r, z=z_source)
    would divide by zero in `(a-r)^2+z^2`. Since dA already represents the
    physical size of that ring's cross-section, `sqrt(dA)/2` is a natural
    scale for how close is "on top of it" — the field is only sampled at
    physically distinguishable points anyway.

    Args:
        r_source, z_source: Coordinates of each source ring.
        corresponding_J_phi: Azimuthal current density at each source ring.
        dA: Cross-sectional area represented by one source ring.
        r, z: The field point.

    Returns:
        (B_r, B_z) at (r, z), summed over all source rings.
    """
    grid_spacing_squared = dA / 4

    k_squared = (
        4
        * r_source
        * r
        / ((r_source + r) ** 2 + (z - z_source) ** 2 + grid_spacing_squared)
    )
    ellip_K = jaxellip.ellipk(k_squared)
    ellip_E = jaxellip.ellipe(k_squared)

    corresponding_current = corresponding_J_phi * dA
    delta_z = z - z_source

    common_multiplier = (
        mu_0
        * corresponding_current
        / (2 * pi * jnp.sqrt(delta_z**2 + (r_source + r) ** 2 + grid_spacing_squared))
    )
    common_denominator = delta_z**2 + (r - r_source) ** 2 + grid_spacing_squared

    B_z = jnp.sum(
        common_multiplier
        * ((r_source**2 - delta_z**2 - r**2) * ellip_E / common_denominator + ellip_K)
    )

    # On-axis (r=0), B_r is 0 by symmetry — but the formula above divides by
    # r to get there, which would poison the gradient (0/0 in the unused
    # branch still propagates a NaN through reverse-mode autodiff even
    # though jnp.where discards its value). Substituting a dummy r_safe=1
    # only in that division keeps the branch finite; jnp.where still
    # selects the correct 0.0 for the actual output.
    r_safe = jnp.where(r == 0, 1.0, r)
    B_r = jnp.where(
        r == 0,
        0.0,
        jnp.sum(
            common_multiplier
            * delta_z
            / r_safe
            * (
                (r_source**2 + delta_z**2 + r**2) * ellip_E / common_denominator
                - ellip_K
            )
        ),
    )

    return B_r, B_z


@jax.jit
def compute_magnetic_field_magnitude_at_a_point_from_source_points(
    r_source: Array,
    z_source: Array,
    corresponding_J_phi: Array,
    dA: float,
    r: float,
    z: float,
) -> Array:
    B_r, B_z = compute_magnetic_field_at_a_point_from_source_points(
        r_source, z_source, corresponding_J_phi, dA, r, z
    )
    return jnp.sqrt(B_r**2 + B_z**2)


@jax.jit
def compute_magnetic_field_magnitude_squared_at_a_point_from_source_points(
    r_source: Array,
    z_source: Array,
    corresponding_J_phi: Array,
    dA: float,
    r: float,
    z: float,
) -> Array:
    B_r, B_z = compute_magnetic_field_at_a_point_from_source_points(
        r_source, z_source, corresponding_J_phi, dA, r, z
    )
    return B_r**2 + B_z**2


def J_critical(
    angle: Array,
    magnetic_field_magnitude: Array,
    I_c0: float,
    thickness: float,
    width: float,
    B_0: float,
    alpha: float,
    beta: float,
    gamma_squared: float,
) -> Array:
    """Critical current density of a REBCO (2G HTS) tape under an arbitrary
    field magnitude and orientation, via the extended-Kim model
    (https://doi.org/10.1007/s10948-018-4678-8).

    Current-carrying capacity drops as the applied field grows, and drops
    faster when the field is more perpendicular to the tape's flat face than
    parallel to it — `epsilon_theta` interpolates between those two cases
    using `gamma_squared` (>1 means the perpendicular field is more
    damaging), and `(B/B_0)^alpha` with the `beta` power sets how sharply
    J_critical falls off with field strength.

    Args:
        angle: Angle between the field and the tape's tangent direction.
        magnetic_field_magnitude: |B| at the point.
        I_c0: Critical current at zero field (B_0 is a fit scale, not the
            field I_c0 was measured at).
        thickness, width: Tape cross-section, converts I_c0 into a density.
        B_0, alpha, beta, gamma_squared: Fitting parameters.

    Returns:
        Critical current density (engineering value, over the full tape
        cross-section).
    """
    J_c0 = I_c0 / thickness / width
    epsilon_theta = jnp.sqrt(gamma_squared * jnp.sin(angle) ** 2 + jnp.cos(angle) ** 2)
    return (
        J_c0 / (1 + epsilon_theta * (magnetic_field_magnitude / B_0) ** alpha) ** beta
    )


@jax.jit
def J_critical_of_SuperPower_SCS2050(
    angle: Array,
    magnetic_field_magnitude: Array,
) -> Array:
    """J_critical with fit parameters for SuperPower SCS2050 tape, from
    https://doi.org/10.1007/s10948-018-4678-8.
    """
    I_c0 = 58.81  # A
    thickness = 0.000095  # m
    width = 0.002  # m
    B_0 = 0.05827  # T
    alpha = 1.29
    beta = 0.5
    gamma_squared = 4.76

    return J_critical(
        angle, magnetic_field_magnitude, I_c0, thickness, width, B_0, alpha, beta,
        gamma_squared,
    )


@jax.jit
def compute_angle_between_field_and_tape_at_a_point_from_source_points(
    r_source: Array,
    z_source: Array,
    corresponding_J_phi: Array,
    dA: float,
    r: float,
    z: float,
    curve_parameters: Array,
    scale_with_respect_to_magnetic_field_magnitude: bool = False,
) -> ArrayLike:
    """Acute angle (in [0, pi/2]) between the local field B(r,z) and the
    tape's own direction at that height (the curve's tangent) — how
    perpendicular the field is to the tape, which is what drives
    field-dependent critical-current derating in J_critical.

    The coil is symmetric in both r (right/left half) and z (top/bottom
    half), so both are folded to their absolute value first: only the
    right-upper quadrant needs to be evaluated.

    Args:
        r_source, z_source, corresponding_J_phi, dA: The coil's source points.
        r, z: The field point.
        curve_parameters: The curve the tape follows (for its tangent).
        scale_with_respect_to_magnetic_field_magnitude: If True, weight the
            angle by |B(r,z)| / |B(0,0)| — a large angle far from the coil,
            where the field is negligible anyway, then contributes little.

    Returns:
        The (possibly scaled) angle.
    """
    r = jnp.abs(r)
    z = jnp.abs(z)

    B_r, B_z = compute_magnetic_field_at_a_point_from_source_points(
        r_source, z_source, corresponding_J_phi, dA, r, z
    )
    B = jnp.array([B_r, B_z])
    B_magnitude = jnp.linalg.norm(B)

    tangent_vector = basics.compute_curve_unit_tangent(z, curve_parameters)

    # cos(angle) = B . tangent / |B|. Guard the |B| -> 0 case (far from the
    # coil) two ways: substitute a safe denominator so reverse-mode autodiff
    # doesn't see a 0/0, and default cos=1 (angle=0) since the angle is
    # physically meaningless there anyway and this keeps it finite.
    eps = 1e-12
    B_magnitude_safe = jnp.where(B_magnitude > eps, B_magnitude, 1.0)
    cos_angle = jnp.where(
        B_magnitude > eps, jnp.dot(B, tangent_vector) / B_magnitude_safe, 1.0
    )

    # arccos has an infinite derivative at +/-1 (perfect alignment), so clip
    # slightly inside [-1, 1] to keep the gradient finite there.
    angle = jnp.arccos(jnp.clip(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7))
    acute_angle = jnp.where(angle > jnp.pi / 2, jnp.pi - angle, angle)

    Bc_r, Bc_z = compute_magnetic_field_at_a_point_from_source_points(
        r_source, z_source, corresponding_J_phi, dA, 0.0, 0.0
    )
    central_magnetic_field = jnp.linalg.norm(jnp.array([Bc_r, Bc_z]))

    return jnp.where(
        scale_with_respect_to_magnetic_field_magnitude,
        acute_angle * B_magnitude / central_magnetic_field,
        acute_angle,
    )


@jax.jit
def compute_J_critical_at_a_point_from_source_points(
    r_source: Array,
    z_source: Array,
    corresponding_J_phi: Array,
    dA: float,
    r: float,
    z: float,
    curve_parameters: Array,
) -> ArrayLike:
    """J_critical at (r, z): combines the local field magnitude and its
    angle to the tape into the extended-Kim derating model.
    """
    r = jnp.abs(r)
    z = jnp.abs(z)

    B_magnitude = compute_magnetic_field_magnitude_at_a_point_from_source_points(
        r_source, z_source, corresponding_J_phi, dA, r, z
    )
    angle = compute_angle_between_field_and_tape_at_a_point_from_source_points(
        r_source, z_source, corresponding_J_phi, dA, r, z, curve_parameters
    )
    return J_critical_of_SuperPower_SCS2050(angle, B_magnitude)


@jax.jit
def compute_lorentz_force_density_at_a_point_from_source_points(
    r_source: Array,
    z_source: Array,
    corresponding_J_phi: Array,
    dA: float,
    r: float,
    z: float,
) -> tuple[Array, Array]:
    """Lorentz force density f = J x B at (r, z): with J purely azimuthal
    (J = J_phi * phi_hat) and B = (B_r, 0, B_z), the cross product reduces
    to (f_r, f_z) = (J_phi * B_z, -J_phi * B_r).

    J_phi at an arbitrary field point isn't itself well-defined by the
    source points (they only carry current at discrete coil locations), so
    this looks up the current density of whichever source point is nearest
    to (r, z) — meant to be called AT the coil's own source points (see
    `compute_something_in_the_whole_coil`), where "nearest" is exact.
    """
    B_r, B_z = compute_magnetic_field_at_a_point_from_source_points(
        r_source, z_source, corresponding_J_phi, dA, r, z
    )

    squared_distance = (r_source - r) ** 2 + (z_source - z) ** 2
    nearest_index = jnp.argmin(squared_distance)
    J_phi_at_field = corresponding_J_phi[nearest_index]

    f_r = J_phi_at_field * B_z
    f_z = -J_phi_at_field * B_r
    return f_r, f_z


@jax.jit
def compute_lorentz_force_density_magnitude_at_a_point_from_source_points(
    r_source: Array,
    z_source: Array,
    corresponding_J_phi: Array,
    dA: float,
    r: float,
    z: float,
) -> ArrayLike:
    f_r, f_z = compute_lorentz_force_density_at_a_point_from_source_points(
        r_source, z_source, corresponding_J_phi, dA, r, z
    )
    return jnp.linalg.norm(jnp.array([f_r, f_z]))


# --- Variable-name dispatch -------------------------------------------------
#
# Every "compute ... at a point" function above shares the same first five
# arguments (r_source, z_source, J_phi, dA, r, z) but differs in what extra
# arguments it needs (curve_parameters for anything tape-angle-related,
# nothing extra for the raw field). The dispatch table below lets calling
# code select one of these functions by name instead of importing all of
# them individually, which is what the optimizer and plotter (built next)
# need: e.g. "give me a grid of magnetic_field_magnitude" without caring
# which specific function that maps to.

NormalScalarVariableName = Literal[
    "magnetic_field_magnitude", "magnetic_field_magnitude_squared"
]
CoilScalarVariableName = Literal[
    "J_critical",
    "angle_between_field_and_tape",
    "angle_between_field_and_tape_scaled",
    "lorentz_force_density_magnitude",
]
ScalarVariableName = NormalScalarVariableName | CoilScalarVariableName

NormalVectorVariableName = Literal["magnetic_field"]
CoilVectorVariableName = Literal["lorentz_force_density"]
VectorVariableName = NormalVectorVariableName | CoilVectorVariableName

VariableName = VectorVariableName | ScalarVariableName

# Each entry: (function to call, extra keyword arguments beyond
# r_source/z_source/J_source/dA/r/z). An extra argument whose value is a
# string is resolved from compute_something_at_a_point's own locals (i.e.
# "pass through whatever curve_parameters I was given"); a literal bool is
# passed as-is.
computers: dict[VariableName, tuple[Callable, dict[str, str | bool] | None]] = {
    "magnetic_field": (compute_magnetic_field_at_a_point_from_source_points, None),
    "magnetic_field_magnitude": (
        compute_magnetic_field_magnitude_at_a_point_from_source_points,
        None,
    ),
    "magnetic_field_magnitude_squared": (
        compute_magnetic_field_magnitude_squared_at_a_point_from_source_points,
        None,
    ),
    "J_critical": (
        compute_J_critical_at_a_point_from_source_points,
        {"curve_parameters": "curve_parameters"},
    ),
    "angle_between_field_and_tape": (
        compute_angle_between_field_and_tape_at_a_point_from_source_points,
        {
            "scale_with_respect_to_magnetic_field_magnitude": False,
            "curve_parameters": "curve_parameters",
        },
    ),
    "angle_between_field_and_tape_scaled": (
        compute_angle_between_field_and_tape_at_a_point_from_source_points,
        {
            "scale_with_respect_to_magnetic_field_magnitude": True,
            "curve_parameters": "curve_parameters",
        },
    ),
    "lorentz_force_density": (
        compute_lorentz_force_density_at_a_point_from_source_points,
        None,
    ),
    "lorentz_force_density_magnitude": (
        compute_lorentz_force_density_magnitude_at_a_point_from_source_points,
        None,
    ),
}


def compute_something_at_a_point(
    r_source: Array,
    z_source: Array,
    J_source: Array,
    dA: float,
    what: VariableName,
    r: float,
    z: float,
    curve_parameters: Array,
) -> Array | tuple[Array, Array]:
    """Dispatch to the right `compute_..._at_a_point` function by name."""
    something_computer, extra_arguments = computers[what]
    resolved_extra_arguments = (
        {
            key: locals()[value] if isinstance(value, str) else value
            for key, value in extra_arguments.items()
        }
        if extra_arguments is not None
        else {}
    )
    return something_computer(
        r_source, z_source, J_source, dA, r, z, **resolved_extra_arguments
    )


def compute_something_in_multiple_points(
    R: Array,
    Z: Array,
    r_source: Array,
    z_source: Array,
    J_source: Array,
    dA: float,
    curve_parameters: Array,
    what: VariableName,
) -> Array | tuple[Array, Array]:
    """Same dispatch, vectorized (jax.vmap) over every (r, z) in R, Z — R and
    Z can be any shape (e.g. a 2D grid or the coil's own flat source-point
    arrays); the result is reshaped back to match.
    """
    something_computer, extra_arguments = computers[what]
    resolved_extra_arguments = (
        {
            key: locals()[value] if isinstance(value, str) else value
            for key, value in extra_arguments.items()
        }
        if extra_arguments is not None
        else {}
    )

    something_flat = jax.vmap(
        lambda r, z: something_computer(
            r_source, z_source, J_source, dA, r, z, **resolved_extra_arguments
        )
    )(R.ravel(), Z.ravel())

    if isinstance(something_flat, tuple):
        return (
            something_flat[0].reshape(R.shape),
            something_flat[1].reshape(Z.shape),
        )
    return something_flat.reshape(R.shape)


@overload
def compute_something_in_the_whole_grid(
    R: Array, Z: Array, r_source: Array, z_source: Array, J_source: Array,
    dA: float, what: NormalScalarVariableName, curve_parameters: Array,
) -> Array: ...


@overload
def compute_something_in_the_whole_grid(
    R: Array, Z: Array, r_source: Array, z_source: Array, J_source: Array,
    dA: float, what: NormalVectorVariableName, curve_parameters: Array,
) -> tuple[Array, Array]: ...


def compute_something_in_the_whole_grid(
    R: Array,
    Z: Array,
    r_source: Array,
    z_source: Array,
    J_source: Array,
    dA: float,
    what: NormalScalarVariableName | NormalVectorVariableName,
    curve_parameters: Array,
) -> Array | tuple[Array, Array]:
    """Evaluate a field-only quantity (magnetic_field / its magnitude) over
    an arbitrary (R, Z) grid, e.g. for plotting.
    """
    return compute_something_in_multiple_points(
        R, Z, r_source, z_source, J_source, dA, curve_parameters, what
    )


@overload
def compute_something_in_the_whole_coil(
    r_source: Array, z_source: Array, J_source: Array, dA: float,
    what: CoilScalarVariableName, curve_parameters: Array,
) -> Array: ...


@overload
def compute_something_in_the_whole_coil(
    r_source: Array, z_source: Array, J_source: Array, dA: float,
    what: CoilVectorVariableName, curve_parameters: Array,
) -> tuple[Array, Array]: ...


def compute_something_in_the_whole_coil(
    r_source: Array,
    z_source: Array,
    J_source: Array,
    dA: float,
    what: CoilScalarVariableName | CoilVectorVariableName,
    curve_parameters: Array,
) -> Array | tuple[Array, Array]:
    """Evaluate a coil-dependent quantity (J_critical, angle-to-tape,
    Lorentz force) AT each of the coil's own source points — the only place
    those quantities are physically meaningful given how they're computed
    (e.g. Lorentz force needs an exact, not nearest-neighbor, J_phi lookup).
    """
    return compute_something_in_multiple_points(
        r_source, z_source, r_source, z_source, J_source, dA, curve_parameters, what
    )


@jax.jit
def compute_minimum_J_critical_of_the_whole_coil(
    r_source: Array,
    z_source: Array,
    J_source: Array,
    dA: float,
    curve_parameters: Array,
) -> ArrayLike:
    """The coil's weakest point: the minimum J_critical over all its own
    source points. An optimizer maximizing field strength must keep the
    actual operating current density below this, or the tape quenches.
    """
    return jnp.min(
        compute_something_in_the_whole_coil(
            r_source, z_source, J_source, dA, "J_critical", curve_parameters
        )
    )


# --- Batched evaluation and the self-consistent current-density solve -----
#
# Everything above vectorizes with a single jax.vmap: fine when the number
# of source points times target points is modest, but the fully-vectorized
# intermediates (one elliptic-integral evaluation per source-target pair)
# can outgrow memory for a fine coil discretization or a fine field grid.
# The functions below use `vmap_chunked` (batching.py) to bound peak memory
# by evaluating in chunks, at the cost of doing so sequentially instead of
# all at once.


def compute_magnetic_field_at_a_point_batched(
    r_source: Array,
    z_source: Array,
    J_phi: Array,
    dA: float,
    r: float,
    z: float,
    chunk_size: int = 256,
) -> Array:
    """Same result as `compute_magnetic_field_at_a_point_from_source_points`,
    but sums the per-source contributions in chunks of `chunk_size` instead
    of computing all of them at once.
    """

    def per_source(r_s: Array, z_s: Array, J_s: Array) -> Array:
        grid_spacing_squared = dA / 4
        delta_z = z - z_s

        k_squared = 4 * r_s * r / ((r_s + r) ** 2 + delta_z**2 + grid_spacing_squared)
        ellip_K = jaxellip.ellipk(k_squared)
        ellip_E = jaxellip.ellipe(k_squared)

        current = J_s * dA
        common_multiplier = (
            mu_0
            * current
            / (2 * pi * jnp.sqrt(delta_z**2 + (r_s + r) ** 2 + grid_spacing_squared))
        )
        common_denominator = delta_z**2 + (r - r_s) ** 2 + grid_spacing_squared

        B_z_contribution = common_multiplier * (
            (r_s**2 - delta_z**2 - r**2) * ellip_E / common_denominator + ellip_K
        )
        r_safe = jnp.where(r == 0, 1.0, r)
        B_r_contribution = jnp.where(
            r == 0,
            0.0,
            common_multiplier
            * delta_z
            / r_safe
            * ((r_s**2 + delta_z**2 + r**2) * ellip_E / common_denominator - ellip_K),
        )
        return jnp.array([B_r_contribution, B_z_contribution])

    per_source_batched = vmap_chunked(
        per_source, in_axes=(0, 0, 0), chunk_size=chunk_size
    )
    contributions = per_source_batched(r_source, z_source, J_phi)
    return jnp.sum(contributions, axis=0)


def compute_magnetic_field_on_coil_batched(
    r_source: Array,
    z_source: Array,
    J_phi: Array,
    dA: float,
    r_targets: Array,
    z_targets: Array,
    chunk_size_sources: int = 256,
    chunk_size_targets: int = 256,
) -> tuple[Array, Array]:
    """(B_r, B_z) at many target points, batching over both the source
    points (per target) and the targets themselves.
    """

    def per_target(r_t: Array, z_t: Array) -> Array:
        return compute_magnetic_field_at_a_point_batched(
            r_source, z_source, J_phi, dA, r_t, z_t, chunk_size=chunk_size_sources
        )

    per_target_batched = vmap_chunked(
        per_target, in_axes=(0, 0), chunk_size=chunk_size_targets
    )
    B_r_and_B_z = per_target_batched(r_targets, z_targets)
    return B_r_and_B_z[:, 0], B_r_and_B_z[:, 1]


def compute_minimum_J_critical_field_batched(
    r_targets: Array,
    z_targets: Array,
    B_r_reference: Array,
    B_z_reference: Array,
    current_density: float,
    curve_parameters: Array,
    chunk_size: int = 256,
) -> ArrayLike:
    """Minimum J_critical over many target points, given a precomputed
    *reference* field (see `compute_maximum_possible_current_density` for
    why the field can be precomputed once and scaled) rather than
    recomputing Biot-Savart from source points.

    Unlike `compute_angle_between_field_and_tape_at_a_point_from_source_points`,
    the raw (unfolded) angle is used here without folding it into
    [0, pi/2] — harmless, since J_critical only ever uses sin(angle)^2 and
    cos(angle)^2, both already symmetric under angle -> pi - angle.
    """

    def per_target(
        r_t: Array, z_t: Array, B_r_ref: Array, B_z_ref: Array
    ) -> Array:
        B_r = current_density * B_r_ref
        B_z = current_density * B_z_ref
        B_magnitude = jnp.sqrt(B_r**2 + B_z**2)

        tangent = basics.compute_curve_unit_tangent(z_t, curve_parameters)
        cos_angle = (B_r * tangent[0] + B_z * tangent[1]) / (B_magnitude + 1e-12)
        angle = jnp.arccos(jnp.clip(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7))

        return J_critical_of_SuperPower_SCS2050(angle, B_magnitude)

    per_target_batched = vmap_chunked(
        per_target, in_axes=(0, 0, 0, 0), chunk_size=chunk_size
    )
    J_critical_at_every_target = per_target_batched(
        r_targets, z_targets, B_r_reference, B_z_reference
    )
    return jnp.min(J_critical_at_every_target)


def compute_maximum_possible_current_density(
    curve_parameters: Array,
    total_arc_length: float,
    width: float,
    nr: int,
    nz: int,
    initial_current_density: float,
) -> ArrayLike:
    """Solve for the coil's self-consistent operating current density: the
    highest uniform current density the coil can carry such that, at every
    point on the coil, the field that current density itself produces
    doesn't push the local J_critical below that same current density.

    This is a fixed-point problem — the field depends on the current
    density, and the maximum allowable current density depends on the
    field — solved by finding the root of
    f(J) = J - min(J_critical over the coil, given field from J).

    The key simplification: Biot-Savart is linear in current density, and
    every source point here shares the same J. So the field at any
    candidate J is just J times the field from a *unit* current density
    (J=1) — computed once outside the root-finding loop — rather than
    re-running the full Biot-Savart sum on every Newton iteration.
    """
    # The current density returned here is discarded: only the geometry
    # (r_source, z_source, dA) is needed, since the field is evaluated at a
    # unit current density below and scaled from there.
    r_source, z_source, _initial_J_source, dA = (
        basics.compute_source_points_at_right_half_for_bent_coil(
            curve_parameters, total_arc_length, width, initial_current_density, nr, nz
        )
    )

    unit_current_density = jnp.ones_like(r_source)
    B_r_reference, B_z_reference = compute_magnetic_field_on_coil_batched(
        r_source,
        z_source,
        unit_current_density,
        dA,
        r_source,
        z_source,
        chunk_size_sources=512,
        chunk_size_targets=1024,
    )

    def f(current_density: Array) -> Array:
        J_critical_minimum = compute_minimum_J_critical_field_batched(
            r_source,
            z_source,
            B_r_reference,
            B_z_reference,
            current_density[0],
            curve_parameters,
            chunk_size=512,
        )
        return current_density[0] - J_critical_minimum

    return basics.solve_f_of_x_equals_0_with_finite_differences(
        f, initial_current_density
    )
