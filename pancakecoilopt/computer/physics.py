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

import jax
import jax.numpy as jnp
import jaxellip
from jax import Array
from scipy.constants import mu_0, pi


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
