"""Validate the Biot-Savart kernel against the textbook single-current-loop
formula, and sanity-check the J_critical model.
"""

import jax.numpy as jnp
import pytest
from scipy.constants import mu_0

from pancakecoilopt.computer import physics


def _analytic_on_axis_Bz(current: float, loop_radius: float, z: float) -> float:
    """B_z on the axis of a single circular loop of current (Jackson-style
    textbook result, independent of the elliptic-integral machinery being
    tested).
    """
    return mu_0 * current * loop_radius**2 / (2 * (loop_radius**2 + z**2) ** 1.5)


@pytest.fixture
def single_loop():
    """A single current ring represented with a negligible source area
    (dA), so the kernel's regularization term (dA/4) doesn't perturb the
    comparison against the exact point-loop formula.
    """
    current = 100.0
    loop_radius = 0.2
    dA = 1e-9
    r_source = jnp.array([loop_radius])
    z_source = jnp.array([0.0])
    J_phi = jnp.array([current / dA])
    return r_source, z_source, J_phi, dA, current, loop_radius


def test_on_axis_field_matches_the_textbook_single_loop_formula(single_loop):
    r_source, z_source, J_phi, dA, current, loop_radius = single_loop
    for z in (0.0, 0.1, -0.1, 0.5, 2.0):
        B_r, B_z = physics.compute_magnetic_field_at_a_point_from_source_points(
            r_source, z_source, J_phi, dA, r=0.0, z=z
        )
        assert float(B_r) == pytest.approx(0.0, abs=1e-9)
        assert float(B_z) == pytest.approx(
            _analytic_on_axis_Bz(current, loop_radius, z), rel=1e-4
        )


def test_field_decays_with_distance_from_the_loop(single_loop):
    r_source, z_source, J_phi, dA, _, _ = single_loop
    _, B_z_near = physics.compute_magnetic_field_at_a_point_from_source_points(
        r_source, z_source, J_phi, dA, r=0.0, z=0.0
    )
    _, B_z_far = physics.compute_magnetic_field_at_a_point_from_source_points(
        r_source, z_source, J_phi, dA, r=0.0, z=5.0
    )
    assert float(B_z_far) < float(B_z_near)


def test_field_scales_linearly_with_current(single_loop):
    r_source, z_source, J_phi, dA, _, _ = single_loop
    _, B_z = physics.compute_magnetic_field_at_a_point_from_source_points(
        r_source, z_source, J_phi, dA, r=0.0, z=0.3
    )
    _, B_z_doubled = physics.compute_magnetic_field_at_a_point_from_source_points(
        r_source, z_source, J_phi * 2, dA, r=0.0, z=0.3
    )
    assert float(B_z_doubled) == pytest.approx(2 * float(B_z), rel=1e-6)


def test_magnitude_helpers_are_consistent_with_the_raw_components(single_loop):
    r_source, z_source, J_phi, dA, _, _ = single_loop
    r, z = 0.05, 0.3
    B_r, B_z = physics.compute_magnetic_field_at_a_point_from_source_points(
        r_source, z_source, J_phi, dA, r, z
    )
    magnitude = physics.compute_magnetic_field_magnitude_at_a_point_from_source_points(
        r_source, z_source, J_phi, dA, r, z
    )
    magnitude_squared = (
        physics.compute_magnetic_field_magnitude_squared_at_a_point_from_source_points(
            r_source, z_source, J_phi, dA, r, z
        )
    )
    assert float(magnitude) == pytest.approx(
        float(jnp.sqrt(B_r**2 + B_z**2)), rel=1e-6
    )
    assert float(magnitude_squared) == pytest.approx(float(magnitude) ** 2, rel=1e-6)


def test_field_superposes_linearly_across_multiple_loops():
    # Field from two loops together must equal the sum of each loop's field
    # computed separately -- this is really testing that the vectorized sum
    # over source points does what it claims, not new physics.
    r_source = jnp.array([0.2, 0.3])
    z_source = jnp.array([0.1, -0.1])
    J_phi = jnp.array([1e8, 1e8])
    dA = 1e-6
    r, z = 0.05, 0.4

    B_r_combined, B_z_combined = (
        physics.compute_magnetic_field_at_a_point_from_source_points(
            r_source, z_source, J_phi, dA, r, z
        )
    )

    B_r_sum, B_z_sum = 0.0, 0.0
    for i in range(2):
        B_r_i, B_z_i = physics.compute_magnetic_field_at_a_point_from_source_points(
            r_source[i : i + 1], z_source[i : i + 1], J_phi[i : i + 1], dA, r, z
        )
        B_r_sum += B_r_i
        B_z_sum += B_z_i

    assert float(B_r_combined) == pytest.approx(float(B_r_sum), rel=1e-6)
    assert float(B_z_combined) == pytest.approx(float(B_z_sum), rel=1e-6)


def test_J_critical_equals_J_c0_at_zero_field():
    # At B=0, the (1 + ...)^beta denominator collapses to 1, leaving just
    # J_c0 = I_c0 / thickness / width regardless of angle.
    J_c0 = 58.81 / 0.000095 / 0.002
    for angle in (0.0, jnp.pi / 4, jnp.pi / 2):
        J_c = physics.J_critical_of_SuperPower_SCS2050(angle, 0.0)
        assert float(J_c) == pytest.approx(J_c0, rel=1e-6)


def test_J_critical_decreases_as_field_increases():
    angle = jnp.pi / 4
    J_c_low_field = physics.J_critical_of_SuperPower_SCS2050(angle, 0.1)
    J_c_high_field = physics.J_critical_of_SuperPower_SCS2050(angle, 5.0)
    assert float(J_c_high_field) < float(J_c_low_field)
