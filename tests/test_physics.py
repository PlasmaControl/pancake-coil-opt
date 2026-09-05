"""Validate the Biot-Savart kernel against the textbook single-current-loop
formula, and sanity-check the J_critical model.
"""

import jax.numpy as jnp
import pytest
from scipy.constants import mu_0

from pancakecoilopt.computer import basics, physics


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


@pytest.fixture
def curved_coil_sources():
    """A small bent-coil source-point cloud (reusing Tier-0's basics.py) with
    genuine curvature, for testing the angle/J_critical/Lorentz functions
    that depend on curve_parameters.
    """
    curve_parameters = jnp.array([0.1, 0.3, 0.0])
    total_arc_length = 0.4
    width = 0.01
    current_density = 1e7
    r_source, z_source, J_source, dA = (
        basics.compute_source_points_at_right_half_for_bent_coil(
            curve_parameters, total_arc_length, width, current_density, nr=4, nz=10
        )
    )
    return r_source, z_source, J_source, dA, curve_parameters


def test_angle_is_zero_on_axis_for_a_straight_coil():
    # A constant-radius curve has a purely-vertical tangent everywhere. On
    # axis (r=0), B is purely vertical too (B_r=0 by symmetry), so the field
    # is perfectly aligned with the tape: angle should be ~0 (only the
    # arccos clipping keeps it from being exactly 0).
    curve_parameters = jnp.array([0.2, 0.0, 0.0])
    r_source = jnp.array([0.2, 0.2])
    z_source = jnp.array([-0.05, 0.05])
    J_phi = jnp.array([1e7, 1e7])
    dA = 1e-7

    angle = physics.compute_angle_between_field_and_tape_at_a_point_from_source_points(
        r_source, z_source, J_phi, dA, r=0.0, z=0.02, curve_parameters=curve_parameters
    )
    assert float(angle) == pytest.approx(0.0, abs=1e-3)


def test_angle_matches_a_manual_recomputation_from_its_own_building_blocks(
    curved_coil_sources,
):
    r_source, z_source, J_source, dA, curve_parameters = curved_coil_sources
    r, z = 0.05, 0.15

    angle = physics.compute_angle_between_field_and_tape_at_a_point_from_source_points(
        r_source, z_source, J_source, dA, r, z, curve_parameters
    )

    B_r, B_z = physics.compute_magnetic_field_at_a_point_from_source_points(
        r_source, z_source, J_source, dA, r, z
    )
    B = jnp.array([B_r, B_z])
    tangent = basics.compute_curve_unit_tangent(z, curve_parameters)
    cos_angle = jnp.dot(B, tangent) / jnp.linalg.norm(B)
    raw_angle = jnp.arccos(jnp.clip(cos_angle, -1 + 1e-7, 1 - 1e-7))
    expected = jnp.where(raw_angle > jnp.pi / 2, jnp.pi - raw_angle, raw_angle)

    assert float(angle) == pytest.approx(float(expected), abs=1e-5)
    assert 0.0 <= float(angle) <= jnp.pi / 2 + 1e-6


def test_scaled_angle_equals_angle_times_field_ratio(curved_coil_sources):
    r_source, z_source, J_source, dA, curve_parameters = curved_coil_sources
    r, z = 0.05, 0.15

    unscaled = physics.compute_angle_between_field_and_tape_at_a_point_from_source_points(
        r_source, z_source, J_source, dA, r, z, curve_parameters,
        scale_with_respect_to_magnetic_field_magnitude=False,
    )
    scaled = physics.compute_angle_between_field_and_tape_at_a_point_from_source_points(
        r_source, z_source, J_source, dA, r, z, curve_parameters,
        scale_with_respect_to_magnetic_field_magnitude=True,
    )
    B_magnitude = physics.compute_magnetic_field_magnitude_at_a_point_from_source_points(
        r_source, z_source, J_source, dA, r, z
    )
    Bc_magnitude = physics.compute_magnetic_field_magnitude_at_a_point_from_source_points(
        r_source, z_source, J_source, dA, 0.0, 0.0
    )
    assert float(scaled) == pytest.approx(
        float(unscaled) * float(B_magnitude) / float(Bc_magnitude), rel=1e-5
    )


def test_J_critical_at_a_point_matches_its_constituent_functions(curved_coil_sources):
    r_source, z_source, J_source, dA, curve_parameters = curved_coil_sources
    r, z = 0.05, 0.15

    J_c = physics.compute_J_critical_at_a_point_from_source_points(
        r_source, z_source, J_source, dA, r, z, curve_parameters
    )
    B_magnitude = physics.compute_magnetic_field_magnitude_at_a_point_from_source_points(
        r_source, z_source, J_source, dA, r, z
    )
    angle = physics.compute_angle_between_field_and_tape_at_a_point_from_source_points(
        r_source, z_source, J_source, dA, r, z, curve_parameters
    )
    expected = physics.J_critical_of_SuperPower_SCS2050(angle, B_magnitude)
    assert float(J_c) == pytest.approx(float(expected), rel=1e-6)


def test_lorentz_force_uses_the_current_density_of_the_nearest_source_point(
    curved_coil_sources,
):
    r_source, z_source, J_source, dA, _ = curved_coil_sources
    # Evaluate exactly at the 3rd source point: nearest neighbor is itself,
    # with zero distance, so the lookup is unambiguous.
    index = 3
    r, z = float(r_source[index]), float(z_source[index])

    f_r, f_z = physics.compute_lorentz_force_density_at_a_point_from_source_points(
        r_source, z_source, J_source, dA, r, z
    )
    B_r, B_z = physics.compute_magnetic_field_at_a_point_from_source_points(
        r_source, z_source, J_source, dA, r, z
    )
    J_phi_at_field = J_source[index]

    assert float(f_r) == pytest.approx(float(J_phi_at_field * B_z), rel=1e-6)
    assert float(f_z) == pytest.approx(float(-J_phi_at_field * B_r), rel=1e-6)


def test_lorentz_force_magnitude_matches_its_components(curved_coil_sources):
    r_source, z_source, J_source, dA, _ = curved_coil_sources
    r, z = float(r_source[0]), float(z_source[0])

    magnitude = (
        physics.compute_lorentz_force_density_magnitude_at_a_point_from_source_points(
            r_source, z_source, J_source, dA, r, z
        )
    )
    f_r, f_z = physics.compute_lorentz_force_density_at_a_point_from_source_points(
        r_source, z_source, J_source, dA, r, z
    )
    assert float(magnitude) == pytest.approx(
        float(jnp.sqrt(f_r**2 + f_z**2)), rel=1e-6
    )


def test_dispatch_by_name_matches_calling_the_function_directly(curved_coil_sources):
    r_source, z_source, J_source, dA, curve_parameters = curved_coil_sources
    r, z = 0.05, 0.15

    for name, expected in [
        (
            "magnetic_field_magnitude",
            physics.compute_magnetic_field_magnitude_at_a_point_from_source_points(
                r_source, z_source, J_source, dA, r, z
            ),
        ),
        (
            "J_critical",
            physics.compute_J_critical_at_a_point_from_source_points(
                r_source, z_source, J_source, dA, r, z, curve_parameters
            ),
        ),
        (
            "lorentz_force_density_magnitude",
            physics.compute_lorentz_force_density_magnitude_at_a_point_from_source_points(
                r_source, z_source, J_source, dA, r, z
            ),
        ),
    ]:
        dispatched = physics.compute_something_at_a_point(
            r_source, z_source, J_source, dA, name, r, z, curve_parameters
        )
        assert float(dispatched) == pytest.approx(float(expected), rel=1e-6)


def test_compute_something_in_the_whole_grid_matches_pointwise_evaluation(
    curved_coil_sources,
):
    r_source, z_source, J_source, dA, curve_parameters = curved_coil_sources
    r_vals = jnp.linspace(0.01, 0.2, 4)
    z_vals = jnp.linspace(-0.1, 0.1, 3)
    R, Z = jnp.meshgrid(r_vals, z_vals, indexing="xy")

    B2_grid = physics.compute_something_in_the_whole_grid(
        R, Z, r_source, z_source, J_source, dA,
        "magnetic_field_magnitude_squared", curve_parameters,
    )
    assert B2_grid.shape == R.shape

    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            expected = (
                physics.compute_magnetic_field_magnitude_squared_at_a_point_from_source_points(
                    r_source, z_source, J_source, dA, R[i, j], Z[i, j]
                )
            )
            assert float(B2_grid[i, j]) == pytest.approx(float(expected), rel=1e-5)

    B_r_grid, B_z_grid = physics.compute_something_in_the_whole_grid(
        R, Z, r_source, z_source, J_source, dA, "magnetic_field", curve_parameters
    )
    assert B_r_grid.shape == R.shape
    assert B_z_grid.shape == Z.shape


def test_compute_something_in_the_whole_coil_and_its_minimum(curved_coil_sources):
    r_source, z_source, J_source, dA, curve_parameters = curved_coil_sources

    J_c_at_every_source_point = physics.compute_something_in_the_whole_coil(
        r_source, z_source, J_source, dA, "J_critical", curve_parameters
    )
    assert J_c_at_every_source_point.shape == r_source.shape

    minimum = physics.compute_minimum_J_critical_of_the_whole_coil(
        r_source, z_source, J_source, dA, curve_parameters
    )
    assert float(minimum) == pytest.approx(
        float(jnp.min(J_c_at_every_source_point)), rel=1e-6
    )
