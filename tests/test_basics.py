"""Sanity-check the curve/geometry math against cases with a known analytic
answer: a constant-radius curve (curve_parameters = [R, 0, 0]) is just a
straight cylindrical solenoid, so arc length, area, and source-point layout
all reduce to elementary formulas.
"""

import jax.numpy as jnp
import pytest

from pancakecoilopt.computer import basics


@pytest.fixture
def constant_radius_curve():
    return jnp.array([2.0, 0.0, 0.0])


def test_compute_curve_is_constant_for_a_straight_coil(constant_radius_curve):
    z = jnp.linspace(-5, 5, 11)
    r = basics.compute_curve(z, constant_radius_curve)
    assert jnp.allclose(r, 2.0)


def test_compute_curve_derivative_is_zero_for_a_straight_coil(
    constant_radius_curve,
):
    z = jnp.linspace(-5, 5, 11)
    assert jnp.allclose(basics.compute_curve_derivative(z, constant_radius_curve), 0.0)


def test_arc_length_of_a_straight_coil_equals_z_top(constant_radius_curve):
    # dr/dz = 0, so sqrt(1 + 0^2) = 1 and the arc length is just z_top itself.
    for z_top in (3.0, -3.0, 0.5):
        length = basics.compute_arc_length(z_top, constant_radius_curve)
        assert length == pytest.approx(z_top, abs=1e-4)


def test_find_z_bound_inverts_arc_length(constant_radius_curve):
    arc_length = 8.0
    z_top = basics.find_z_bound_for_a_given_arc_length(
        arc_length, constant_radius_curve
    )
    # For a straight coil, the height for a given tape length is exact: z = L/2.
    assert float(z_top) == pytest.approx(4.0, abs=1e-4)


def test_area_below_a_straight_curve_is_a_rectangle(constant_radius_curve):
    area = basics.compute_area_below_curve(-3.0, 3.0, constant_radius_curve)
    assert float(area) == pytest.approx(2.0 * 6.0, abs=1e-6)  # R * height


def test_newton_solver_finds_a_known_root():
    # x arrives wrapped as a length-1 array (so it can be threaded through
    # jax.lax.fori_loop); f must index into it to hand jax.grad a true
    # scalar output, not a shape-(1,) array.
    root = basics.solve_f_of_x_equals_0_with_grad(lambda x: x[0] ** 2 - 4.0, x0=3.0)
    assert float(root) == pytest.approx(2.0, abs=1e-6)


def test_finite_difference_newton_solver_finds_a_known_root():
    root = basics.solve_f_of_x_equals_0_with_finite_differences(
        lambda x: x[0] ** 2 - 4.0, x0=3.0
    )
    assert float(root) == pytest.approx(2.0, abs=1e-3)


def test_finite_difference_solver_stays_finite_at_current_density_scale():
    # This is the scenario the relative step size was chosen for: x0 around
    # 1e7 (typical current densities), where an absolute step size would
    # underflow float32 precision and produce a NaN Newton step.
    root = basics.solve_f_of_x_equals_0_with_finite_differences(
        lambda x: x[0] - 5_000_000.0, x0=1e7
    )
    assert jnp.isfinite(root)
    assert float(root) == pytest.approx(5_000_000.0, rel=1e-3)


def test_source_points_for_a_straight_coil_span_the_expected_annulus(
    constant_radius_curve,
):
    total_arc_length = 8.0  # -> z_top = 4.0
    width = 1.0
    nr, nz = 5, 20
    r_points, z_points, J_source, dA = (
        basics.compute_source_points_at_right_half_for_bent_coil(
            constant_radius_curve,
            total_arc_length=total_arc_length,
            width=width,
            current_density=1e6,
            nr=nr,
            nz=nz,
        )
    )

    assert r_points.shape == (nr * nz,)
    assert z_points.shape == (nr * nz,)
    assert jnp.all(r_points >= 2.0 - 1e-9)
    assert jnp.all(r_points < 2.0 + width)
    assert jnp.all(jnp.abs(z_points) <= 4.0 + 1e-3)
    assert jnp.allclose(J_source, 1e6)

    # Every point represents an equal slice of the total rectangular
    # cross-section (width * total_arc_length).
    total_area_from_points = dA * nr * nz
    assert float(total_area_from_points) == pytest.approx(
        width * total_arc_length, abs=1e-4
    )


@pytest.fixture
def a_bent_coil():
    from pancakecoilopt.computer import schemas

    # Chosen so r-extent and z-extent are comparable (avoids the default
    # nz-from-aspect-ratio formula degenerating to a 1-point axis).
    return schemas.BentPancakeCoil(
        number_of_pancakes=5,
        separation_between_pancakes=0.05,
        current_source=100.0,
        tape=schemas.HTSTape(thickness=0.001, width=0.004),
        number_of_turns=10,
        curve_parameters=[0.05, 0.0, 0.0],
    )


def test_generate_a_grid_covers_the_coil_by_default(a_bent_coil):
    R, Z = basics.generate_a_grid(a_bent_coil, nr=8, numpy_or_jax="numpy")
    assert R.shape == Z.shape
    assert float(R.min()) == pytest.approx(-a_bent_coil.maximum_outer_radius, abs=1e-6)
    assert float(R.max()) == pytest.approx(a_bent_coil.maximum_outer_radius, abs=1e-6)
    assert float(Z.min()) == pytest.approx(-a_bent_coil.maximum_height / 2, abs=1e-4)
    assert float(Z.max()) == pytest.approx(a_bent_coil.maximum_height / 2, abs=1e-4)


def test_generate_a_grid_ignores_a_narrower_explicit_range(a_bent_coil):
    # A range narrower than the coil's own extent is NOT honored. The
    # grid always covers at least the whole coil, silently widening a
    # too-small request back out to the coil's default extent.
    narrower_than_the_coil = (-0.001, 0.001)
    R, _ = basics.generate_a_grid(
        a_bent_coil, nr=8, r_range=narrower_than_the_coil, numpy_or_jax="numpy"
    )
    assert float(R.min()) == pytest.approx(-a_bent_coil.maximum_outer_radius, abs=1e-6)
    assert float(R.max()) == pytest.approx(a_bent_coil.maximum_outer_radius, abs=1e-6)


def test_generate_a_grid_honors_a_wider_explicit_range(a_bent_coil):
    wider_than_the_coil = (-100.0, 100.0)
    R, _ = basics.generate_a_grid(
        a_bent_coil, nr=8, r_range=wider_than_the_coil, numpy_or_jax="numpy"
    )
    assert float(R.min()) == pytest.approx(-100.0, abs=1e-6)
    assert float(R.max()) == pytest.approx(100.0, abs=1e-6)


def test_generate_a_grid_range_endpoints_are_independent(a_bent_coil):
    # One side wider than the coil (honored), the other narrower (clipped
    # back to the coil's own default extent).
    R, _ = basics.generate_a_grid(
        a_bent_coil, nr=8, r_range=(-100.0, 0.001), numpy_or_jax="numpy"
    )
    assert float(R.min()) == pytest.approx(-100.0, abs=1e-6)
    assert float(R.max()) == pytest.approx(a_bent_coil.maximum_outer_radius, abs=1e-6)
