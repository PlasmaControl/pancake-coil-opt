import jax.numpy as jnp
import pytest
from scipy.constants import mu_0

from pancakecoilopt import computer
from pancakecoilopt.computer import schemas as computer_schemas
from pancakecoilopt.optimizer import optimizer, schemas


@pytest.fixture
def a_small_bent_coil():
    # Deliberately tiny (few pancakes, coarse discretization) so tests run
    # fast: this exercises the optimization machinery, not realistic coil
    # design.
    return computer_schemas.BentPancakeCoil(
        number_of_pancakes=2,
        separation_between_pancakes=0.01,
        current_source=50.0,
        tape=computer_schemas.HTSTape(thickness=0.001, width=0.004),
        number_of_turns=10,
        curve_parameters=[0.1, 0.0, 0.0],
    )


def _settings(objectives, **overrides):
    defaults = {
        "solver": "Nelder-Mead",
        "objectives": objectives,
        "constraints": schemas.OptimizationConstraints(
            minimum_central_inner_radius=0.05,
            maximum_central_inner_radius=0.2,
        ),
        "discretization": schemas.Discretization(
            number_of_radial_divisions_of_coil=4
        ),
    }
    defaults.update(overrides)
    return schemas.PancakeCoilOptimizationSettings(**defaults)


def test_objective_matches_a_direct_computation_of_the_central_field(
    a_small_bent_coil,
):
    settings = _settings(["maximize_central_magnetic_field_magnitude"])
    objective = optimizer.objective_maker(a_small_bent_coil, settings)

    parameters = jnp.array(a_small_bent_coil.curve_parameters)
    objective_value = objective(parameters)

    nr, nz = optimizer.compute_nr_and_nz(a_small_bent_coil, settings)
    current_density = computer.compute_maximum_possible_current_density(
        parameters,
        a_small_bent_coil.total_vertical_arc_length,
        a_small_bent_coil.width,
        nr,
        nz,
        a_small_bent_coil.current_density,
    )
    r_source, z_source, J_source, dA = (
        computer.compute_source_points_at_right_half_for_bent_coil(
            parameters,
            a_small_bent_coil.total_vertical_arc_length,
            a_small_bent_coil.width,
            current_density,
            nr,
            nz,
        )
    )
    central_field = computer.compute_something_at_a_point(
        r_source, z_source, J_source, dA, "magnetic_field_magnitude",
        0.0, 0.0, parameters,
    )

    # "maximize" objectives are negated, since scipy.optimize.minimize only
    # ever minimizes.
    assert float(objective_value) == pytest.approx(-float(central_field), rel=1e-5)


def test_objective_maker_combines_weighted_objectives(a_small_bent_coil):
    settings_single = _settings(["maximize_average_J_critical"])
    settings_weighted = _settings(
        ["maximize_average_J_critical"], objective_weights=[3.0]
    )

    parameters = jnp.array(a_small_bent_coil.curve_parameters)
    single_value = optimizer.objective_maker(a_small_bent_coil, settings_single)(
        parameters
    )
    weighted_value = optimizer.objective_maker(a_small_bent_coil, settings_weighted)(
        parameters
    )

    assert float(weighted_value) == pytest.approx(3.0 * float(single_value), rel=1e-5)


def test_optimize_reduces_the_objective_value(a_small_bent_coil, tmp_path):
    settings = _settings(["maximize_central_magnetic_field_magnitude"])
    result = optimizer.optimize(a_small_bent_coil, settings, tmp_path)

    # scipy.optimize.minimize always minimizes; a successful "maximize"
    # objective should end up no worse (i.e. no larger) than where it started.
    assert result.optimized_objective_value <= result.original_objective_value + 1e-6
    assert result.number_of_iterations >= 1


def test_optimize_writes_a_readable_yaml_report(a_small_bent_coil, tmp_path):
    import ruamel.yaml

    settings = _settings(["maximize_central_magnetic_field_magnitude"])
    optimizer.optimize(a_small_bent_coil, settings, tmp_path)

    yaml_files = list(tmp_path.rglob("*.yaml"))
    assert len(yaml_files) == 1

    loaded = ruamel.yaml.YAML().load(yaml_files[0].read_text())
    assert "datetime" in loaded
    assert "original_coil_result" in loaded
    assert "optimized_coil_result" in loaded
    assert loaded["optimized_coil_result"]["central_magnetic_field_magnitude"] > 0


def test_optimize_populates_coil_results_with_finite_values(
    a_small_bent_coil, tmp_path
):
    settings = _settings(["maximize_average_magnetic_field_magnitude"])
    result = optimizer.optimize(a_small_bent_coil, settings, tmp_path)

    for coil_result in (result.original_coil_result, result.optimized_coil_result):
        assert coil_result.central_magnetic_field_magnitude > 0
        assert coil_result.current > 0
        assert coil_result.average_angle_between_field_and_tape is not None
        assert coil_result.average_J_critical > 0
        assert coil_result.absolute_average_magnetic_field_magnitude_squared > 0


def _perturbed_parameters(coil):
    # A curve distinct from the coil's own initial parameters, so
    # "_on_grid"/"_on_axis" objective checks aren't trivially self-consistent
    # (e.g. a B2-on-grid ratio of exactly 1.0 at the initial point either way).
    return jnp.array(coil.curve_parameters) + jnp.array([0.01, 0.05, 0.0])


def _source_points_for(coil, settings, parameters):
    nr, nz = optimizer.compute_nr_and_nz(coil, settings)
    current_density = computer.compute_maximum_possible_current_density(
        parameters, coil.total_vertical_arc_length, coil.width, nr, nz,
        coil.current_density,
    )
    return computer.compute_source_points_at_right_half_for_bent_coil(
        parameters, coil.total_vertical_arc_length, coil.width, current_density,
        nr, nz,
    )


def test_average_B2_on_grid_objective_matches_manual_computation(a_small_bent_coil):
    settings = _settings(["maximize_average_magnetic_field_magnitude_squared_on_grid"])
    objective = optimizer.objective_maker(a_small_bent_coil, settings)
    parameters = _perturbed_parameters(a_small_bent_coil)

    # Reference <B^2>, from the *original* (unperturbed) coil.
    r0, z0, J0, dA0 = _source_points_for(
        a_small_bent_coil, settings, jnp.array(a_small_bent_coil.curve_parameters)
    )
    R, Z = computer.generate_a_grid(a_small_bent_coil, nr=32, numpy_or_jax="numpy")
    R, Z = jnp.asarray(R), jnp.asarray(Z)
    B2_ref_grid = computer.compute_something_in_the_whole_grid(
        R, Z, r0, z0, J0, dA0, "magnetic_field_magnitude_squared",
        jnp.array(a_small_bent_coil.curve_parameters),
    )
    B2_ref = jnp.sum(B2_ref_grid * jnp.abs(R)) / (jnp.sum(jnp.abs(R)) + 1e-30)

    # <B^2> at the perturbed parameters.
    r_source, z_source, J_source, dA = _source_points_for(
        a_small_bent_coil, settings, parameters
    )
    B2_grid = computer.compute_something_in_the_whole_grid(
        R, Z, r_source, z_source, J_source, dA,
        "magnetic_field_magnitude_squared", parameters,
    )
    B2 = jnp.sum(B2_grid * jnp.abs(R)) / (jnp.sum(jnp.abs(R)) + 1e-30)

    expected = -float(B2 / B2_ref)
    assert float(objective(parameters)) == pytest.approx(expected, rel=1e-4)


def test_average_on_axis_objective_matches_manual_computation(a_small_bent_coil):
    settings = _settings(["maximize_average_magnetic_field_magnitude_on_axis"])
    objective = optimizer.objective_maker(a_small_bent_coil, settings)
    parameters = _perturbed_parameters(a_small_bent_coil)

    r_source, z_source, J_source, dA = _source_points_for(
        a_small_bent_coil, settings, parameters
    )
    z_extent = a_small_bent_coil.maximum_height / 2
    Z_on_axis = jnp.linspace(-z_extent, z_extent, 64)
    R_axis = jnp.zeros_like(Z_on_axis)
    values = computer.compute_something_in_the_whole_grid(
        R_axis, Z_on_axis, r_source, z_source, J_source, dA,
        "magnetic_field_magnitude", parameters,
    )
    expected = -float(jnp.mean(values))

    assert float(objective(parameters)) == pytest.approx(expected, rel=1e-4)


def test_field_efficiency_objective_matches_manual_computation(a_small_bent_coil):
    settings = _settings(["maximize_field_efficiency_on_grid"])
    objective = optimizer.objective_maker(a_small_bent_coil, settings)
    parameters = _perturbed_parameters(a_small_bent_coil)

    r_source, z_source, J_source, dA = _source_points_for(
        a_small_bent_coil, settings, parameters
    )
    R, Z = computer.generate_a_grid(a_small_bent_coil, nr=32, numpy_or_jax="numpy")
    R, Z = jnp.asarray(R), jnp.asarray(Z)
    b2 = computer.compute_something_in_the_whole_grid(
        R, Z, r_source, z_source, J_source, dA,
        "magnetic_field_magnitude_squared", parameters,
    )
    average_b2 = jnp.sum(b2 * jnp.abs(R)) / (jnp.sum(jnp.abs(R)) + 1e-30)

    z_top = computer.find_z_bound_for_a_given_arc_length(
        a_small_bent_coil.total_vertical_arc_length, parameters
    )
    z_grid = jnp.linspace(-z_top, z_top, 128)
    outer_radius = (
        jnp.max(computer.compute_curve(z_grid, parameters)) + a_small_bent_coil.width
    )
    expected = -float(average_b2 / outer_radius)

    assert float(objective(parameters)) == pytest.approx(expected, rel=1e-4)


def test_field_inhomogeneity_objective_matches_manual_computation(a_small_bent_coil):
    settings = _settings(["minimize_field_inhomogeneity"])
    objective = optimizer.objective_maker(a_small_bent_coil, settings)
    parameters = _perturbed_parameters(a_small_bent_coil)

    r_source, z_source, J_source, dA = _source_points_for(
        a_small_bent_coil, settings, parameters
    )
    radius = 0.2 * float(a_small_bent_coil.curve_parameters[0])
    r_lin = jnp.linspace(0.0, radius, 9)
    z_lin = jnp.linspace(-radius, radius, 17)
    R, Z = jnp.meshgrid(r_lin, z_lin)
    inside_region = (R**2 + Z**2) <= radius**2
    W = R * inside_region

    b = computer.compute_something_in_the_whole_grid(
        R, Z, r_source, z_source, J_source, dA, "magnetic_field_magnitude", parameters
    )
    weight_sum = jnp.sum(W) + 1e-30
    mean_field = jnp.sum(W * b) / weight_sum
    variance = jnp.sum(W * (b - mean_field) ** 2) / weight_sum
    expected = float(jnp.sqrt(variance + 1e-30) / (mean_field + 1e-30))

    assert float(objective(parameters)) == pytest.approx(expected, rel=1e-4)


def test_field_inhomogeneity_respects_an_explicit_region_size(a_small_bent_coil):
    parameters = _perturbed_parameters(a_small_bent_coil)

    small_region = _settings(
        ["minimize_field_inhomogeneity"], homogeneity_region_half_size=0.005
    )
    large_region = _settings(
        ["minimize_field_inhomogeneity"], homogeneity_region_half_size=0.05
    )

    small_value = optimizer.objective_maker(a_small_bent_coil, small_region)(parameters)
    large_value = optimizer.objective_maker(a_small_bent_coil, large_region)(parameters)

    # A bigger region should generally show more field variation (harder to
    # keep uniform over a larger volume), not a strict theorem, but true
    # for this coil/perturbation, and a good sanity check that region size
    # actually changes the computation rather than being silently ignored.
    assert float(large_value) != pytest.approx(float(small_value), rel=1e-3)


def test_magnetic_energy_in_bore_objective_matches_manual_computation(
    a_small_bent_coil,
):
    settings = _settings(["maximize_magnetic_energy_in_bore"])
    objective = optimizer.objective_maker(a_small_bent_coil, settings)
    parameters = _perturbed_parameters(a_small_bent_coil)

    r_source, z_source, J_source, dA = _source_points_for(
        a_small_bent_coil, settings, parameters
    )
    z_top = computer.find_z_bound_for_a_given_arc_length(
        a_small_bent_coil.total_vertical_arc_length, parameters
    )
    bore_grid_nr, bore_grid_nz = 32, 64
    dz = 2 * z_top / bore_grid_nz
    z_centers = jnp.linspace(-z_top + dz / 2, z_top - dz / 2, bore_grid_nz)
    r_inner_at_z = computer.compute_curve(jnp.abs(z_centers), parameters)
    dr_per_row = r_inner_at_z / bore_grid_nr
    r_offsets = jnp.arange(bore_grid_nr) + 0.5
    R = dr_per_row[:, None] * r_offsets[None, :]
    Z = jnp.broadcast_to(z_centers[:, None], R.shape)
    b2 = computer.compute_something_in_the_whole_grid(
        R, Z, r_source, z_source, J_source, dA,
        "magnetic_field_magnitude_squared", parameters,
    )
    energy_density = b2 / (2 * mu_0)
    dV = 2 * jnp.pi * R * dr_per_row[:, None] * dz
    expected = -float(jnp.sum(energy_density * dV))

    assert float(objective(parameters)) == pytest.approx(expected, rel=1e-4)


def test_magnetic_energy_in_bore_stays_within_the_actual_bore(a_small_bent_coil):
    # At every z, the evaluation grid's outermost sample must stay strictly
    # inside r_inner(z) (never at or past the winding's own inner surface).
    # Otherwise, the "bore" integral would clip into the conductor, where the
    # Biot-Savart field from a coarse set of discrete source loops isn't a
    # physically meaningful approximation.
    parameters = _perturbed_parameters(a_small_bent_coil)
    z_top = computer.find_z_bound_for_a_given_arc_length(
        a_small_bent_coil.total_vertical_arc_length, parameters
    )
    bore_grid_nr, bore_grid_nz = 32, 64
    dz = 2 * z_top / bore_grid_nz
    z_centers = jnp.linspace(-z_top + dz / 2, z_top - dz / 2, bore_grid_nz)
    r_inner_at_z = computer.compute_curve(jnp.abs(z_centers), parameters)
    dr_per_row = r_inner_at_z / bore_grid_nr
    r_offsets = jnp.arange(bore_grid_nr) + 0.5
    R = dr_per_row[:, None] * r_offsets[None, :]

    assert bool(jnp.all(R < r_inner_at_z[:, None]))


def test_optimize_end_to_end_with_a_magnetic_energy_in_bore_objective(
    a_small_bent_coil, tmp_path
):
    settings = _settings(["maximize_magnetic_energy_in_bore"])
    result = optimizer.optimize(a_small_bent_coil, settings, tmp_path)
    assert result.optimized_objective_value <= result.original_objective_value + 1e-6


def test_field_inhomogeneity_in_bore_objective_matches_manual_computation(
    a_small_bent_coil,
):
    settings = _settings(["minimize_field_inhomogeneity_in_bore"])
    objective = optimizer.objective_maker(a_small_bent_coil, settings)
    parameters = _perturbed_parameters(a_small_bent_coil)

    r_source, z_source, J_source, dA = _source_points_for(
        a_small_bent_coil, settings, parameters
    )
    z_top = computer.find_z_bound_for_a_given_arc_length(
        a_small_bent_coil.total_vertical_arc_length, parameters
    )
    bore_grid_nr, bore_grid_nz = 32, 64
    dz = 2 * z_top / bore_grid_nz
    z_centers = jnp.linspace(-z_top + dz / 2, z_top - dz / 2, bore_grid_nz)
    r_inner_at_z = computer.compute_curve(jnp.abs(z_centers), parameters)
    dr_per_row = r_inner_at_z / bore_grid_nr
    r_offsets = jnp.arange(bore_grid_nr) + 0.5
    R = dr_per_row[:, None] * r_offsets[None, :]
    Z = jnp.broadcast_to(z_centers[:, None], R.shape)
    b = computer.compute_something_in_the_whole_grid(
        R, Z, r_source, z_source, J_source, dA, "magnetic_field_magnitude", parameters
    )
    weight_sum = jnp.sum(R) + 1e-30
    mean_field = jnp.sum(R * b) / weight_sum
    variance = jnp.sum(R * (b - mean_field) ** 2) / weight_sum
    expected = float(jnp.sqrt(variance + 1e-30) / (mean_field + 1e-30))

    assert float(objective(parameters)) == pytest.approx(expected, rel=1e-4)


def test_field_inhomogeneity_in_bore_differs_from_the_small_central_region_version(
    a_small_bent_coil,
):
    # Same underlying std/mean(|B|) formula as minimize_field_inhomogeneity,
    # but volume-weighted over the WHOLE bore instead of a small central
    # region (default 20% of the bore radius). These should generally
    # disagree, since a shape that's uniform right at the center can still
    # vary a lot further out in the bore, and vice versa. Not a strict
    # theorem, but true for this coil/perturbation, and a good sanity check
    # that the two objectives aren't secretly computing the same thing.
    parameters = _perturbed_parameters(a_small_bent_coil)
    small_region_settings = _settings(["minimize_field_inhomogeneity"])
    whole_bore_settings = _settings(["minimize_field_inhomogeneity_in_bore"])

    small_region_value = optimizer.objective_maker(a_small_bent_coil, small_region_settings)(
        parameters
    )
    whole_bore_value = optimizer.objective_maker(a_small_bent_coil, whole_bore_settings)(
        parameters
    )

    assert float(whole_bore_value) != pytest.approx(float(small_region_value), rel=1e-3)


def test_optimize_end_to_end_with_a_field_inhomogeneity_in_bore_objective(
    a_small_bent_coil, tmp_path
):
    settings = _settings(["minimize_field_inhomogeneity_in_bore"])
    result = optimizer.optimize(a_small_bent_coil, settings, tmp_path)
    assert result.optimized_objective_value <= result.original_objective_value + 1e-6


def test_optimize_end_to_end_with_a_field_efficiency_objective(
    a_small_bent_coil, tmp_path
):
    settings = _settings(["maximize_field_efficiency_on_grid"])
    result = optimizer.optimize(a_small_bent_coil, settings, tmp_path)
    assert result.optimized_objective_value <= result.original_objective_value + 1e-6


def test_optimize_end_to_end_with_a_field_inhomogeneity_objective(
    a_small_bent_coil, tmp_path
):
    settings = _settings(["minimize_field_inhomogeneity"])
    result = optimizer.optimize(a_small_bent_coil, settings, tmp_path)
    assert result.optimized_objective_value <= result.original_objective_value + 1e-6


def test_max_outer_radius_constraint_rejects_nelder_mead(a_small_bent_coil, tmp_path):
    settings = _settings(
        ["maximize_central_magnetic_field_magnitude"],
        constraints=schemas.OptimizationConstraints(
            minimum_central_inner_radius=0.05,
            maximum_central_inner_radius=0.2,
            maximum_outer_radius=0.15,
        ),
    )
    with pytest.raises(ValueError, match="Nelder-Mead"):
        optimizer.optimize(a_small_bent_coil, settings, tmp_path)


def test_max_outer_radius_constraint_is_enforced_by_a_constrained_solver(
    a_small_bent_coil, tmp_path
):
    # Cap the outer radius at exactly the coil's own starting size: whatever
    # the objective would otherwise want, the optimized coil's outer radius,
    # recomputed independently via BentPancakeCoil.maximum_outer_radius,
    # not the constraint function's own internal formula, must not exceed it.
    initial_outer_radius = a_small_bent_coil.maximum_outer_radius
    settings = _settings(
        ["maximize_central_magnetic_field_magnitude"],
        solver="COBYQA",
        constraints=schemas.OptimizationConstraints(
            minimum_central_inner_radius=0.05,
            maximum_central_inner_radius=0.2,
            maximum_outer_radius=initial_outer_radius,
        ),
    )
    result = optimizer.optimize(a_small_bent_coil, settings, tmp_path)
    assert (
        result.optimized_pancake_coil.maximum_outer_radius
        <= initial_outer_radius + 1e-3
    )
