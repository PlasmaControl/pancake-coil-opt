import jax.numpy as jnp
import pytest

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
