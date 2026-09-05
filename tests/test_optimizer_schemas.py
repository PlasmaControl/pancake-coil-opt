import pydantic
import pytest

from pancakecoilopt.computer import schemas as computer_schemas
from pancakecoilopt.optimizer import schemas


@pytest.fixture
def settings_kwargs():
    return {
        "solver": "Nelder-Mead",
        "objectives": ["maximize_central_magnetic_field_magnitude"],
        "constraints": schemas.OptimizationConstraints(
            minimum_central_inner_radius=0.05, maximum_central_inner_radius=0.15
        ),
        "discretization": schemas.Discretization(
            number_of_radial_divisions_of_coil=8
        ),
    }


def test_settings_accept_matching_objective_weights(settings_kwargs):
    settings = schemas.PancakeCoilOptimizationSettings(
        **settings_kwargs, objective_weights=[2.0]
    )
    assert settings.objective_weights == [2.0]


def test_settings_reject_mismatched_objective_weights_length(settings_kwargs):
    with pytest.raises(pydantic.ValidationError):
        schemas.PancakeCoilOptimizationSettings(
            **settings_kwargs, objective_weights=[1.0, 2.0]
        )


def test_settings_reject_an_unknown_objective_name(settings_kwargs):
    settings_kwargs["objectives"] = ["not_a_real_objective"]
    with pytest.raises(pydantic.ValidationError):
        schemas.PancakeCoilOptimizationSettings(**settings_kwargs)


@pytest.fixture
def a_bent_coil():
    return computer_schemas.BentPancakeCoil(
        number_of_pancakes=1,
        separation_between_pancakes=0.002,
        current_source=100.0,
        tape=computer_schemas.HTSTape(thickness=0.001, width=0.004),
        number_of_turns=10,
        curve_parameters=[0.1, 0.0, 0.0],
    )


def test_optimization_result_computes_datetime_and_commit_hash(a_bent_coil):
    coil_result = schemas.CoilResult(central_magnetic_field_magnitude=1.0, current=5.0)
    result = schemas.OptimizationResult(
        time_took_to_optimize=1.23,
        number_of_iterations=10,
        original_coil_result=coil_result,
        optimized_coil_result=coil_result,
        original_objective_value=-1.0,
        optimized_objective_value=-2.0,
        original_pancake_coil=a_bent_coil,
        optimized_pancake_coil=a_bent_coil,
        input_optimization_settings=schemas.PancakeCoilOptimizationSettings(
            solver="Nelder-Mead",
            objectives=["maximize_central_magnetic_field_magnitude"],
            constraints=schemas.OptimizationConstraints(
                minimum_central_inner_radius=0.05,
                maximum_central_inner_radius=0.15,
            ),
            discretization=schemas.Discretization(
                number_of_radial_divisions_of_coil=8
            ),
        ),
    )
    assert result.pancakecoilopt_last_commit_hash_used
    assert result.datetime

    dumped = result.model_dump()
    ordered_keys = list(dumped.keys())
    assert ordered_keys[0] == "datetime"
    assert ordered_keys[1] == "time_took_to_optimize"
    assert set(ordered_keys) == set(type(result).model_fields) | set(
        type(result).model_computed_fields
    )
