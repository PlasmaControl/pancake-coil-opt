"""Settings and result models for a coil optimization run.

`Objective` names encode both what to compute and how to reduce it, parsed
by convention in optimizer.py: "maximize_central_magnetic_field_magnitude"
means do_what="maximize", compute_what="central" (i.e. at r=0,z=0),
variable_name="magnetic_field_magnitude" (one of physics.py's dispatch
names). This lets new objectives be added by combining an existing variable
name with average/minimum/maximum/central, without writing a new function
for each combination.
"""

import datetime
import subprocess
from datetime import UTC
from typing import Any, Literal

import pydantic

from ..computer import schemas as computer_schemas

Objective = Literal[
    "minimize_average_angle_between_field_and_tape",
    "minimize_average_angle_between_field_and_tape_scaled_with_central_magnetic_field_magnitude",
    "maximize_central_magnetic_field_magnitude",
    "maximize_average_J_critical",
    "maximize_average_magnetic_field_magnitude",
    "maximize_minimum_J_critical",
    "minimize_average_lorentz_force_density_magnitude",
    "minimize_maximum_lorentz_force_density_magnitude",
    "maximize_average_magnetic_field_magnitude_squared_on_grid",
    "maximize_average_magnetic_field_magnitude_on_axis",
    "maximize_field_efficiency_on_grid",
    "maximize_magnetic_energy_in_bore",
    "maximize_average_magnetic_field_magnitude_in_bore",
    "minimize_field_inhomogeneity",
    "minimize_field_inhomogeneity_in_bore",
]


class Discretization(pydantic.BaseModel):
    number_of_radial_divisions_of_coil: int = pydantic.Field(ge=1)
    number_of_vertical_divisions_of_coil: int | None = pydantic.Field(
        default=None, ge=4
    )


class OptimizationConstraints(pydantic.BaseModel):
    minimum_central_inner_radius: float = pydantic.Field(gt=0)
    maximum_central_inner_radius: float = pydantic.Field(gt=0)
    keep_current_constant: bool = pydantic.Field(default=False)
    maximum_outer_radius: float | None = pydantic.Field(
        default=None,
        gt=0,
        description=(
            "Optional cap on the coil's maximum outer radius (max over z of"
            " r(z) + width). Keeps the winding within a buildable envelope."
            " Requires a constraint-capable solver ('COBYQA' or 'trust-constr')."
        ),
    )
    minimum_inner_radius: float | None = pydantic.Field(
        default=None,
        gt=0,
        description=(
            "Optional floor on the coil's inner radius over the WHOLE curve"
            " (min over z of r(z)), not just at the center like"
            " minimum_central_inner_radius. Without this, only r(0) is"
            " bounded. The higher-order curve coefficients (bounded only to"
            " [-2, 2] each) can still drive r(z) toward or through 0 away from"
            " z=0, letting the two mirrored halves of the winding touch or"
            " cross. Requires a constraint-capable solver ('COBYQA' or"
            " 'trust-constr')."
        ),
    )


class PancakeCoilOptimizationSettings(pydantic.BaseModel):
    solver: Literal["COBYQA", "trust-constr", "Nelder-Mead"]
    objectives: list[Objective] = pydantic.Field(min_length=1)
    objective_weights: list[float] | None = pydantic.Field(
        default=None,
        description=(
            "Optional per-objective weights, same length and order as"
            " `objectives`. The (maximize/minimize-signed) objectives are summed"
            " weighted. Use this to balance objectives with very different scales."
            " Defaults to all 1.0."
        ),
    )
    homogeneity_region_half_size: float | None = pydantic.Field(
        default=None,
        gt=0,
        description=(
            "Radius/half-size (in metres) of the central region over which"
            " `minimize_field_inhomogeneity` measures field uniformity."
            " Defaults to 0.2 x bore when unset. A larger region demands"
            " uniformity over a bigger volume and is harder to achieve."
        ),
    )
    homogeneity_region_shape: Literal["sphere", "cylinder"] = pydantic.Field(
        default="sphere",
        description=(
            "Shape of the central region used by `minimize_field_inhomogeneity`."
            " 'sphere' measures a spherical region of radius"
            " `homogeneity_region_half_size`; 'cylinder' measures a cylindrical"
            " region with that radius and half-height. Both are volume-weighted"
            " by the revolved element r (cylindrical coordinates)."
        ),
    )
    constraints: OptimizationConstraints
    discretization: Discretization

    @pydantic.model_validator(mode="after")
    def _check_objective_weights_length(self):
        if self.objective_weights is not None and len(
            self.objective_weights
        ) != len(self.objectives):
            message = (
                "objective_weights must have the same length as objectives"
                f" ({len(self.objective_weights)} != {len(self.objectives)})."
            )
            raise ValueError(message)
        return self


class CoilResult(pydantic.BaseModel):
    central_magnetic_field_magnitude: float
    current: float
    average_angle_between_field_and_tape: float | None = None
    average_J_critical: float | None = None
    average_lorentz_force_density_magnitude: float | None = None
    absolute_average_magnetic_field_magnitude_squared: float | None = pydantic.Field(
        default=None, description="Grid-averaged <B^2> [T^2]."
    )
    normalized_average_magnetic_field_magnitude_squared: float | None = pydantic.Field(
        default=None, description="Grid-averaged <B^2>, relative to the original coil."
    )


class OptimizationResult(pydantic.BaseModel):
    time_took_to_optimize: float
    number_of_iterations: int

    original_coil_result: CoilResult
    optimized_coil_result: CoilResult

    original_objective_value: float
    optimized_objective_value: float

    original_pancake_coil: computer_schemas.PancakeCoil
    optimized_pancake_coil: computer_schemas.PancakeCoil

    input_optimization_settings: PancakeCoilOptimizationSettings

    @pydantic.computed_field
    @property
    def pancakecoilopt_last_commit_hash_used(self) -> str:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )

    @pydantic.computed_field
    @property
    def datetime(self) -> str:
        return datetime.datetime.now(UTC).isoformat()

    @pydantic.model_serializer(when_used="always")
    def _ordered_for_readability(self) -> dict[str, Any]:
        """Put the most-interesting-to-a-human fields first in the exported
        YAML (datetime and headline results), rather than pydantic's default
        declaration order.
        """
        priority_order = [
            "datetime",
            "time_took_to_optimize",
            "number_of_iterations",
            "original_objective_value",
            "optimized_objective_value",
            "original_coil_result",
            "optimized_coil_result",
            "original_pancake_coil",
            "optimized_pancake_coil",
            "input_optimization_settings",
            "pancakecoilopt_last_commit_hash_used",
        ]
        all_fields = list(type(self).model_computed_fields) + list(
            type(self).model_fields
        )
        remaining_fields = [f for f in all_fields if f not in priority_order]
        full_order = priority_order + remaining_fields

        if len(full_order) != len(all_fields):
            message = (
                "priority_order contains a key not present on this model."
                " Keep it in sync with OptimizationResult's fields."
            )
            raise ValueError(message)

        return {key: getattr(self, key) for key in full_order}
