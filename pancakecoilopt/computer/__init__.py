from . import basics, physics, schemas
from .basics import (
    compute_curve,
    compute_source_points_at_right_half_for_bent_coil,
    find_z_bound_for_a_given_arc_length,
    solve_f_of_x_equals_0_with_finite_differences,
    solve_f_of_x_equals_0_with_grad,
)
from .physics import (
    J_critical_of_SuperPower_SCS2050,
    VariableName,
    compute_maximum_possible_current_density,
    compute_minimum_J_critical_of_the_whole_coil,
    compute_something_at_a_point,
    compute_something_in_the_whole_coil,
    compute_something_in_the_whole_grid,
)

__all__ = [
    "J_critical_of_SuperPower_SCS2050",
    "VariableName",
    "basics",
    "compute_curve",
    "compute_maximum_possible_current_density",
    "compute_minimum_J_critical_of_the_whole_coil",
    "compute_something_at_a_point",
    "compute_something_in_the_whole_coil",
    "compute_something_in_the_whole_grid",
    "compute_source_points_at_right_half_for_bent_coil",
    "find_z_bound_for_a_given_arc_length",
    "physics",
    "schemas",
    "solve_f_of_x_equals_0_with_finite_differences",
    "solve_f_of_x_equals_0_with_grad",
]
