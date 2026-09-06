"""Build an objective function from an optimization-settings object, and run
`scipy.optimize.minimize` over a bent coil's curve parameters.

Only bent coils are handled so far (StraightPancakeCoil's per-pancake
inner-radius/turn-count parameterization, the maximum-outer-radius
constraint, and plotting integration are deferred to a later step).
"""

import datetime
import pathlib
import time
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import numpy
import scipy.optimize
from jax import Array
from jax.typing import ArrayLike

from .. import computer, reporter
from . import schemas


def compute_nr_and_nz(
    pancake_coil: computer.schemas.BentPancakeCoil,
    pancake_coil_optimization_settings: schemas.PancakeCoilOptimizationSettings,
) -> tuple[int, int]:
    """The radial discretization comes directly from settings; the vertical
    one defaults to whatever keeps source-point cells roughly square given
    the coil's own aspect ratio (height / width), if not given explicitly.
    """
    nr = (
        pancake_coil_optimization_settings.discretization
        .number_of_radial_divisions_of_coil
    )
    nz = (
        pancake_coil_optimization_settings.discretization
        .number_of_vertical_divisions_of_coil
    )
    if nz is None:
        nz = int(pancake_coil.total_vertical_arc_length / (pancake_coil.width / nr))
    return nr, nz


def _parse_objective_name(
    objective: str,
) -> tuple[computer.VariableName, Literal["average", "minimum", "maximum"] | tuple[float, float], Literal["maximize", "minimize"]]:
    """Split e.g. "maximize_average_J_critical" into
    (variable_name="J_critical", compute_what="average", do_what="maximize").

    "central" (as in "maximize_central_magnetic_field_magnitude") is special-
    cased to mean "evaluate at the single point (r=0, z=0)" rather than a
    reduction over many points.
    """
    parts = objective.split("_")
    do_what = parts[0]
    compute_what = parts[1]
    variable_name = "_".join(parts[2:])

    resolved_compute_what: Literal["average", "minimum", "maximum"] | tuple[float, float]
    resolved_compute_what = (0.0, 0.0) if compute_what == "central" else compute_what  # type: ignore[assignment]

    return variable_name, resolved_compute_what, do_what  # type: ignore[return-value]


def objective_maker(
    pancake_coil: computer.schemas.BentPancakeCoil,
    pancake_coil_optimization_settings: schemas.PancakeCoilOptimizationSettings,
) -> Callable[[Array], ArrayLike]:
    """Build a single scalar objective function of just the coil's curve
    parameters, combining every configured objective into one signed,
    weighted sum (scipy.optimize.minimize always minimizes, so "maximize"
    objectives are negated).
    """
    nr, nz = compute_nr_and_nz(pancake_coil, pancake_coil_optimization_settings)
    objectives = pancake_coil_optimization_settings.objectives

    # A "_on_grid" objective averages over a field-evaluation grid spanning
    # the whole coil, built once here (not per-objective, and not inside the
    # jitted objective itself) so every "_on_grid" objective shares the same
    # fixed evaluation points as the shape is optimized.
    RZ_on_grid: tuple[Array, Array] | None = None
    if any(objective.endswith("_on_grid") for objective in objectives):
        R_np, Z_np = computer.generate_a_grid(pancake_coil, nr=32, numpy_or_jax="numpy")
        RZ_on_grid = (jnp.asarray(R_np), jnp.asarray(Z_np))

    # A "_on_axis" objective averages along the on-axis (r=0) line, spanning
    # the initial coil's half-height -- also built once, so the averaging
    # domain stays fixed rather than shrinking/growing with the shape.
    Z_on_axis: Array | None = None
    if any(objective.endswith("_on_axis") for objective in objectives):
        z_extent = pancake_coil.maximum_height / 2
        Z_on_axis = jnp.linspace(-z_extent, z_extent, 64)

    # A small central region near (r=0, z=0), for "minimize_field_inhomogeneity".
    # Sampling a box and keeping only the points inside the selected region
    # (sphere or cylinder), weighted by the revolved (cylindrical) volume
    # element 2*pi*r*dr*dz -- the constant 2*pi*dr*dz cancels in a weighted
    # mean/std, so the weight is simply r (and 0 on the axis or outside the
    # region).
    RZ_central: tuple[Array, Array] | None = None
    W_central: Array | None = None
    if "minimize_field_inhomogeneity" in objectives:
        radius = pancake_coil_optimization_settings.homogeneity_region_half_size
        if radius is None:
            radius = 0.2 * float(pancake_coil.curve_parameters[0])
        r_lin = jnp.linspace(0.0, radius, 9)
        z_lin = jnp.linspace(-radius, radius, 17)
        R_central, Z_central = jnp.meshgrid(r_lin, z_lin)
        if pancake_coil_optimization_settings.homogeneity_region_shape == "sphere":
            inside_region = (R_central**2 + Z_central**2) <= radius**2
        else:
            inside_region = jnp.ones_like(R_central, dtype=bool)
        RZ_central = (R_central, Z_central)
        W_central = R_central * inside_region

    def compute_reference_average_B2_on_grid() -> float:
        """<B^2> on the grid for the coil as originally given, before any
        optimization -- used to normalize the B^2-on-grid objective so its
        value starts at 1 and is dimensionless, comparable across coils of
        very different absolute field strength.
        """
        assert RZ_on_grid is not None
        initial_parameters = jnp.array(pancake_coil.curve_parameters)
        if pancake_coil_optimization_settings.constraints.keep_current_constant:
            current_density = pancake_coil.current_density
        else:
            current_density = computer.compute_maximum_possible_current_density(
                initial_parameters,
                pancake_coil.total_vertical_arc_length,
                pancake_coil.width,
                nr,
                nz,
                pancake_coil.current_density,
            )
        r0, z0, J0, dA0 = computer.compute_source_points_at_right_half_for_bent_coil(
            initial_parameters,
            pancake_coil.total_vertical_arc_length,
            pancake_coil.width,
            current_density,
            nr,
            nz,
        )
        R_ref, Z_ref = RZ_on_grid
        B2_ref_grid = computer.compute_something_in_the_whole_grid(
            R_ref, Z_ref, r0, z0, J0, dA0,
            "magnetic_field_magnitude_squared", initial_parameters,
        )
        W_ref = jnp.abs(R_ref)
        B2_ref = jnp.sum(B2_ref_grid * W_ref) / (jnp.sum(W_ref) + 1e-30)
        return float(jnp.maximum(B2_ref, 1e-12))

    def make_compute_function(
        variable_name: computer.VariableName,
        compute_what: Literal["average", "minimum", "maximum"] | tuple[float, float],
    ) -> Callable[[Array, Array, Array, Array, float], ArrayLike]:
        """A function of (curve_parameters, r_source, z_source, J_source, dA)
        that evaluates `variable_name` and reduces it per `compute_what`.
        """
        if isinstance(compute_what, tuple):
            r_point, z_point = compute_what

            def compute(curve_parameters, r_source, z_source, J_source, dA):
                return computer.compute_something_at_a_point(
                    r_source, z_source, J_source, dA, variable_name,
                    r_point, z_point, curve_parameters,
                )

            return compute

        if compute_what in ("minimum", "maximum"):
            reduction = {"minimum": jnp.min, "maximum": jnp.max}[compute_what]

            def compute(curve_parameters, r_source, z_source, J_source, dA):
                return reduction(
                    computer.compute_something_in_the_whole_coil(
                        r_source, z_source, J_source, dA, variable_name, curve_parameters
                    )
                )

            return compute

        # compute_what == "average" from here on.
        if variable_name.endswith("_on_grid"):
            base_variable_name = variable_name.removesuffix("_on_grid")
            is_B2 = base_variable_name == "magnetic_field_magnitude_squared"
            B2_reference = compute_reference_average_B2_on_grid() if is_B2 else None

            def compute(curve_parameters, r_source, z_source, J_source, dA):
                assert RZ_on_grid is not None
                R, Z = RZ_on_grid
                values = computer.compute_something_in_the_whole_grid(
                    R, Z, r_source, z_source, J_source, dA,
                    base_variable_name, curve_parameters,
                )
                W = jnp.abs(R)
                average_value = jnp.sum(values * W) / (jnp.sum(W) + 1e-30)
                return average_value / B2_reference if is_B2 else average_value

            return compute

        if variable_name.endswith("_on_axis"):
            base_variable_name = variable_name.removesuffix("_on_axis")

            def compute(curve_parameters, r_source, z_source, J_source, dA):
                assert Z_on_axis is not None
                R_axis = jnp.zeros_like(Z_on_axis)
                values = computer.compute_something_in_the_whole_grid(
                    R_axis, Z_on_axis, r_source, z_source, J_source, dA,
                    base_variable_name, curve_parameters,
                )
                return jnp.mean(values)

            return compute

        def compute(curve_parameters, r_source, z_source, J_source, dA):
            return jnp.mean(
                computer.compute_something_in_the_whole_coil(
                    r_source, z_source, J_source, dA, variable_name, curve_parameters
                )
            )

        return compute

    # "Field efficiency" = grid-averaged B^2 divided by the coil's outer
    # radius: normalizing by size keeps the optimizer from trivially raising
    # <B^2> by just growing/spreading the winding.
    def compute_field_efficiency(curve_parameters, r_source, z_source, J_source, dA):
        assert RZ_on_grid is not None
        R, Z = RZ_on_grid
        b2 = computer.compute_something_in_the_whole_grid(
            R, Z, r_source, z_source, J_source, dA,
            "magnetic_field_magnitude_squared", curve_parameters,
        )
        W = jnp.abs(R)
        average_b2 = jnp.sum(b2 * W) / (jnp.sum(W) + 1e-30)

        z_top = computer.find_z_bound_for_a_given_arc_length(
            pancake_coil.total_vertical_arc_length, curve_parameters
        )
        z_grid = jnp.linspace(-z_top, z_top, 128)
        outer_radius = (
            jnp.max(computer.compute_curve(z_grid, curve_parameters))
            + pancake_coil.width
        )
        return average_b2 / (outer_radius + 1e-30)

    # Field homogeneity: relative spatial variation (std/mean, a coefficient
    # of variation) of |B| over the central region. Minimizing it flattens
    # the field there, as wanted for NMR/MRI-style magnets; being scale-
    # invariant, it can't be gamed by just lowering |B| everywhere.
    def compute_field_inhomogeneity(curve_parameters, r_source, z_source, J_source, dA):
        assert RZ_central is not None
        assert W_central is not None
        R, Z = RZ_central
        b = computer.compute_something_in_the_whole_grid(
            R, Z, r_source, z_source, J_source, dA,
            "magnetic_field_magnitude", curve_parameters,
        )
        weight_sum = jnp.sum(W_central) + 1e-30
        mean_field = jnp.sum(W_central * b) / weight_sum
        variance = jnp.sum(W_central * (b - mean_field) ** 2) / weight_sum
        # +eps keeps the gradient finite at variance=0.
        rms_deviation = jnp.sqrt(variance + 1e-30)
        return rms_deviation / (mean_field + 1e-30)

    weights = pancake_coil_optimization_settings.objective_weights or [1.0] * len(
        objectives
    )
    signed_compute_functions = []
    for objective, weight in zip(objectives, weights, strict=True):
        if objective == "maximize_field_efficiency_on_grid":
            signed_compute_functions.append((compute_field_efficiency, -weight))
            continue
        if objective == "minimize_field_inhomogeneity":
            signed_compute_functions.append((compute_field_inhomogeneity, weight))
            continue
        variable_name, compute_what, do_what = _parse_objective_name(objective)
        compute_fn = make_compute_function(variable_name, compute_what)
        multiplier = -weight if do_what == "maximize" else weight
        signed_compute_functions.append((compute_fn, multiplier))

    @jax.jit
    def objective_to_optimize(parameters: Array) -> ArrayLike:
        if pancake_coil_optimization_settings.constraints.keep_current_constant:
            current_density = pancake_coil.current_density
        else:
            current_density = computer.compute_maximum_possible_current_density(
                parameters,
                pancake_coil.total_vertical_arc_length,
                pancake_coil.width,
                nr,
                nz,
                pancake_coil.current_density,
            )
        r_source, z_source, J_source, dA = (
            computer.compute_source_points_at_right_half_for_bent_coil(
                parameters,
                pancake_coil.total_vertical_arc_length,
                pancake_coil.width,
                current_density,
                nr,
                nz,
            )
        )

        objective_value = sum(
            multiplier * compute_fn(parameters, r_source, z_source, J_source, dA)
            for compute_fn, multiplier in signed_compute_functions
        )

        # A solver probing a pathological shape (e.g. one that momentarily
        # collapses the winding radius) can hit a NaN objective; return a
        # large finite penalty instead so it's steered away rather than
        # crashing the optimization.
        return jnp.where(jnp.isfinite(objective_value), objective_value, 1e10)

    return objective_to_optimize


def optimize(
    pancake_coil: computer.schemas.BentPancakeCoil,
    pancake_coil_optimization_settings: schemas.PancakeCoilOptimizationSettings,
    output_folder: pathlib.Path,
) -> schemas.OptimizationResult:
    """Optimize a bent coil's curve parameters against the configured
    objective(s), write a YAML report, and return the full result.
    """
    nr, nz = compute_nr_and_nz(pancake_coil, pancake_coil_optimization_settings)
    objective = objective_maker(pancake_coil, pancake_coil_optimization_settings)

    initial_parameters = pancake_coil.curve_parameters
    constraints = pancake_coil_optimization_settings.constraints
    parameter_bounds = [
        (constraints.minimum_central_inner_radius, constraints.maximum_central_inner_radius)
    ] + [(-2.0, 2.0)] * (len(initial_parameters) - 1)

    solver = pancake_coil_optimization_settings.solver
    minimize_kwargs: dict = {
        "method": solver,
        "bounds": parameter_bounds,
        "options": {"disp": True, "maxiter": 1000},
    }

    # The objective (including the self-consistent current solve) is fully
    # differentiable, so trust-constr can be handed the exact autodiff
    # gradient instead of falling back to its own finite-difference one.
    if solver == "trust-constr":
        gradient = jax.jit(jax.grad(objective))

        def jac(x):
            # A pathological shape the solver probes can produce a non-finite
            # gradient component; zero it out (a "flat" direction) rather
            # than letting trust-constr choke on a NaN/inf Jacobian.
            g = numpy.asarray(gradient(jnp.asarray(x)), dtype=numpy.float64)
            return numpy.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

        minimize_kwargs["jac"] = jac

    start_time = time.time()
    scipy_result = scipy.optimize.minimize(objective, initial_parameters, **minimize_kwargs)
    time_took_to_optimize = time.time() - start_time

    optimized_parameters = scipy_result.x
    original_parameters = jnp.array(initial_parameters)

    def resolve_current_density(parameters: Array) -> ArrayLike:
        if constraints.keep_current_constant:
            return pancake_coil.current_density
        return computer.compute_maximum_possible_current_density(
            parameters,
            pancake_coil.total_vertical_arc_length,
            pancake_coil.width,
            nr,
            nz,
            pancake_coil.current_density,
        )

    original_current_density = resolve_current_density(original_parameters)
    optimized_current_density = resolve_current_density(
        jnp.array(optimized_parameters)
    )

    def summarize(
        curve_parameters: Array, current_density: ArrayLike, B2_reference: float | None
    ) -> tuple[schemas.CoilResult, float]:
        r_source, z_source, J_source, dA = (
            computer.compute_source_points_at_right_half_for_bent_coil(
                curve_parameters,
                pancake_coil.total_vertical_arc_length,
                pancake_coil.width,
                current_density,
                nr,
                nz,
            )
        )
        central_field = computer.compute_something_at_a_point(
            r_source, z_source, J_source, dA, "magnetic_field_magnitude",
            0.0, 0.0, curve_parameters,
        )
        angle_avg = jnp.mean(
            computer.compute_something_in_the_whole_coil(
                r_source, z_source, J_source, dA,
                "angle_between_field_and_tape", curve_parameters,
            )
        )
        J_critical_avg = jnp.mean(
            computer.compute_something_in_the_whole_coil(
                r_source, z_source, J_source, dA, "J_critical", curve_parameters
            )
        )
        lorentz_force_avg = jnp.mean(
            computer.compute_something_in_the_whole_coil(
                r_source, z_source, J_source, dA,
                "lorentz_force_density_magnitude", curve_parameters,
            )
        )

        R, Z = computer.generate_a_grid(pancake_coil, nr=32, numpy_or_jax="numpy")
        R, Z = jnp.asarray(R), jnp.asarray(Z)
        B2_grid = computer.compute_something_in_the_whole_grid(
            R, Z, r_source, z_source, J_source, dA,
            "magnetic_field_magnitude_squared", curve_parameters,
        )
        B2_avg = float(
            jnp.sum(B2_grid * jnp.abs(R)) / (jnp.sum(jnp.abs(R)) + 1e-30)
        )
        B2_reference = B2_avg if B2_reference is None else B2_reference

        coil_result = schemas.CoilResult(
            current=current_density * pancake_coil.tape.width * pancake_coil.tape.thickness,
            central_magnetic_field_magnitude=float(central_field),
            average_angle_between_field_and_tape=float(angle_avg),
            average_J_critical=float(J_critical_avg),
            average_lorentz_force_density_magnitude=float(lorentz_force_avg),
            absolute_average_magnetic_field_magnitude_squared=B2_avg,
            normalized_average_magnetic_field_magnitude_squared=(
                B2_avg / (B2_reference + 1e-30)
            ),
        )
        return coil_result, B2_reference

    original_result, B2_reference = summarize(
        original_parameters, original_current_density, None
    )
    optimized_result, _ = summarize(
        jnp.array(optimized_parameters), optimized_current_density, B2_reference
    )

    now = datetime.datetime.now(datetime.UTC)
    output_name = f"pancakecoilopt_{now.isoformat(timespec='seconds')}"
    results_folder = output_folder / output_name
    results_folder.mkdir(parents=True)

    result = schemas.OptimizationResult(
        time_took_to_optimize=time_took_to_optimize,
        number_of_iterations=scipy_result.nit,
        original_coil_result=original_result,
        optimized_coil_result=optimized_result,
        original_objective_value=float(objective(original_parameters)),
        optimized_objective_value=float(scipy_result.fun),
        original_pancake_coil=pancake_coil.model_copy(
            update={
                "curve_parameters": list(original_parameters),
                "effective_current_density": float(original_current_density),
            }
        ),
        optimized_pancake_coil=pancake_coil.model_copy(
            update={
                "curve_parameters": list(optimized_parameters),
                "effective_current_density": float(optimized_current_density),
            }
        ),
        input_optimization_settings=pancake_coil_optimization_settings,
    )
    reporter.export_model_as_yaml(result, results_folder / f"{output_name}.yaml")
    return result
