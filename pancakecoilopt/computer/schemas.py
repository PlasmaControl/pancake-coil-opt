"""Data models describing a pancake coil.

A coil is a stack of "pancakes": flat spiral windings of HTS (high-
temperature superconducting) tape, stacked along z with a fixed separation
between them. Two coil shapes are modeled:

  - BentPancakeCoil: a single continuously-curved winding whose radius
    follows the polynomial curve r=f(z) from basics.py. The stack of
    pancakes is treated as one solid radial "width" of tape.
  - StraightPancakeCoil: a stack of ordinary flat pancakes, each with its own
    inner radius and turn count — no curve, radius is constant within each
    pancake.

These are pydantic models so that inputs get validated (e.g. positive
thickness, at least one pancake) and so `computed_field` properties can
derive geometry (area, height, current density) from the handful of physical
parameters an optimizer actually varies.
"""

from typing import Annotated, Literal

import jax.numpy as jnp
import pydantic


class HTSTape(pydantic.BaseModel):
    """Cross-section of the superconducting tape used to wind the coil."""

    thickness: float = pydantic.Field(gt=0)
    width: float = pydantic.Field(gt=0)


class PancakeCoilBase(pydantic.BaseModel):
    """Fields shared by both coil shapes."""

    number_of_pancakes: int = pydantic.Field(ge=1)
    separation_between_pancakes: float = pydantic.Field(gt=0)
    current_source: float = pydantic.Field(gt=0)
    tape: HTSTape

    # Set by the optimizer after solving for the self-consistent current
    # density (the density at which the coil's peak field-dependent critical
    # current is reached) — not a design input, just a place to record the
    # result once known.
    effective_current_density: float | None = None

    @pydantic.computed_field
    @property
    def current_density(self) -> float:
        """Current per unit cross-sectional area of tape, from the nominal
        (not field-derated) current source.
        """
        return self.current_source / (self.tape.thickness * self.tape.width)

    @pydantic.computed_field
    @property
    def total_vertical_arc_length(self) -> float:
        """Total tape length needed to stack every pancake with the required
        separation between them — i.e. the coil's total height measured
        along the winding, not straight-line height.
        """
        return (
            self.number_of_pancakes * self.tape.width
            + (self.number_of_pancakes - 1) * self.separation_between_pancakes
        )


class BentPancakeCoil(PancakeCoilBase):
    type: Literal["bent"] = "bent"
    number_of_turns: float = pydantic.Field(ge=10)
    curve_parameters: list[float] = pydantic.Field(
        description=(
            "Coefficients (A, B, C, ...) of the curve r=f(z)=A+B*z^2+C*z^4+..."
            " that the coil's winding radius follows."
        ),
    )

    @pydantic.computed_field
    @property
    def width(self) -> float:
        """Total radial build of the stacked tape turns."""
        return self.tape.thickness * self.number_of_turns

    @pydantic.computed_field
    @property
    def maximum_width(self) -> float:
        """Alias for `width`, mirroring StraightPancakeCoil's `maximum_width`
        (there, widths vary per pancake, so this is a max over them) so
        callers can use one name regardless of coil type.
        """
        return self.width

    @pydantic.computed_field
    @property
    def maximum_height(self) -> float:
        from .basics import find_z_bound_for_a_given_arc_length

        z_top = find_z_bound_for_a_given_arc_length(
            self.total_vertical_arc_length, jnp.array(self.curve_parameters)
        )
        return float(z_top) * 2

    @pydantic.computed_field
    @property
    def maximum_outer_radius(self) -> float:
        """The curve's own maximum could occur at the midplane (z=0) or at
        the top/bottom edge depending on its shape, so both are checked;
        adding `width` accounts for the tape's radial build outward from the
        curve.
        """
        from .basics import compute_curve

        curve_parameters = jnp.array(self.curve_parameters)
        z_top = self.maximum_height / 2
        r_at_top = compute_curve(z_top, curve_parameters)
        r_at_center = compute_curve(0.0, curve_parameters)
        return float(max(r_at_top, r_at_center)) + self.width


class StraightPancakeCoil(PancakeCoilBase):
    type: Literal["straight"] = "straight"
    inner_radii: list[float] = pydantic.Field(
        description="Inner winding radius of each pancake."
    )
    number_of_turns: list[float] = pydantic.Field(
        description="Number of tape turns in each pancake."
    )

    @pydantic.model_validator(mode="after")
    def check_inner_radii_and_number_of_turns_match_number_of_pancakes(self):
        if (
            len(self.inner_radii) != self.number_of_pancakes
            or len(self.number_of_turns) != self.number_of_pancakes
        ):
            message = (
                "inner_radii and number_of_turns must each have exactly"
                f" number_of_pancakes ({self.number_of_pancakes}) entries; got"
                f" inner_radii={self.inner_radii},"
                f" number_of_turns={self.number_of_turns}"
            )
            raise ValueError(message)
        return self

    @pydantic.computed_field
    @property
    def widths(self) -> list[float]:
        return [turns * self.tape.thickness for turns in self.number_of_turns]

    @pydantic.computed_field
    @property
    def maximum_width(self) -> float:
        return max(self.widths)

    @pydantic.computed_field
    @property
    def maximum_height(self) -> float:
        return self.total_vertical_arc_length

    @pydantic.computed_field
    @property
    def maximum_outer_radius(self) -> float:
        return max(
            inner_radius + width
            for inner_radius, width in zip(
                self.inner_radii, self.widths, strict=True
            )
        )


PancakeCoil = Annotated[
    BentPancakeCoil | StraightPancakeCoil,
    pydantic.Field(discriminator="type"),
]
pancake_coil_adapter = pydantic.TypeAdapter(PancakeCoil)
