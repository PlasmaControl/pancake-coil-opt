import pydantic
import pytest

from pancakecoilopt.computer import schemas


@pytest.fixture
def tape():
    return schemas.HTSTape(thickness=0.001, width=0.004)


def test_current_density_and_arc_length(tape):
    coil = schemas.BentPancakeCoil(
        number_of_pancakes=3,
        separation_between_pancakes=0.002,
        current_source=100.0,
        tape=tape,
        number_of_turns=10,
        curve_parameters=[0.1, 0.0, 0.0],
    )
    assert coil.current_density == pytest.approx(100.0 / (0.001 * 0.004))
    # 3 pancakes of width 0.004 plus 2 gaps of 0.002 each.
    assert coil.total_vertical_arc_length == pytest.approx(3 * 0.004 + 2 * 0.002)


def test_bent_coil_geometry_for_a_constant_radius_curve(tape):
    # curve_parameters=[R, 0, 0] is a plain cylinder: no curvature, so
    # maximum_height should exactly equal the tape's total arc length
    # (matches the Tier-0 basics.py finding), and maximum_outer_radius
    # should just be R + the radial tape build.
    coil = schemas.BentPancakeCoil(
        number_of_pancakes=1,
        separation_between_pancakes=0.002,
        current_source=100.0,
        tape=tape,
        number_of_turns=10,
        curve_parameters=[0.1, 0.0, 0.0],
    )
    assert coil.maximum_height == pytest.approx(
        coil.total_vertical_arc_length, abs=1e-4
    )
    assert coil.width == pytest.approx(0.001 * 10)
    assert coil.maximum_width == coil.width
    assert coil.maximum_outer_radius == pytest.approx(0.1 + coil.width, abs=1e-4)


def test_straight_coil_requires_matching_list_lengths(tape):
    with pytest.raises(pydantic.ValidationError):
        schemas.StraightPancakeCoil(
            number_of_pancakes=2,
            separation_between_pancakes=0.002,
            current_source=100.0,
            tape=tape,
            inner_radii=[0.1],  # only one entry for 2 pancakes
            number_of_turns=[10, 20],
        )


def test_straight_coil_geometry(tape):
    coil = schemas.StraightPancakeCoil(
        number_of_pancakes=2,
        separation_between_pancakes=0.002,
        current_source=100.0,
        tape=tape,
        inner_radii=[0.1, 0.15],
        number_of_turns=[10, 5],
    )
    assert coil.widths == pytest.approx([0.01, 0.005])
    assert coil.maximum_width == pytest.approx(0.01)
    # second pancake's outer edge (0.15+0.005=0.155) exceeds the first's
    # (0.1+0.01=0.11)
    assert coil.maximum_outer_radius == pytest.approx(0.155)
    assert coil.maximum_height == pytest.approx(coil.total_vertical_arc_length)


def test_pancake_coil_union_discriminates_on_type(tape):
    bent = schemas.pancake_coil_adapter.validate_python(
        {
            "type": "bent",
            "number_of_pancakes": 1,
            "separation_between_pancakes": 0.002,
            "current_source": 100.0,
            "tape": tape.model_dump(),
            "number_of_turns": 10,
            "curve_parameters": [0.1, 0.0, 0.0],
        }
    )
    assert isinstance(bent, schemas.BentPancakeCoil)

    straight = schemas.pancake_coil_adapter.validate_python(
        {
            "type": "straight",
            "number_of_pancakes": 1,
            "separation_between_pancakes": 0.002,
            "current_source": 100.0,
            "tape": tape.model_dump(),
            "inner_radii": [0.1],
            "number_of_turns": [10],
        }
    )
    assert isinstance(straight, schemas.StraightPancakeCoil)
