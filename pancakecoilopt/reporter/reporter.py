"""Export any pydantic model to a human-readable YAML file.

No domain dependencies on purpose — this is used to save optimization
results, but doesn't need to know anything about coils or physics.
"""

import io
import json
import pathlib

import pydantic
import ruamel.yaml


def export_model_as_yaml(
    model: pydantic.BaseModel, output_path: pathlib.Path
) -> pathlib.Path:
    """Write `model` to `output_path` as YAML.

    Routed through JSON first (`model_dump_json`) rather than dumping
    `model.model_dump()` directly: that way pydantic's own encoders handle
    every field type (datetimes, nested models, etc.) consistently, and
    `ruamel.yaml` only ever sees plain dicts/lists/scalars it already knows
    how to format.
    """
    model_as_dictionary = json.loads(model.model_dump_json(exclude_none=False))

    yaml_object = ruamel.yaml.YAML()
    yaml_object.width = 88
    yaml_object.indent(mapping=2, sequence=4, offset=2)

    # ruamel's default representation of Python's None is the YAML "~" alias;
    # spell it out as the more readable "null" instead.
    def represent_none(representer: ruamel.yaml.representer.RoundTripRepresenter, _):
        return representer.represent_scalar("tag:yaml.org,2002:null", "null")

    yaml_object.representer.add_representer(type(None), represent_none)

    with io.StringIO() as string_stream:
        yaml_object.dump(model_as_dictionary, string_stream)
        yaml_string = string_stream.getvalue()

    output_path.write_text(yaml_string)
    return output_path
