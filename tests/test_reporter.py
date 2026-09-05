import pydantic

from pancakecoilopt.reporter import export_model_as_yaml


class _Nested(pydantic.BaseModel):
    value: float


class _Sample(pydantic.BaseModel):
    name: str
    count: int
    maybe_missing: float | None = None
    nested: _Nested


def test_export_model_as_yaml_round_trips_through_yaml(tmp_path):
    import ruamel.yaml

    model = _Sample(name="coil-1", count=3, nested=_Nested(value=1.5))
    output_path = export_model_as_yaml(model, tmp_path / "result.yaml")

    assert output_path.exists()
    loaded = ruamel.yaml.YAML().load(output_path.read_text())
    assert loaded["name"] == "coil-1"
    assert loaded["count"] == 3
    assert loaded["nested"]["value"] == 1.5


def test_export_model_as_yaml_writes_null_for_missing_optional_fields(tmp_path):
    model = _Sample(name="coil-1", count=3, nested=_Nested(value=1.5))
    output_path = export_model_as_yaml(model, tmp_path / "result.yaml")

    text = output_path.read_text()
    assert "maybe_missing: null" in text
