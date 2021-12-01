import pytest

from feature_engine.parameter_checks import (
    _define_numerical_dict,
    _parse_features_to_extract,
)


def test_numerical_dict():
    input_dict = {"a": 1, "b": 2}
    expected_output = {"a": 1, "b": 2}

    assert _define_numerical_dict(input_dict) == expected_output


def test_not_numerical_dict():
    input_dict = {"a": 1, "b": "c"}

    with pytest.raises(ValueError):
        assert _define_numerical_dict(input_dict)


def test_input_type():
    input_dict = [1, 2, 3]

    with pytest.raises(TypeError):
        assert _define_numerical_dict(input_dict)


def test_parse_features_to_extract():

    supported = ["supported1", "supported2"]

    with pytest.raises(TypeError):
        _parse_features_to_extract(14198, supported)
        _parse_features_to_extract("supported1", "supported1")
        _parse_features_to_extract("supported1", ["supported1", 1.543])

    with pytest.raises(ValueError):
        _parse_features_to_extract("not_supported", supported)
        _parse_features_to_extract(["also not supp"], supported)
        _parse_features_to_extract(["supported1", 100], supported)

    assert _parse_features_to_extract("supported1", supported) == ["supported1"]
    assert _parse_features_to_extract(["supported2"], supported) == ["supported2"]
    assert _parse_features_to_extract("all", supported) == supported
