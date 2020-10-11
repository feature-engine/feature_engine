
import pytest

from feature_engine.parameter_checks import _define_numerical_dict


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
