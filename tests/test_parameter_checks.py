import pytest

from feature_engine._check_input_parameters.check_input_dictionary import (
    check_numerical_dict,
)


@pytest.mark.parametrize("input_dict", [{"a": 1, "b": "c"}, {1: 1, 2: "c"}])
def test_not_numerical_dict(input_dict):
    with pytest.raises(ValueError):
        check_numerical_dict(input_dict)


@pytest.mark.parametrize("input_dict", [[1, 2, 3], (1, 2, 3), "hola", 5])
def test_input_type(input_dict):
    with pytest.raises(TypeError):
        check_numerical_dict(input_dict)
