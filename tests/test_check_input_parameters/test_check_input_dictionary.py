import pytest

from feature_engine._check_init_parameters.check_input_dictionary import (
    _check_numerical_dict,
)


@pytest.mark.parametrize("input_dict", [{"a": 1, "b": "c"}, {1: 1, 2: "c"}])
def test_raises_error_when_item_in_dict_not_numerical(input_dict):
    with pytest.raises(ValueError):
        _check_numerical_dict(input_dict)


@pytest.mark.parametrize("input_dict", [[1, 2, 3], (1, 2, 3), "hola", 5])
def test_raises_error_when_input_not_dictionary_or_none(input_dict):
    with pytest.raises(TypeError):
        _check_numerical_dict(input_dict)
