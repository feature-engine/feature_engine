import pytest

from feature_engine.variable_handling._init_parameter_checks import (
    _check_init_parameter_variables,
)


@pytest.mark.parametrize(
    "_input_vars",
    [
        ("var1", "var2"),
        {"var1": 1, "var2": 2},
        ["var1", "var2", "var2", "var3"],
        [0, 1, 1, 2],
    ],
)
def test_check_init_parameter_variables_raises_errors(_input_vars):
    with pytest.raises(ValueError):
        assert _check_init_parameter_variables(_input_vars)


@pytest.mark.parametrize(
    "_input_vars",
    [["var1", "var2", "var3"], [0, 1, 2, 3], "var1", ["var1"], 0, [0]],
)
def test_check_init_parameter_variables(_input_vars):
    assert _check_init_parameter_variables(_input_vars) == _input_vars


def test_check_init_parameter_variables_is_none():
    assert _check_init_parameter_variables(None) is None
