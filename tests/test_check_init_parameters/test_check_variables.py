import pytest

from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)


@pytest.mark.parametrize("_input_vars", [("var1", "var2"), {"var1": 1, "var2": 2}])
def test_raises_errors_when_not_list_str_or_int(_input_vars):
    with pytest.raises(ValueError) as record:
        assert _check_variables_input_value(_input_vars)
    msg = (
        "`variables` should contain a string, an integer or a list of strings or "
        f"integers. Got {_input_vars} instead."
    )
    assert str(record.value) == msg


@pytest.mark.parametrize(
    "_input_vars", [["var1", "var2", "var2", "var3"], [0, 1, 1, 2]]
)
def test_raises_error_when_duplicated_var_names(_input_vars):
    with pytest.raises(ValueError) as record:
        assert _check_variables_input_value(_input_vars)
    msg = "The list entered in `variables` contains duplicated variable names."
    assert str(record.value) == msg


def test_raises_error_when_empty_list():
    with pytest.raises(ValueError) as record:
        assert _check_variables_input_value([])
    msg = "The list of `variables` is empty."
    assert str(record.value) == msg


@pytest.mark.parametrize(
    "_input_vars",
    [["var1", "var2", "var3"], [0, 1, 2, 3], "var1", ["var1"], 0, [0]],
)
def test_return_variables(_input_vars):
    assert _check_variables_input_value(_input_vars) == _input_vars


def test_return_when_variables_is_none():
    assert _check_variables_input_value(None) is None
