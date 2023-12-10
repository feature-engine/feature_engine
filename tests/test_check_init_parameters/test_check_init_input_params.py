import pytest

from feature_engine._check_init_parameters.check_init_input_params import (
    _check_param_drop_original,
    _check_param_missing_values,
)


@pytest.mark.parametrize("missing_vals", [None, ["Hola"], True, "Hola"])
def test_check_param_missing_values(missing_vals):
    with pytest.raises(ValueError):
        _check_param_missing_values(missing_vals)


@pytest.mark.parametrize("drop_orig", [None, ["Hola"], 10, "Hola"])
def test_check_param_drop_original(drop_orig):
    with pytest.raises(ValueError):
        _check_param_drop_original(drop_orig)
