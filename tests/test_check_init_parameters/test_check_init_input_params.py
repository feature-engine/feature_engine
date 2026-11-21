import re

import pytest

from feature_engine._check_init_parameters.check_init_input_params import (
    _check_param_drop_original,
    _check_param_missing_values,
)


@pytest.mark.parametrize(
    "missing_vals", [None, ["Hola"], ["raise"], ("ignore",), True, 1, "Hola", "Raise"]
)
def test_check_param_missing_values(missing_vals):
    msg = (
        "missing_values takes only values 'raise' or 'ignore'. "
        f"Got {missing_vals} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        _check_param_missing_values(missing_vals)


@pytest.mark.parametrize("drop_orig", [None, ["Hola"], 10, "Hola"])
def test_check_param_drop_original(drop_orig):
    msg = (
        "drop_original takes only boolean values True and False. "
        f"Got {drop_orig} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        _check_param_drop_original(drop_orig)
