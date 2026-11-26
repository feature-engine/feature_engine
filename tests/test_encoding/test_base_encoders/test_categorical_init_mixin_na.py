import re

import pytest

from feature_engine.encoding.base_encoder import CategoricalInitMixinNA


@pytest.mark.parametrize("param", [1, "hola", [1, 2, 0], (True, False)])
def test_raises_error_when_ignore_format_not_permitted(param):
    msg = f"ignore_format takes only booleans True and False. Got {param} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        CategoricalInitMixinNA(ignore_format=param)


@pytest.mark.parametrize(
    "param", [1, "hola", "Raise", None, [1, 2, 0], ["raise"], (True, False)]
)
def test_raises_error_when_missing_values_not_permitted(param):
    msg = f"missing_values takes only values 'raise' or 'ignore'. Got {param} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        CategoricalInitMixinNA(missing_values=param)


@pytest.mark.parametrize("param", [(True, "ignore"), (False, "raise")])
def test_correct_param_value_assignment(param):
    format_, na_ = param
    enc = CategoricalInitMixinNA(ignore_format=format_, missing_values=na_)
    assert enc.ignore_format == format_
    assert enc.missing_values == na_
