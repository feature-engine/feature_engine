import pytest

from feature_engine.encoding.base_encoder import CategoricalInitMixinNA


@pytest.mark.parametrize("param", [1, "hola", [1, 2, 0], (True, False)])
def test_raises_error_when_ignore_format_not_permitted(param):
    with pytest.raises(ValueError) as record:
        CategoricalInitMixinNA(ignore_format=param)
    msg = f"ignore_format takes only booleans True and False. Got {param} instead."
    assert str(record.value) == msg


@pytest.mark.parametrize("param", [1, "hola", [1, 2, 0], (True, False)])
def test_raises_error_when_missing_values_not_permitted(param):
    with pytest.raises(ValueError) as record:
        CategoricalInitMixinNA(missing_values=param)
    msg = f"missing_values takes only values 'raise' or 'ignore'. Got {param} instead."
    assert str(record.value) == msg


@pytest.mark.parametrize("param", [(True, "ignore"), (False, "raise")])
def test_correct_param_value_assignment(param):
    format_, na_ = param
    enc = CategoricalInitMixinNA(ignore_format=format_, missing_values=na_)
    assert enc.ignore_format == format_
    assert enc.missing_values == na_
