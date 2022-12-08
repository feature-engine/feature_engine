import pytest

from feature_engine.encoding.base_encoder import CategoricalInitMixinNA


@pytest.mark.parametrize("param", [1, "hola", [1, 2, 0], (True, False)])
def test_categorical_init_mixin_na_raises_error(param):
    with pytest.raises(ValueError):
        CategoricalInitMixinNA(ignore_format=param)

    with pytest.raises(ValueError):
        CategoricalInitMixinNA(missing_values=param)


@pytest.mark.parametrize("param", [(True, "ignore"), (False, "raise")])
def test_categorical_init_mixin_na_assings_values_correctly(param):
    format_, na_ = param
    enc = CategoricalInitMixinNA(ignore_format=format_, missing_values=na_)

    assert enc.ignore_format == format_
    assert enc.missing_values == na_
