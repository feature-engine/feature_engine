import pytest

from feature_engine.encoding.base_encoder import CategoricalInitMixin


@pytest.mark.parametrize("param", [1, "hola", [1, 2, 0], (True, False)])
def test_categorical_init_mixin_raises_error(param):
    with pytest.raises(ValueError):
        CategoricalInitMixin(ignore_format=param)


@pytest.mark.parametrize("param", [True, False])
def test_categorical_init_mixin_assings_values_correctly(param):
    enc = CategoricalInitMixin(ignore_format=param)
    assert enc.ignore_format == param
