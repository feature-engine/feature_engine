import pytest

from feature_engine.encoding.base_encoder import CategoricalInitMixin


@pytest.mark.parametrize("param", [1, "hola", [1, 2, 0], (True, False)])
def test_raises_error_when_ignore_format_not_permitted(param):
    with pytest.raises(ValueError) as record:
        CategoricalInitMixin(ignore_format=param)
    msg = f"ignore_format takes only booleans True and False. Got {param} instead."
    assert str(record.value) == msg


@pytest.mark.parametrize("param", [True, False])
def test_ignore_format_value_assignment(param):
    enc = CategoricalInitMixin(ignore_format=param)
    assert enc.ignore_format == param
