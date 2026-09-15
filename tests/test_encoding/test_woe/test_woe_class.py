import math
import re

import pytest

from feature_engine.encoding.woe import WoE
from tests.backend_helpers import frame_to_dict, make_series

DATA_ZERO = {
    "var_A": ["A"] * 9 + ["B"] * 6 + ["C"] * 3 + ["D"] * 2,
    "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
}


def test_woe_calculation(make_df, data_enc):
    X = make_df(data_enc)
    y = make_series(make_df, data_enc["target"])

    woe = WoE()._calculate_woe(X, y, "var_A").to_native()

    # 6 positive and 14 negative cases
    pos = [2 / 6, 2 / 6, 2 / 6]
    neg = [4 / 14, 8 / 14, 2 / 14]
    assert isinstance(woe, make_df)
    assert frame_to_dict(woe) == {
        "__category__": ["A", "B", "C"],
        "__pos__": pytest.approx(pos),
        "__neg__": pytest.approx(neg),
        "__woe__": pytest.approx([math.log(p / n) for p, n in zip(pos, neg)]),
    }


def test_woe_error(make_df):
    X = make_df(DATA_ZERO)
    y = make_series(make_df, DATA_ZERO["target"])
    msg = (
        "The proportion of one of the classes for a category in variable var_A "
        "is zero, and log of zero is not defined"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        WoE()._calculate_woe(X, y, "var_A")


@pytest.mark.parametrize("fill_value", [1, 10, 0.1])
def test_fill_value(make_df, fill_value):
    X = make_df(DATA_ZERO)
    y = make_series(make_df, DATA_ZERO["target"])

    woe = WoE()._calculate_woe(X, y, "var_A", fill_value=fill_value).to_native()

    # 7 positive and 13 negative cases; C has no negatives and D no positives
    pos = [2 / 7, 2 / 7, 3 / 7, fill_value]
    neg = [7 / 13, 4 / 13, fill_value, 2 / 13]
    assert isinstance(woe, make_df)
    assert frame_to_dict(woe) == {
        "__category__": ["A", "B", "C", "D"],
        "__pos__": pytest.approx(pos),
        "__neg__": pytest.approx(neg),
        "__woe__": pytest.approx([math.log(p / n) for p, n in zip(pos, neg)]),
    }
