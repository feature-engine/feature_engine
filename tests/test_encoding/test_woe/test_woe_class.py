import math

import pytest

from feature_engine.encoding.woe import WoE
from tests.backend_helpers import frame_to_dict, make_series


def test_woe_calculation(make_df, data_enc):
    X = make_df(data_enc)
    y = make_series(make_df, data_enc["target"])

    woe, has_zero_counts = WoE()._calculate_woe(X, y, "var_A")
    woe = woe.to_native()

    # 6 positive and 14 negative cases
    pos = [2 / 6, 2 / 6, 2 / 6]
    neg = [4 / 14, 8 / 14, 2 / 14]
    assert has_zero_counts is False
    assert isinstance(woe, make_df)
    assert frame_to_dict(woe) == {
        "__category__": ["A", "B", "C"],
        "__pos__": pytest.approx(pos),
        "__neg__": pytest.approx(neg),
        "__woe__": pytest.approx([math.log(p / n) for p, n in zip(pos, neg)]),
    }


def test_zero_counts_are_replaced_by_half(make_df):
    data = {
        "var_A": ["A"] * 9 + ["B"] * 6 + ["C"] * 3 + ["D"] * 2,
        "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
    }
    X = make_df(data)
    y = make_series(make_df, data["target"])

    woe, has_zero_counts = WoE()._calculate_woe(X, y, "var_A")
    woe = woe.to_native()

    # 7 positive and 13 negative cases; C has no negatives and D no positives
    pos = [2 / 7, 2 / 7, 3 / 7, 0.5 / 7]
    neg = [7 / 13, 4 / 13, 0.5 / 13, 2 / 13]
    assert has_zero_counts is True
    assert isinstance(woe, make_df)
    assert frame_to_dict(woe) == {
        "__category__": ["A", "B", "C", "D"],
        "__pos__": pytest.approx(pos),
        "__neg__": pytest.approx(neg),
        "__woe__": pytest.approx([math.log(p / n) for p, n in zip(pos, neg)]),
    }
