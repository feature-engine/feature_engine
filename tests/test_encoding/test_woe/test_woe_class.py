import numpy as np
import pandas as pd
import pytest

from feature_engine.encoding.woe import WoE


def test_woe_calculation(df_enc):
    pos_exp = pd.Series({"A": 0.333333, "B": 0.333333, "C": 0.333333})
    neg_exp = pd.Series({"A": 0.285714, "B": 0.571429, "C": 0.142857})

    woe_class = WoE()
    pos, neg, woe = woe_class._calculate_woe(df_enc, df_enc["target"], "var_A")

    pd.testing.assert_series_equal(pos, pos_exp, check_names=False)
    pd.testing.assert_series_equal(neg, neg_exp, check_names=False)
    pd.testing.assert_series_equal(np.log(pos_exp / neg_exp), woe, check_names=False)


def test_woe_error():
    df = {
        "var_A": ["B"] * 9 + ["A"] * 6 + ["C"] * 3 + ["D"] * 2,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
    }
    df = pd.DataFrame(df)
    woe_class = WoE()

    with pytest.raises(ValueError):
        woe_class._calculate_woe(df, df["target"], "var_A")


@pytest.mark.parametrize("fill_value", [1, 10, 0.1])
def test_fill_value(fill_value):
    df = {
        "var_A": ["A"] * 9 + ["B"] * 6 + ["C"] * 3 + ["D"] * 2,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
    }
    df = pd.DataFrame(df)

    pos_exp = pd.Series(
        {
            "A": 0.2857142857142857,
            "B": 0.2857142857142857,
            "C": 0.42857142857142855,
            "D": fill_value,
        }
    )
    neg_exp = pd.Series(
        {
            "A": 0.5384615384615384,
            "B": 0.3076923076923077,
            "C": fill_value,
            "D": 0.15384615384615385,
        }
    )

    woe_class = WoE()
    pos, neg, woe = woe_class._calculate_woe(
        df, df["target"], "var_A", fill_value=fill_value
    )

    pd.testing.assert_series_equal(pos, pos_exp, check_names=False)
    pd.testing.assert_series_equal(neg, neg_exp, check_names=False)
    pd.testing.assert_series_equal(np.log(pos_exp / neg_exp), woe, check_names=False)
