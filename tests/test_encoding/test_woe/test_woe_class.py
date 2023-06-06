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
    pd.testing.assert_series_equal(np.log(pos / neg), woe, check_names=False)


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
