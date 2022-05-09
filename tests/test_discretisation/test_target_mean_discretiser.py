import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.discretisation import TargetMeanDiscretiser


def test_discretiser_using_equal_frequency():
    data = {
        "var_A": list(range(1, 11)),
        "var_B": list(range(2, 22, 2)),
        "var_C": ["A"] * 3 + ["B"] + ["C"] * 4 + ["D"] * 2,
        "var_D": list(range(3, 33, 3)),
    }
    df = pd.DataFrame(data)
    target = list(range(10))

    transformer = TargetMeanDiscretiser(
        variables=["var_A", "var_D"],
        bins=2
    )

    df_tr = transformer.fit_transform(df, target)

    #
    expected_results = {
        "var_A": [2.0] * 5 + [7.0] * 5,
        "var_B": list(range(2, 22, 2)),
        "var_C": ["A"] * 3 + ["B"] + ["C"] * 4 + ["D"] * 2,
        "var_D": [2.0] * 5 + [7.0] * 5,
    }
    expected_results_df = pd.DataFrame(expected_results)

    assert df_tr.equals(expected_results_df)
