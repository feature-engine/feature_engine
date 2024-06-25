"""Dataframe used as input by many estimator checks."""

from typing import Tuple

import pandas as pd
from sklearn.datasets import make_classification


def test_df(
    categorical: bool = False, datetime: bool = False
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Creates a dataframe that contains only numerical features, or additionally,
    categorical and datetime features.

    Parameters
    ----------
    categorical: bool, default=False
        Whether to add 2 additional categorical features.

    datetime: bool, default=False
        Whether to add one additional datetime feature.

    Returns
    -------
    X: pd.DataFrame
        A pandas dataframe.
    """
    X, y = make_classification(
        n_samples=1000,
        n_features=12,
        n_redundant=4,
        n_clusters_per_class=1,
        weights=[0.50],
        class_sep=2,
        random_state=1,
    )

    # transform arrays into pandas df and series
    colnames = [f"var_{i}" for i in range(12)]
    X = pd.DataFrame(X, columns=colnames)
    y = pd.Series(y)

    if categorical is True:
        X["cat_var1"] = ["A"] * 1000
        X["cat_var2"] = ["B"] * 1000

    if datetime is True:
        X["date1"] = pd.date_range("2020-02-24", periods=1000, freq="min")
        X["date2"] = pd.date_range("2021-09-29", periods=1000, freq="h")

    return X, y
