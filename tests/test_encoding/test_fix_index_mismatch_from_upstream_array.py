import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.encoding import (
    MeanEncoder,
    WoEEncoder,
    PRatioEncoder,
)

from sklearn.impute import SimpleImputer


@pytest.mark.parametrize(
    # Encoders that encode X as a function of y; this is what
    # breaks down when X becomes an array and indexes don't accidentally match in final
    # concantenation
    "encoder", [MeanEncoder(), WoEEncoder(), PRatioEncoder()]
)
def test_fix_index_mismatch_from_upstream_array(encoder):
    """
    Created 2022-02-19 to test fix to issue # 376
    Code adapted from: https://github.com/scikit-learn-contrib/category_encoders/issues/280
    """

    # test dataframe; setup for a transfromation where
    # coded version of 'x' will be a function of target 'y'
    df: pd.DataFrame = pd.DataFrame({
        'x': ['a', 'a', 'b', 'b', 'c', 'c'],
        'y': [1, 0, 1, 0, 1, 0],
    })
    # Key - "non-standard" index that is not the usual
    # contiguous range starting a t 0
    df.index = [101, 105, 42, 76, 88, 92]

    # Set up for standard pipeline/training etc.
    X: pd.DataFrame = df[["x"]]
    y: pd.Series = df["y"]

    # Will serve as a no-op whose chief purpose is to turn the
    # X into an np.ndarray
    si = SimpleImputer(strategy="constant", fill_value="a")

    # Sequence leading to issue:
    # 1) X becomes an array
    assert type(X) == pd.DataFrame
    X_2: np.array = si.fit_transform(X)
    assert type(X_2) == np.ndarray

    # 2) Encoder encodes as function of X, y
    df_result: pd.DataFrame = encoder.fit_transform(X_2, y)
    assert type(df_result) == pd.DataFrame

    # Assertion fails: breakdown in index matches causes results to be all nan
    assert all(df_result.iloc[:, 0].notnull())