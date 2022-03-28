import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.encoding import (
    CountFrequencyEncoder,
    DecisionTreeEncoder,
    MeanEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    PRatioEncoder,
    RareLabelEncoder,
    WoEEncoder,
)
from feature_engine.estimator_checks import check_feature_engine_estimator

_estimators = [
    CountFrequencyEncoder(ignore_format=True),
    DecisionTreeEncoder(regression=False, ignore_format=True),
    MeanEncoder(ignore_format=True),
    OneHotEncoder(ignore_format=True),
    OrdinalEncoder(ignore_format=True),
    RareLabelEncoder(
        tol=0.00000000001,
        n_categories=100000000000,
        replace_with=10,
        ignore_format=True,
    ),
    WoEEncoder(ignore_format=True),
    PRatioEncoder(ignore_format=True),
]

@pytest.mark.parametrize(
    "Estimator",
    [
        CountFrequencyEncoder(ignore_format=True),
        DecisionTreeEncoder(regression=False, ignore_format=True),
        MeanEncoder(ignore_format=True),
        OneHotEncoder(ignore_format=True),
        OrdinalEncoder(ignore_format=True),
        RareLabelEncoder(
            tol=0.00000000001,
            n_categories=100000000000,
            replace_with=10,
            ignore_format=True,
        ),
        WoEEncoder(ignore_format=True),
        PRatioEncoder(ignore_format=True),
    ],
)
def test_all_transformers(Estimator):
    return check_estimator(Estimator)


@pytest.mark.parametrize(
    # Encoders that encode X as a function of y; this is what
    # breaks down when X becomes an array and indexes don't
    # accidentally match in final concantenation
    # All test DataFrames have same data except DecisionTreeEncoder(),
    # which needs different y values.
    # Key to all: - "non-standard" index that is not the usual
    # contiguous range starting a t 0
    "encoder, df_test, df_expected",
    [
        (
            DecisionTreeEncoder(),
            pd.DataFrame(
                {"x": ["a", "a", "b", "b", "c", "c"], "y": [21, 30, 21, 30, 51, 40]},
                index=[101, 105, 42, 76, 88, 92],
            ),
            pd.DataFrame(
                {"0": [25.5, 25.5, 25.5, 25.5, 45.5, 45.5]},
            ),
        ),
        (
            MeanEncoder(),
            pd.DataFrame(
                {"x": ["a", "a", "b", "b", "c", "c"], "y": [1, 0, 1, 0, 1, 0]},
                index=[101, 105, 42, 76, 88, 92],
            ),
            pd.DataFrame(
                {"0": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]},
            ),
        ),
        (
            OrdinalEncoder(encoding_method="ordered"),
            pd.DataFrame(
                {
                    "x": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
                    "y": [3, 3, 3, 2, 2, 2, 1, 1, 1],
                },
                index=[33, 5412, 66, 99, 334, 1212, 22, 555, 1],
            ),
            pd.DataFrame({"0": [2, 2, 2, 1, 1, 1, 0, 0, 0]}),
        ),
        (
            PRatioEncoder(),
            pd.DataFrame(
                {"x": ["a", "a", "b", "b", "c", "c"], "y": [1, 0, 1, 0, 1, 0]},
                index=[101, 105, 42, 76, 88, 92],
            ),
            pd.DataFrame(
                {"0": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
            ),
        ),
        (
            WoEEncoder(),
            pd.DataFrame(
                {"x": ["a", "a", "b", "b", "c", "c"], "y": [1, 0, 1, 0, 1, 0]},
                index=[101, 105, 42, 76, 88, 92],
            ),
            pd.DataFrame(
                {"0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            ),
        ),
    ],
)
def test_fix_index_mismatch_from_x_numpy_y_pandas(encoder, df_test, df_expected):
    """
    Created 2022-03-27 to test fix to issue # 376
    Code adapted from:
    https://github.com/scikit-learn-contrib/category_encoders/issues/280
    """

    # Set up for standard pipeline/training etc.
    X: pd.DataFrame = df_test[["x"]]
    y: pd.Series = df_test["y"]

    # Test issue fix where X becomes array,
    # y remains Series with original DataFrame index
    X_2: np.ndarray = X.to_numpy()
    df_result: pd.DataFrame = encoder.fit_transform(X_2, y)
    assert df_result.equals(df_expected)


@pytest.mark.parametrize(
    # Encoders that encode X as a function of y; this is what
    # breaks down when y becomes an array and indexes don't
    # accidentally match in final concantenation
    # All test DataFrames have same data except DecisionTreeEncoder(),
    # which needs different y values.
    # Key to all: - "non-standard" index that is not the usual
    # contiguous range starting a t 0
    "encoder, df_test, df_expected",
    [
        (
            DecisionTreeEncoder(),
            pd.DataFrame(
                {"x": ["a", "a", "b", "b", "c", "c"], "y": [21, 30, 21, 30, 51, 40]},
                index=[101, 105, 42, 76, 88, 92],
            ),
            pd.DataFrame(
                {"x": [25.5, 25.5, 25.5, 25.5, 45.5, 45.5]},
                index=[101, 105, 42, 76, 88, 92],
            ),
        ),
        (
            MeanEncoder(),
            pd.DataFrame(
                {"x": ["a", "a", "b", "b", "c", "c"], "y": [1, 0, 1, 0, 1, 0]},
                index=[101, 105, 42, 76, 88, 92],
            ),
            pd.DataFrame(
                {"x": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}, index=[101, 105, 42, 76, 88, 92]
            ),
        ),
        (
            OrdinalEncoder(encoding_method="ordered"),
            pd.DataFrame(
                {
                    "x": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
                    "y": [3, 3, 3, 2, 2, 2, 1, 1, 1],
                },
                index=[33, 5412, 66, 99, 334, 1212, 22, 555, 1],
            ),
            pd.DataFrame(
                {"x": [2, 2, 2, 1, 1, 1, 0, 0, 0]},
                index=[33, 5412, 66, 99, 334, 1212, 22, 555, 1],
            ),
        ),
        (
            PRatioEncoder(),
            pd.DataFrame(
                {"x": ["a", "a", "b", "b", "c", "c"], "y": [1, 0, 1, 0, 1, 0]},
                index=[101, 105, 42, 76, 88, 92],
            ),
            pd.DataFrame(
                {"x": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}, index=[101, 105, 42, 76, 88, 92]
            ),
        ),
        (
            WoEEncoder(),
            pd.DataFrame(
                {"x": ["a", "a", "b", "b", "c", "c"], "y": [1, 0, 1, 0, 1, 0]},
                index=[101, 105, 42, 76, 88, 92],
            ),
            pd.DataFrame(
                {"x": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}, index=[101, 105, 42, 76, 88, 92]
            ),
        ),
    ],
)
def test_fix_index_mismatch_from_x_pandas_y_numpy(encoder, df_test, df_expected):
    """
    Created 2022-03-27 to test fix to issue # 376
    Code adapted from:
    https://github.com/scikit-learn-contrib/category_encoders/issues/280
    """

    # Set up for standard pipeline/training etc.
    X: pd.DataFrame = df_test[["x"]]
    y: pd.Series = df_test["y"]

    # Test issue fix where X becomes array,
    # y remains Series with original DataFrame index
    y_2: np.ndarray = y.to_numpy()
    df_result: pd.DataFrame = encoder.fit_transform(X, y_2)
    assert df_result.equals(df_expected)
