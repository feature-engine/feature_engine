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
    StringSimilarityEncoder
)
from tests.estimator_checks.estimator_checks import check_feature_engine_estimator

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
    StringSimilarityEncoder(ignore_format=True),
]


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_sklearn(estimator):
    return check_estimator(estimator)


_estimators = [
    CountFrequencyEncoder(),
    DecisionTreeEncoder(regression=False),
    MeanEncoder(),
    OneHotEncoder(),
    OrdinalEncoder(),
    RareLabelEncoder(),
    WoEEncoder(),
    PRatioEncoder(),
    StringSimilarityEncoder(),
]


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_feature_engine(estimator):
    return check_feature_engine_estimator(estimator)


@pytest.mark.parametrize(
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
                {"x0": [25.5, 25.5, 25.5, 25.5, 45.5, 45.5]},
            ),
        ),
        (
            MeanEncoder(),
            pd.DataFrame(
                {"x": ["a", "a", "b", "b", "c", "c"], "y": [1, 0, 1, 0, 1, 0]},
                index=[101, 105, 42, 76, 88, 92],
            ),
            pd.DataFrame(
                {"x0": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]},
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
            pd.DataFrame({"x0": [2, 2, 2, 1, 1, 1, 0, 0, 0]}),
        ),
        (
            PRatioEncoder(),
            pd.DataFrame(
                {"x": ["a", "a", "b", "b", "c", "c"], "y": [1, 0, 1, 0, 1, 0]},
                index=[101, 105, 42, 76, 88, 92],
            ),
            pd.DataFrame(
                {"x0": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
            ),
        ),
        (
            WoEEncoder(),
            pd.DataFrame(
                {"x": ["a", "a", "b", "b", "c", "c"], "y": [1, 0, 1, 0, 1, 0]},
                index=[101, 105, 42, 76, 88, 92],
            ),
            pd.DataFrame(
                {"x0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            ),
        ),
    ],
)
def test_encoders_when_x_numpy_y_pandas(encoder, df_test, df_expected):
    """
    Created 2022-03-27 to test fix to issue # 376
    Code adapted from:
    https://github.com/scikit-learn-contrib/category_encoders/issues/280
    """

    X = df_test[["x"]]
    y = df_test["y"]

    # Test issue where X is array,
    # y remains Series with original index
    X_2 = X.to_numpy()
    df_result = encoder.fit_transform(X_2, y)
    assert df_result.equals(df_expected)


@pytest.mark.parametrize(
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
def test_encoders_when_x_pandas_y_numpy(encoder, df_test, df_expected):
    """
    Created 2022-03-27 to test fix to issue # 376
    Code adapted from:
    https://github.com/scikit-learn-contrib/category_encoders/issues/280
    """

    X = df_test[["x"]]
    y = df_test["y"]

    # Test issue fix where X becomes array,
    # y remains Series with original DataFrame index
    y_2 = y.to_numpy()
    df_result = encoder.fit_transform(X, y_2)
    assert df_result.equals(df_expected)


@pytest.mark.parametrize(
    "encoder, df_test",
    [
        (
            DecisionTreeEncoder(),
            pd.DataFrame(
                {"x": ["a", "a", "b", "b", "c", "c"], "y": [21, 30, 21, 30, 51, 40]},
                index=[101, 105, 42, 76, 88, 92],
            ),
        ),
        (
            MeanEncoder(),
            pd.DataFrame(
                {"x": ["a", "a", "b", "b", "c", "c"], "y": [1, 0, 1, 0, 1, 0]},
                index=[101, 105, 42, 76, 88, 92],
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
        ),
        (
            PRatioEncoder(),
            pd.DataFrame(
                {"x": ["a", "a", "b", "b", "c", "c"], "y": [1, 0, 1, 0, 1, 0]},
                index=[101, 105, 42, 76, 88, 92],
            ),
        ),
        (
            WoEEncoder(),
            pd.DataFrame(
                {"x": ["a", "a", "b", "b", "c", "c"], "y": [1, 0, 1, 0, 1, 0]},
                index=[101, 105, 42, 76, 88, 92],
            ),
        ),
    ],
)
def test_encoders_raise_error_when_x_pandas_y_pandas_index_mismatch(encoder, df_test):
    """
    Created 2022-03-27 to test fix to issue # 376
    """

    X: pd.DataFrame = df_test[["x"]]
    y: pd.Series = df_test["y"]

    # Test issue fix where indexes of pandas objects are mismatched
    y = y.reset_index(drop=True)

    e: Exception
    with pytest.raises(Exception) as e:
        encoder.fit_transform(X, y)
    assert "indexes" in e.value.args[0].lower()
