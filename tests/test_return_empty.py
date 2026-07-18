"""Tests for the return_empty init parameter.

When variables is None and the dataframe has no variables of the type required
by the transformer, fit raises an error by default, and returns an empty
variable list when return_empty is True.
"""

import pandas as pd
import pytest

from feature_engine.datetime import (
    DatetimeFeatures,
    DatetimeOrdinal,
    DatetimeSubtraction,
)
from feature_engine.outliers import OutlierTrimmer, Winsorizer
from feature_engine.scaling import MeanNormalizationScaler
from feature_engine.timeseries.forecasting import (
    ExpandingWindowFeatures,
    LagFeatures,
    WindowFeatures,
)
from feature_engine.transformation import (
    ArcsinTransformer,
    ArcSinhTransformer,
    BoxCoxTransformer,
    LogCpTransformer,
    LogTransformer,
    PowerTransformer,
    ReciprocalTransformer,
    YeoJohnsonTransformer,
)
from feature_engine.wrappers import SklearnTransformerWrapper

_numerical_transformers = [
    ArcsinTransformer(),
    ArcSinhTransformer(),
    BoxCoxTransformer(),
    LogTransformer(),
    LogCpTransformer(),
    PowerTransformer(),
    ReciprocalTransformer(),
    YeoJohnsonTransformer(),
    MeanNormalizationScaler(),
    Winsorizer(),
    OutlierTrimmer(),
    LagFeatures(),
    WindowFeatures(),
    ExpandingWindowFeatures(),
]

_datetime_transformers = [
    DatetimeFeatures(),
    DatetimeOrdinal(),
    DatetimeSubtraction(),
]


@pytest.fixture
def df_categorical_only():
    return pd.DataFrame({"var_cat": ["A", "B", "C", "D"]})


@pytest.fixture
def df_numerical_only():
    return pd.DataFrame({"var_num": [1.0, 2.0, 3.0, 4.0]})


@pytest.mark.parametrize("transformer", _numerical_transformers)
def test_numerical_transformers_raise_by_default(transformer, df_categorical_only):
    with pytest.raises(TypeError):
        transformer.fit(df_categorical_only)


@pytest.mark.parametrize("transformer", _numerical_transformers)
def test_numerical_transformers_return_empty(transformer, df_categorical_only):
    transformer = type(transformer)(return_empty=True)
    with pytest.warns(UserWarning):
        transformer.fit(df_categorical_only)
    assert transformer.variables_ == []


@pytest.mark.parametrize("transformer", _datetime_transformers)
def test_datetime_transformers_raise_by_default(transformer, df_numerical_only):
    with pytest.raises(TypeError):
        transformer.fit(df_numerical_only)


@pytest.mark.parametrize("transformer", _datetime_transformers)
def test_datetime_transformers_return_empty(transformer, df_numerical_only):
    transformer = type(transformer)(return_empty=True)
    with pytest.warns(UserWarning):
        transformer.fit(df_numerical_only)
    assert transformer.variables_ == []


def test_wrapper_raises_by_default(df_categorical_only):
    from sklearn.preprocessing import StandardScaler

    transformer = SklearnTransformerWrapper(transformer=StandardScaler())
    with pytest.raises(TypeError):
        transformer.fit(df_categorical_only)


@pytest.mark.parametrize(
    "transformer", _numerical_transformers + _datetime_transformers
)
def test_return_empty_takes_booleans_only(transformer):
    with pytest.raises(ValueError):
        type(transformer)(return_empty="yes")
