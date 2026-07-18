"""Tests for the return_empty init parameter in the transformers that already
expose it: discretisers, encoders, imputers and creation transformers.

When variables is None and the dataframe has no variables of the required type,
fit raises an error by default, and returns an empty variable list when
return_empty is True.
"""

import pandas as pd
import pytest

from feature_engine.creation import CyclicalFeatures
from feature_engine.discretisation import (
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser,
    GeometricWidthDiscretiser,
)
from feature_engine.encoding import (
    CountFrequencyEncoder,
    MeanEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    RareLabelEncoder,
    StringSimilarityEncoder,
)
from feature_engine.imputation import (
    ArbitraryNumberImputer,
    CategoricalImputer,
    EndTailImputer,
    MeanMedianImputer,
)

_numerical_transformers = [
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser,
    GeometricWidthDiscretiser,
    CyclicalFeatures,
    ArbitraryNumberImputer,
    EndTailImputer,
    MeanMedianImputer,
]

_categorical_transformers = [
    CountFrequencyEncoder,
    MeanEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    RareLabelEncoder,
    StringSimilarityEncoder,
    CategoricalImputer,
]


@pytest.fixture
def df_categorical_only():
    return pd.DataFrame({"var_cat": ["A", "B", "A", "B"]})


@pytest.fixture
def df_numerical_only():
    return pd.DataFrame({"var_num": [1.0, 2.0, 3.0, 4.0]})


@pytest.fixture
def target():
    return pd.Series([0, 1, 0, 1])


@pytest.mark.parametrize("transformer", _numerical_transformers)
def test_numerical_transformers_raise_by_default(
    transformer, df_categorical_only, target
):
    with pytest.raises(TypeError):
        transformer().fit(df_categorical_only, target)


@pytest.mark.parametrize("transformer", _numerical_transformers)
def test_numerical_transformers_return_empty(
    transformer, df_categorical_only, target
):
    transformer = transformer(return_empty=True)
    with pytest.warns(UserWarning):
        transformer.fit(df_categorical_only, target)
    assert transformer.variables_ == []


@pytest.mark.parametrize("transformer", _categorical_transformers)
def test_categorical_transformers_raise_by_default(
    transformer, df_numerical_only, target
):
    with pytest.raises(TypeError):
        transformer().fit(df_numerical_only, target)


@pytest.mark.parametrize("transformer", _categorical_transformers)
def test_categorical_transformers_return_empty(
    transformer, df_numerical_only, target
):
    transformer = transformer(return_empty=True)
    with pytest.warns(UserWarning):
        transformer.fit(df_numerical_only, target)
    assert transformer.variables_ == []


@pytest.mark.parametrize(
    "transformer", _numerical_transformers + _categorical_transformers
)
def test_return_empty_takes_booleans_only(transformer):
    with pytest.raises(ValueError):
        transformer(return_empty="yes")
