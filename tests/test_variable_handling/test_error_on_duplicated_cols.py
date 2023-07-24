import pandas as pd
import pytest

from feature_engine._base_transformers.base_numerical import BaseNumericalTransformer
from feature_engine.datetime.datetime import DatetimeFeatures
from feature_engine.encoding.base_encoder import CategoricalMethodsMixin

df = pd.DataFrame(
    {
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"],
        "col3": pd.date_range("2023-01-01", periods=3),
    }
)
df.columns = ["same", "same", "same"]

class MockedCategoricalTransformer(CategoricalMethodsMixin):
    
    def __sklearn_is_fitted__(self):
        return True
    
    def fit(self, X, y=None, *args, **kwargs):
        return self


def test_error_on_autodetected_numerical():
    with pytest.raises(ValueError):
        est = BaseNumericalTransformer()
        est.variables = None
        est.fit(df)


def test_error_on_autodetected_categorical():
    with pytest.raises(ValueError):
        est = MockedCategoricalTransformer()
        est.variables = None
        est.ignore_format = True
        est.transform(df)


def test_error_on_autodetected_datetime():
    with pytest.raises(ValueError):
        DatetimeFeatures().fit(df)


def test_error_on_manually_specified_numerical():
    with pytest.raises(ValueError):
        est = BaseNumericalTransformer()
        est.variables = ["same"]
        est.fit(df)


def test_error_on_manually_specified_categorical():
    with pytest.raises(ValueError):
        est = MockedCategoricalTransformer()
        est.variables = ["same"]
        est.ignore_format = True
        est.transform(df)


def test_error_on_manually_specified_datetime():
    with pytest.raises(ValueError):
        DatetimeFeatures(variables=["same"]).fit(df)
