import pytest
from numpy import inf
from pandas.testing import assert_frame_equal

from feature_engine._base_transformers.base_numerical import BaseNumericalTransformer
from tests.estimator_checks.non_fitted_error_checks import check_raises_non_fitted_error


class MockClass(BaseNumericalTransformer):
    def __init__(self):
        self.variables = None
        self.return_empty = False

    def transform(self, X):
        return self._check_transform_input_and_state(X)


def test_empty_find_numerical_variables(df_vartypes):
    transformer = MockClass()
    with pytest.raises(TypeError):
        transformer.fit(df_vartypes.drop(columns=["Age", "Marks"]))
    transformer = MockClass()
    transformer.return_empty = True
    transformer.fit(df_vartypes.drop(columns=["Age", "Marks"]))
    assert transformer.variables_ == []


def test_fit_method(df_vartypes, df_na):
    transformer = MockClass()
    res = transformer.fit(df_vartypes)
    assert transformer.feature_names_in_ == list(df_vartypes.columns)
    assert transformer.n_features_in_ == len(df_vartypes.columns)
    assert_frame_equal(res, df_vartypes)

    with pytest.raises(ValueError):
        transformer.fit(df_na)

    df_na = df_na.fillna(inf)
    with pytest.raises(ValueError):
        assert transformer.fit(df_na)


def test_transform_method(df_vartypes, df_na):
    transformer = MockClass()
    transformer.fit(df_vartypes)
    assert_frame_equal(
        transformer._check_transform_input_and_state(df_vartypes), df_vartypes
    )
    assert_frame_equal(
        transformer._check_transform_input_and_state(
            df_vartypes[["City", "Age", "Name", "Marks", "dob"]]
        ),
        df_vartypes,
    )

    with pytest.raises(ValueError):
        transformer.fit(df_na)

    df_na = df_na.fillna(inf)
    with pytest.raises(ValueError):
        assert transformer.fit(df_na)

    with pytest.raises(ValueError):
        assert transformer._check_transform_input_and_state(
            df_vartypes[["Age", "Marks"]]
        )


def test_raises_non_fitted_error():
    check_raises_non_fitted_error(MockClass())
