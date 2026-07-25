import re

import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.scaling import MeanNormalisationScaler, MeanNormalizationScaler
from tests.estimator_checks.fit_functionality_checks import check_return_empty

DEPRECATION_WARNING = (
    "MeanNormalizationScaler was deprecated in favour of "
    "MeanNormalisationScaler in version 2.0.0 and will be removed in version 2.1.0. "
    "To silence this warning, use MeanNormalisationScaler instead."
)


@pytest.fixture(
    params=[MeanNormalisationScaler, MeanNormalizationScaler],
    ids=["MeanNormalisationScaler", "MeanNormalizationScaler"],
)
def transformer_class(request):
    return request.param


def make_transformer(transformer_class, **kwargs):
    if transformer_class is MeanNormalizationScaler:
        with pytest.warns(FutureWarning, match=re.escape(DEPRECATION_WARNING)):
            return transformer_class(**kwargs)
    return transformer_class(**kwargs)


def test_mean_normalization_scaler_raises_future_warning():
    with pytest.warns(FutureWarning, match=re.escape(DEPRECATION_WARNING)):
        MeanNormalizationScaler()


def test_transforming_int_vars(transformer_class):
    # input test case
    df = pd.DataFrame(
        {
            "var1": [1.0, 2.0, 3.0],
            "var2": [4.0, 5.0, 3.0],
            "var3": [40.0, 20.0, 30.0],
        }
    )
    # expected output
    expected_df = pd.DataFrame(
        {
            "var1": [-0.5, 0.0, 0.5],
            "var2": [0, 0.5, -0.5],
            "var3": [0.5, -0.5, 0.0],
        }
    )

    transformer = make_transformer(transformer_class, variables=None)
    X = transformer.fit_transform(df)

    pd.testing.assert_frame_equal(X, expected_df)

    # test inverse_transform
    Xit = transformer.inverse_transform(X)

    pd.testing.assert_frame_equal(Xit, df)


def test_mean_normalization_plus_automatically_find_variables(
    df_vartypes, transformer_class
):
    # test case 1: automatically select variables
    transformer = make_transformer(transformer_class, variables=None)
    X = transformer.fit_transform(df_vartypes)

    # expected output
    transf_df = df_vartypes.copy()
    transf_df["Age"] = [0.16666, 0.5, -0.16666, -0.5]
    transf_df["Marks"] = [0.49999, 0.16666, -0.16666, -0.5]

    # test init params
    assert transformer.variables is None
    # test fit attr
    assert transformer.variables_ == ["Age", "Marks"]
    assert transformer.n_features_in_ == 5
    # test transform output
    pd.testing.assert_frame_equal(X, transf_df, rtol=10e-3)

    # test inverse_transform
    Xit = transformer.inverse_transform(X)

    # convert numbers to original format.
    Xit["Age"] = Xit["Age"].round().astype("int64")
    Xit["Marks"] = Xit["Marks"].round(1)

    # test
    pd.testing.assert_frame_equal(Xit, df_vartypes, rtol=10e-3)


def test_mean_normalization_plus_user_passes_var_list(df_vartypes, transformer_class):
    # test case 2: user passes variables
    transformer = make_transformer(transformer_class, variables="Age")
    X = transformer.fit_transform(df_vartypes)

    # expected output
    transf_df = df_vartypes.copy()
    transf_df["Age"] = [0.16666, 0.5, -0.16666, -0.5]

    # test init params
    assert transformer.variables == "Age"
    # test fit attr
    assert transformer.variables_ == ["Age"]
    assert transformer.n_features_in_ == 5
    # test transform output
    pd.testing.assert_frame_equal(X, transf_df, rtol=10e-3)

    # test inverse_transform
    Xit = transformer.inverse_transform(X)

    # convert numbers to original format.
    Xit["Age"] = Xit["Age"].round().astype("int64")

    # test
    pd.testing.assert_frame_equal(Xit, df_vartypes, rtol=10e-3)


def test_fit_raises_error_if_na_in_df(df_na, transformer_class):
    # test case 3: when dataset contains na, fit method
    transformer = make_transformer(transformer_class)
    with pytest.raises(ValueError):
        transformer.fit(df_na)


def test_transform_raises_error_if_na_in_df(df_vartypes, df_na, transformer_class):
    # test case 4: when dataset contains na, transform method
    transformer = make_transformer(transformer_class)
    transformer.fit(df_vartypes)
    with pytest.raises(ValueError):
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])


def test_non_fitted_error(df_vartypes, transformer_class):
    transformer = make_transformer(transformer_class)
    with pytest.raises(NotFittedError):
        transformer.transform(df_vartypes)


def test_constant_columns_error(transformer_class):
    # input test case
    df = pd.DataFrame(
        {
            "var1": [1.0, 2.0, 3.0],
            "var2": [4.0, 5.0, 3.0],
            "var3": [7.0, 7.0, 7.0],
        }
    )

    transformer = make_transformer(transformer_class)
    with pytest.raises(ValueError, match=re.escape("Division by zero is not allowed")):
        transformer.fit(df)


def test_check_return_empty(transformer_class):
    transformer = make_transformer(transformer_class)
    if transformer_class is MeanNormalizationScaler:
        with pytest.warns(FutureWarning, match=re.escape(DEPRECATION_WARNING)):
            check_return_empty(transformer)
    else:
        check_return_empty(transformer)
