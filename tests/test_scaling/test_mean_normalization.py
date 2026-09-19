import re

import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.scaling import MeanNormalisationScaler, MeanNormalizationScaler
from tests.backend_helpers import frame_to_dict
from tests.estimator_checks.fit_functionality_checks import check_return_empty
from tests.estimator_checks.non_fitted_error_checks import (
    check_raises_non_fitted_error_when_fit_fails,
)

DEPRECATION_WARNING = (
    "MeanNormalizationScaler was deprecated in favour of "
    "MeanNormalisationScaler in version 2.0.0 and will be removed in version 2.1.0. "
    "To silence this warning, use MeanNormalisationScaler instead."
)

MSG_NA = (
    "Some of the variables in the dataset contain NaN. Check and "
    "remove those before using this transformer."
)

DATA = {
    "Name": ["tom", "nick", "krish", "jack"],
    "City": ["London", "Manchester", "Liverpool", "Bristol"],
    "Age": [20, 21, 19, 18],
    "Marks": [0.9, 0.8, 0.7, 0.6],
}


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


def test_transform_and_inverse_transform_numerical_variables(
    make_df, transformer_class
):
    data = {
        "var1": [1.0, 2.0, 3.0],
        "var2": [4.0, 5.0, 3.0],
        "var3": [40.0, 20.0, 30.0],
    }

    transformer = make_transformer(transformer_class, variables=None)
    X = transformer.fit_transform(make_df(data))
    assert isinstance(X, make_df)
    assert frame_to_dict(X) == {
        "var1": pytest.approx([-0.5, 0.0, 0.5]),
        "var2": pytest.approx([0, 0.5, -0.5]),
        "var3": pytest.approx([0.5, -0.5, 0.0]),
    }

    Xit = transformer.inverse_transform(X)
    assert isinstance(Xit, make_df)
    assert frame_to_dict(Xit) == {
        col: pytest.approx(values) for col, values in data.items()
    }


def test_mean_normalization_plus_automatically_find_variables(
    make_df, transformer_class
):
    transformer = make_transformer(transformer_class, variables=None)
    X = transformer.fit_transform(make_df(DATA))

    assert transformer.variables_ == ["Age", "Marks"]
    assert transformer.n_features_in_ == 4

    assert isinstance(X, make_df)
    assert frame_to_dict(X) == {
        "Name": DATA["Name"],
        "City": DATA["City"],
        "Age": pytest.approx([0.16667, 0.5, -0.16667, -0.5], abs=1e-4),
        "Marks": pytest.approx([0.5, 0.16667, -0.16667, -0.5], abs=1e-4),
    }

    Xit = transformer.inverse_transform(X)
    assert isinstance(Xit, make_df)
    assert frame_to_dict(Xit) == {
        "Name": DATA["Name"],
        "City": DATA["City"],
        "Age": pytest.approx(DATA["Age"]),
        "Marks": pytest.approx(DATA["Marks"]),
    }


def test_mean_normalization_plus_user_passes_var_list(make_df, transformer_class):
    transformer = make_transformer(transformer_class, variables="Age")
    X = transformer.fit_transform(make_df(DATA))

    assert transformer.variables_ == ["Age"]
    assert transformer.n_features_in_ == 4

    assert isinstance(X, make_df)
    assert frame_to_dict(X) == {
        "Name": DATA["Name"],
        "City": DATA["City"],
        "Age": pytest.approx([0.16667, 0.5, -0.16667, -0.5], abs=1e-4),
        "Marks": DATA["Marks"],
    }

    Xit = transformer.inverse_transform(X)
    assert isinstance(Xit, make_df)
    assert frame_to_dict(Xit) == {
        "Name": DATA["Name"],
        "City": DATA["City"],
        "Age": pytest.approx(DATA["Age"]),
        "Marks": DATA["Marks"],
    }


def test_fit_raises_error_if_na_in_df(make_df, transformer_class):
    data_na = dict(DATA)
    data_na["Age"] = [20, None, 19, 18]

    transformer = make_transformer(transformer_class)
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        transformer.fit(make_df(data_na))


def test_transform_raises_error_if_na_in_df(make_df, transformer_class):
    data_na = dict(DATA)
    data_na["Age"] = [20, None, 19, 18]

    transformer = make_transformer(transformer_class)
    transformer.fit(make_df(DATA))
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        transformer.transform(make_df(data_na))


def test_non_fitted_error(make_df, transformer_class):
    transformer = make_transformer(transformer_class)
    msg = (
        f"This {transformer_class.__name__} instance is not fitted yet. Call 'fit' "
        "with appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=re.escape(msg)):
        transformer.transform(make_df(DATA))


def test_constant_columns_error(make_df, transformer_class):
    data = {
        "var1": [1.0, 2.0, 3.0],
        "var2": [4.0, 5.0, 3.0],
        "var3": [7.0, 7.0, 7.0],
    }

    transformer = make_transformer(transformer_class)
    msg = (
        "The following variable(s) are constant: ['var3']. "
        "Division by zero is not allowed. Please remove constant columns."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer.fit(make_df(data))


def test_raises_non_fitted_error_when_error_during_fit(transformer_class):
    # fit fails on the constant column after computing mean_ and range_; the
    # shared check builds pandas frames, so this test is pandas-only
    df = pd.DataFrame(
        {
            "var1": [1.0, 2.0, 3.0],
            "var2": [4.0, 5.0, 3.0],
            "var3": [7.0, 7.0, 7.0],
        }
    )
    transformer = make_transformer(transformer_class)
    check_raises_non_fitted_error_when_fit_fails(transformer, df)


def test_check_return_empty(transformer_class):
    # check_return_empty itself is pandas-only (builds pd.DataFrame internally).
    transformer = make_transformer(transformer_class)
    if transformer_class is MeanNormalizationScaler:
        with pytest.warns(FutureWarning, match=re.escape(DEPRECATION_WARNING)):
            check_return_empty(transformer)
    else:
        check_return_empty(transformer)


def test_integer_column_names(transformer_class):
    # integer column names are pandas-only
    X = pd.DataFrame({0: [1.0, 2.0, 3.0], 1: [10.0, 20.0, 40.0], 2: ["a", "b", "c"]})
    transformer = make_transformer(transformer_class)
    Xt = transformer.fit_transform(X)
    expected = make_transformer(transformer_class).fit_transform(X.rename(columns=str))

    pd.testing.assert_frame_equal(Xt, expected.set_axis([0, 1, 2], axis=1))
    pd.testing.assert_frame_equal(transformer.inverse_transform(Xt), X)
