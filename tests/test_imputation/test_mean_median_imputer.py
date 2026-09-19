import re

import pytest

from feature_engine.imputation import MeanImputer, MeanMedianImputer
from tests.backend_helpers import frame_to_dict, null_count

DEPRECATION_WARNING = (
    "MeanMedianImputer was deprecated in favour of MeanImputer in version "
    "2.0.0 and will be removed in version 2.1.0. To silence this warning, "
    "use MeanImputer instead."
)


@pytest.fixture(
    params=[MeanImputer, MeanMedianImputer],
    ids=["MeanImputer", "MeanMedianImputer"],
)
def imputer_class(request):
    return request.param


def make_imputer(imputer_class, **kwargs):
    if imputer_class is MeanMedianImputer:
        with pytest.warns(FutureWarning, match=re.escape(DEPRECATION_WARNING)):
            return imputer_class(**kwargs)
    return imputer_class(**kwargs)


# init parameters
@pytest.mark.parametrize(
    "imputation_method", ["arbitrary", "mode", 1, None, ("mean",), ["median"]]
)
def test_error_with_wrong_imputation_method(imputer_class, imputation_method):
    msg = (
        "imputation_method takes only values 'median' or 'mean'. "
        f"Got {imputation_method} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        make_imputer(imputer_class, imputation_method=imputation_method)


@pytest.mark.parametrize("imputation_method", ["mean", "median"])
def test_init_param_assignment(imputer_class, imputation_method):
    imputer = make_imputer(imputer_class, imputation_method=imputation_method)
    assert imputer.imputation_method == imputation_method


# fit and transform
def test_mean_imputation_and_automatically_select_variables(
    make_df, data_na, imputer_class
):
    imputer = make_imputer(imputer_class, imputation_method="mean", variables=None)
    X_transformed = imputer.fit_transform(make_df(data_na))

    # test fit attributes
    assert imputer.variables_ == ["Age", "Marks"]
    rounded_dict = {
        key: round(value, 3) for (key, value) in imputer.imputer_dict_.items()
    }
    assert rounded_dict == {"Age": 28.714, "Marks": 0.683}
    assert imputer.n_features_in_ == 5

    # test transform output:
    # selected variables should have no NA
    # not selected variables should still have NA
    assert isinstance(X_transformed, make_df)
    assert null_count(X_transformed, "Age") == 0
    assert null_count(X_transformed, "Marks") == 0
    assert null_count(X_transformed, "Name") > 0
    assert null_count(X_transformed, "City") > 0
    result = frame_to_dict(X_transformed)
    assert result["Age"] == pytest.approx(
        [20, 21, 19, 28.714285714285715, 23, 40, 41, 37]
    )
    assert result["Marks"] == pytest.approx(
        [0.9, 0.8, 0.7, 0.6833333333333332, 0.3, 0.6833333333333332, 0.8, 0.6]
    )


def test_median_imputation_when_user_enters_single_variables(
    make_df, data_na, imputer_class
):
    imputer = make_imputer(
        imputer_class, imputation_method="median", variables=["Age"]
    )
    X_transformed = imputer.fit_transform(make_df(data_na))

    # test fit attributes
    assert imputer.n_features_in_ == 5
    assert imputer.imputer_dict_ == {"Age": 23.0}

    # test transform output
    assert isinstance(X_transformed, make_df)
    assert null_count(X_transformed, "Age") == 0
    assert frame_to_dict(X_transformed)["Age"] == [20, 21, 19, 23.0, 23, 40, 41, 37]


def test_mean_median_imputer_raises_future_warning():
    with pytest.warns(FutureWarning, match=re.escape(DEPRECATION_WARNING)):
        MeanMedianImputer()
