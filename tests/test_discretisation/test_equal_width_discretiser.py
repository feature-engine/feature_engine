import re

import narwhals as nw
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.discretisation import EqualWidthDiscretiser
from tests.backend_helpers import frame_to_dict

MSG_NA = (
    "Some of the variables in the dataset contain NaN. Check and "
    "remove those before using this transformer."
)


# init parameters
@pytest.mark.parametrize("bins", ["other", 1.5, None, [10]])
def test_error_when_bins_not_number(bins):
    msg = f"bins must be an integer. Got {bins} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        EqualWidthDiscretiser(bins=bins)


@pytest.mark.parametrize("return_object", ["other", 1, None])
def test_error_if_return_object_not_bool(return_object):
    msg = f"return_object must be True or False. Got {return_object} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        EqualWidthDiscretiser(return_object=return_object)


@pytest.mark.parametrize(
    "bins, return_object, return_boundaries, precision",
    [(10, False, False, 3), (5, True, False, 1), (2, False, True, 7)],
)
def test_init_param_assignment(bins, return_object, return_boundaries, precision):
    transformer = EqualWidthDiscretiser(
        bins=bins,
        return_object=return_object,
        return_boundaries=return_boundaries,
        precision=precision,
    )
    assert transformer.bins == bins
    assert transformer.return_object is return_object
    assert transformer.return_boundaries is return_boundaries
    assert transformer.precision == precision


# fit and transform

def _expected_bins_and_codes(values, n_bins):
    # ground truth bin edges via pandas.cut, same widening/duplicates-drop
    # rules the fit() replicates in plain numpy.
    series = pd.Series(values)
    _, bins = pd.cut(x=series, bins=n_bins, retbins=True, duplicates="drop")
    bins[0] = float("-inf")
    bins[len(bins) - 1] = float("inf")
    codes = pd.cut(series, bins=list(bins), labels=False, include_lowest=True)
    return bins, codes.tolist()


def test_automatically_find_variables_and_return_as_numeric(
    make_df, data_normal_dist
):
    transformer = EqualWidthDiscretiser(bins=10, variables=None, return_object=False)
    X = transformer.fit_transform(make_df(data_normal_dist))

    bins, expected_codes = _expected_bins_and_codes(data_normal_dist["var"], 10)

    # fit params
    assert transformer.variables_ == ["var"]
    assert transformer.n_features_in_ == 1
    assert np.allclose(transformer.binner_dict_["var"], bins)
    # transform params: same bin codes on both backends
    assert isinstance(X, make_df)
    assert frame_to_dict(X)["var"] == expected_codes


def test_automatically_find_variables_and_return_as_object(make_df, data_normal_dist):
    transformer = EqualWidthDiscretiser(bins=10, variables=None, return_object=True)
    X = transformer.fit_transform(make_df(data_normal_dist))
    assert isinstance(X, make_df)
    assert nw.from_native(X, eager_only=True).schema["var"] == nw.Object


def test_constant_variable_produces_single_bin(make_df):
    # a constant variable still fits, with every value in the same bin, as with
    # pandas.cut(bins=10)
    data = {"var": [5.0] * 10}
    transformer = EqualWidthDiscretiser(bins=10)
    X = transformer.fit_transform(make_df(data))

    _, expected_codes = _expected_bins_and_codes(data["var"], 10)

    assert transformer.binner_dict_["var"][0] == float("-inf")
    assert transformer.binner_dict_["var"][-1] == float("inf")
    assert isinstance(X, make_df)
    assert frame_to_dict(X)["var"] == expected_codes


def test_error_if_input_df_contains_na_in_fit(make_df, data_na):
    transformer = EqualWidthDiscretiser()
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        transformer.fit(make_df(data_na))


def test_error_if_input_df_contains_na_in_transform(make_df, data_vartypes, data_na):
    transform_data = make_df(
        {k: data_na[k] for k in ["Name", "City", "Age", "Marks", "dob"]}
    )
    transformer = EqualWidthDiscretiser()
    transformer.fit(make_df(data_vartypes))
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        transformer.transform(transform_data)


def test_non_fitted_error(make_df, data_vartypes):
    transformer = EqualWidthDiscretiser()
    msg = (
        "This EqualWidthDiscretiser instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=re.escape(msg)):
        transformer.transform(make_df(data_vartypes))
