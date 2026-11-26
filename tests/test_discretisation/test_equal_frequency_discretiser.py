import re
from collections import Counter

import narwhals as nw
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.discretisation import EqualFrequencyDiscretiser
from tests.backend_helpers import frame_to_dict

MSG_NA = (
    "Some of the variables in the dataset contain NaN. Check and "
    "remove those before using this transformer."
)


# init parameters
@pytest.mark.parametrize("q", ["other", 1.5, None, [10]])
def test_error_when_q_not_number(q):
    msg = f"q must be an integer. Got {q} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        EqualFrequencyDiscretiser(q=q)


@pytest.mark.parametrize("return_object", ["other", 1, None])
def test_error_if_return_object_not_bool(return_object):
    msg = f"return_object must be True or False. Got {return_object} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        EqualFrequencyDiscretiser(return_object=return_object)


@pytest.mark.parametrize(
    "q, return_object, return_boundaries, precision",
    [(10, False, False, 3), (5, True, False, 1), (2, False, True, 7)],
)
def test_init_param_assignment(q, return_object, return_boundaries, precision):
    transformer = EqualFrequencyDiscretiser(
        q=q,
        return_object=return_object,
        return_boundaries=return_boundaries,
        precision=precision,
    )
    assert transformer.q == q
    assert transformer.return_object is return_object
    assert transformer.return_boundaries is return_boundaries
    assert transformer.precision == precision


# fit and transform

def test_automatically_find_variables_and_return_as_numeric(
    make_df, data_normal_dist
):
    # test case 1: automatically select variables, return_object=False
    transformer = EqualFrequencyDiscretiser(q=10, variables=None, return_object=False)
    X = transformer.fit_transform(make_df(data_normal_dist))

    # output expected for fit attr, computed via pandas.qcut (verified bit-exact
    # against the transformer's own numpy-based bin edges on both backends)
    _, bins = pd.qcut(
        x=pd.Series(data_normal_dist["var"]), q=10, retbins=True, duplicates="drop"
    )
    bins = list(bins)
    bins[0] = float("-inf")
    bins[len(bins) - 1] = float("inf")

    # test fit attr
    assert transformer.variables_ == ["var"]
    assert transformer.n_features_in_ == 1
    assert transformer.binner_dict_["var"] == bins
    # test transform output
    assert isinstance(X, make_df)
    values = frame_to_dict(X)["var"]
    assert set(values) == set(range(10))
    # in equal frequency discretisation, all intervals get same proportion of values
    assert len(set(Counter(values).values())) == 1


def test_automatically_find_variables_and_return_as_object(make_df, data_normal_dist):
    # test case 2: return variables cast as object
    transformer = EqualFrequencyDiscretiser(q=10, variables=None, return_object=True)
    X = transformer.fit_transform(make_df(data_normal_dist))
    assert isinstance(X, make_df)
    assert nw.from_native(X, eager_only=True).schema["var"] == nw.Object


def test_error_if_input_df_contains_na_in_fit(make_df, data_na):
    # test case 3: when dataset contains na, fit method
    transformer = EqualFrequencyDiscretiser()
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        transformer.fit(make_df(data_na))


def test_error_if_input_df_contains_na_in_transform(make_df, data_vartypes, data_na):
    # test case 4: when dataset contains na, transform method
    transform_data = make_df(
        {k: data_na[k] for k in ["Name", "City", "Age", "Marks", "dob"]}
    )
    transformer = EqualFrequencyDiscretiser()
    transformer.fit(make_df(data_vartypes))
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        transformer.transform(transform_data)


def test_non_fitted_error(make_df, data_vartypes):
    transformer = EqualFrequencyDiscretiser()
    msg = (
        "This EqualFrequencyDiscretiser instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=re.escape(msg)):
        transformer.transform(make_df(data_vartypes))
