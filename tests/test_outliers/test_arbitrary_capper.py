import re

import numpy as np
import pytest

from feature_engine.outliers import ArbitraryOutlierCapper
from tests.backend_helpers import frame_to_dict

MSG_NA = (
    "Some of the variables in the dataset contain NaN. Check and "
    "remove those before using this transformer."
)


# init parameters
@pytest.mark.parametrize("param", ["max_capping_dict", "min_capping_dict"])
@pytest.mark.parametrize("value", ["other", 1, ["var"], ("var", 1)])
def test_error_if_capping_dict_not_dict(param, value):
    msg = f"The parameter can only take a dictionary or None. Got {value} instead."
    with pytest.raises(TypeError, match=re.escape(msg)):
        ArbitraryOutlierCapper(**{param: value})


@pytest.mark.parametrize("param", ["max_capping_dict", "min_capping_dict"])
@pytest.mark.parametrize("value", [{"var": "a"}, {"var": None}, {"a": 1, "b": [2]}])
def test_error_if_capping_dict_values_not_numerical(param, value):
    msg = (
        "All values in the dictionary must be integer or float. "
        f"Got {value} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        ArbitraryOutlierCapper(**{param: value})


@pytest.mark.parametrize(
    "max_capping_dict, min_capping_dict",
    [(None, None), ({}, None), (None, {}), ({}, {})],
)
def test_error_if_no_capping_values(max_capping_dict, min_capping_dict):
    msg = "Please provide at least 1 dictionary with the capping values."
    with pytest.raises(ValueError, match=re.escape(msg)):
        ArbitraryOutlierCapper(
            max_capping_dict=max_capping_dict, min_capping_dict=min_capping_dict
        )


@pytest.mark.parametrize(
    "missing_values", ["HOLA", "Raise", 1, True, None, ["raise"], {"key": "raise"}]
)
def test_error_if_missing_values_not_permitted(missing_values):
    msg = (
        "missing_values must be 'raise' or 'ignore'. "
        f"Got {missing_values} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        ArbitraryOutlierCapper(
            min_capping_dict={"var": -0.15}, missing_values=missing_values
        )


@pytest.mark.parametrize(
    "max_capping_dict, min_capping_dict, missing_values",
    [
        ({"var": 0.1}, None, "raise"),
        (None, {"var": -0.15}, "ignore"),
        ({"var": 0.1}, {"var": -0.15, "other": 2}, "raise"),
        ({"var": 1}, {}, "ignore"),
    ],
)
def test_init_param_assignment(max_capping_dict, min_capping_dict, missing_values):
    transformer = ArbitraryOutlierCapper(
        max_capping_dict=max_capping_dict,
        min_capping_dict=min_capping_dict,
        missing_values=missing_values,
    )
    assert transformer.max_capping_dict == max_capping_dict
    assert transformer.min_capping_dict == min_capping_dict
    assert transformer.missing_values == missing_values


# fit and transform
@pytest.mark.parametrize(
    "max_capping_dict, min_capping_dict",
    [({"var": 0.1}, None), (None, {"var": -0.15}), ({"var": 0.1}, {"var": -0.15})],
)
def test_capping(make_df, data_normal_dist, max_capping_dict, min_capping_dict):
    transformer = ArbitraryOutlierCapper(
        max_capping_dict=max_capping_dict, min_capping_dict=min_capping_dict
    )
    Xt = transformer.fit_transform(make_df(data_normal_dist))

    upper = np.inf if max_capping_dict is None else max_capping_dict["var"]
    lower = -np.inf if min_capping_dict is None else min_capping_dict["var"]
    expected = [min(max(v, lower), upper) for v in data_normal_dist["var"]]

    assert transformer.right_tail_caps_ == (max_capping_dict or {})
    assert transformer.left_tail_caps_ == (min_capping_dict or {})
    assert transformer.variables_ == ["var"]
    assert transformer.feature_names_in_ == ["var"]
    assert transformer.n_features_in_ == 1
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"var": pytest.approx(expected)}


def test_variables_are_taken_from_both_dicts(make_df):
    X = make_df({"a": [0, 5, 10], "b": [0, 5, 10], "c": [0, 5, 10]})
    transformer = ArbitraryOutlierCapper(
        max_capping_dict={"a": 8}, min_capping_dict={"b": 2, "a": 1}
    )
    Xt = transformer.fit_transform(X)

    assert transformer.variables_ == ["b", "a"]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"a": [1, 5, 8], "b": [2, 5, 10], "c": [0, 5, 10]}


def test_empty_dict_is_ignored(make_df):
    X = make_df({"a": [0, 5, 10], "b": [0, 5, 10]})
    transformer = ArbitraryOutlierCapper(max_capping_dict={"a": 8}, min_capping_dict={})
    Xt = transformer.fit_transform(X)

    assert transformer.variables_ == ["a"]
    assert transformer.left_tail_caps_ == {}
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"a": [0, 5, 8], "b": [0, 5, 10]}


def test_ignores_na_in_input_df(make_df, data_na):
    transformer = ArbitraryOutlierCapper(
        min_capping_dict={"Age": 21}, missing_values="ignore"
    )
    Xt = transformer.fit_transform(make_df(data_na))

    expected = [None if v is None else max(v, 21) for v in data_na["Age"]]

    assert transformer.n_features_in_ == 5
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt)["Age"] == expected


def test_fit_raises_error_if_df_contains_na(make_df, data_na):
    transformer = ArbitraryOutlierCapper(min_capping_dict={"Age": 21})
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        transformer.fit(make_df(data_na))


def test_transform_raises_error_if_df_contains_na(make_df, data_normal_dist):
    data_na = {"var": list(data_normal_dist["var"])}
    data_na["var"][1] = None
    transformer = ArbitraryOutlierCapper(min_capping_dict={"var": -0.15})
    transformer.fit(make_df(data_normal_dist))
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        transformer.transform(make_df(data_na))
