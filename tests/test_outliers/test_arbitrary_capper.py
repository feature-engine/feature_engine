import datetime
import re

import numpy as np
import pytest

from feature_engine.outliers import ArbitraryOutlierCapper
from tests.backend_helpers import frame_to_dict

DATA = {"var": np.random.RandomState(0).normal(0, 0.1, 20).tolist()}

DATA_NA = {
    "Name": ["tom", "nick", "krish", "jack", "tom", "eric"],
    "City": ["London", "Manchester", "Liverpool", "Bristol", "Manchester", "Liverpool"],
    "Age": [20.0, 21.0, 19.0, 18.0, None, 41.0],
    "Marks": [0.9, 0.8, 0.7, 0.6, 0.5, 0.6],
    "dob": [datetime.datetime(2020, 2, 24, 0, i) for i in range(6)],
}

MSG_NA = (
    "Some of the variables in the dataset contain NaN. Check and "
    "remove those before using this transformer."
)


def test_right_end_capping(make_df):
    transformer = ArbitraryOutlierCapper(
        max_capping_dict={"var": 0.10727677848029868}, min_capping_dict=None
    )
    Xt = transformer.fit_transform(make_df(DATA))

    # expected output
    expected = [min(v, 0.10727677848029868) for v in DATA["var"]]

    # test init params
    assert np.round(transformer.max_capping_dict["var"], 3) == np.round(
        0.10727677848029868, 3
    )
    assert transformer.min_capping_dict is None
    assert transformer.variables_ == ["var"]
    # test fit attrs
    assert np.round(transformer.right_tail_caps_["var"], 3) == np.round(
        0.10727677848029868, 3
    )
    assert transformer.left_tail_caps_ == {}
    assert transformer.n_features_in_ == 1
    # test transform output
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"var": pytest.approx(expected)}
    assert max(frame_to_dict(Xt)["var"]) <= 0.10727677848029868 + 1e-8


def test_both_ends_capping(make_df):
    transformer = ArbitraryOutlierCapper(
        max_capping_dict={"var": 0.20857275540714884},
        min_capping_dict={"var": -0.19661115230025186},
    )
    Xt = transformer.fit_transform(make_df(DATA))

    # expected output
    expected = [
        min(max(v, -0.19661115230025186), 0.20857275540714884) for v in DATA["var"]
    ]

    # test fit params
    assert np.round(transformer.right_tail_caps_["var"], 3) == np.round(
        0.20857275540714884, 3
    )
    assert np.round(transformer.left_tail_caps_["var"], 3) == np.round(
        -0.19661115230025186, 3
    )
    # test transform output
    assert isinstance(Xt, make_df)
    result = frame_to_dict(Xt)
    assert result == {"var": pytest.approx(expected)}
    assert max(result["var"]) <= 0.20857275540714884 + 1e-8
    assert min(result["var"]) >= -0.19661115230025186 - 1e-8


def test_left_tail_capping(make_df):
    transformer = ArbitraryOutlierCapper(
        max_capping_dict=None, min_capping_dict={"var": -0.17486039103044}
    )
    Xt = transformer.fit_transform(make_df(DATA))

    # expected output
    expected = [max(v, -0.17486039103044) for v in DATA["var"]]

    # test init param
    assert transformer.max_capping_dict is None
    assert np.round(transformer.min_capping_dict["var"], 3) == np.round(
        -0.17486039103044, 3
    )
    # test fit attr
    assert transformer.right_tail_caps_ == {}
    assert np.round(transformer.left_tail_caps_["var"], 3) == np.round(
        -0.17486039103044, 3
    )
    # test transform output
    assert isinstance(Xt, make_df)
    result = frame_to_dict(Xt)
    assert result == {"var": pytest.approx(expected)}
    assert min(result["var"]) >= -0.17486039103044 - 1e-8


def test_ignores_na_in_input_df(make_df):
    transformer = ArbitraryOutlierCapper(
        max_capping_dict=None, min_capping_dict={"Age": 20}, missing_values="ignore"
    )
    Xt = transformer.fit_transform(make_df(DATA_NA))

    # expected output
    expected = [None if v is None else max(v, 20) for v in DATA_NA["Age"]]

    # test fit params
    assert transformer.max_capping_dict is None
    assert transformer.min_capping_dict == {"Age": 20}
    assert transformer.n_features_in_ == 5
    # test transform output
    assert isinstance(Xt, make_df)
    result = frame_to_dict(Xt)
    assert result["Age"] == expected
    assert min(v for v in result["Age"] if v is not None) >= 20


def test_error_if_max_capping_dict_wrong_input():
    with pytest.raises(TypeError):
        ArbitraryOutlierCapper(max_capping_dict="other")
    with pytest.raises(ValueError):
        ArbitraryOutlierCapper(max_capping_dict={"a": "a"})


def test_error_if_min_capping_dict_wrong_input():
    with pytest.raises(TypeError):
        ArbitraryOutlierCapper(min_capping_dict="other")
    with pytest.raises(ValueError):
        ArbitraryOutlierCapper(min_capping_dict={"a": "a"})


def test_error_if_both_capping_dicts_are_none():
    with pytest.raises(ValueError):
        ArbitraryOutlierCapper(min_capping_dict=None, max_capping_dict=None)


def test_error_if_missing_values_not_bool():
    with pytest.raises(ValueError):
        ArbitraryOutlierCapper(missing_values="other")


def test_fit_and_transform_raise_error_if_df_contains_na(make_df):
    data_na = {"var": list(DATA["var"])}
    data_na["var"][1] = None

    # test case: when dataset contains na, fit method
    transformer = ArbitraryOutlierCapper(min_capping_dict={"var": -0.17486039103044})
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        transformer.fit(make_df(data_na))

    # test case: when dataset contains na, transform method
    transformer = ArbitraryOutlierCapper(min_capping_dict={"var": -0.17486039103044})
    transformer.fit(make_df(DATA))
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        transformer.transform(make_df(data_na))


@pytest.mark.parametrize(
    "missing_values",
    ["HOLA", 1, True, {"key1": "value1", "key2": "value2", "key3": "value3"}],
)
def test_error_if_missing_values_wrong_type(missing_values):
    msg = "missing_values takes only values 'raise' or 'ignore'"
    with pytest.raises(ValueError, match=re.escape(msg)):
        ArbitraryOutlierCapper(
            min_capping_dict={"var": -0.17486039103044}, missing_values=missing_values
        )
