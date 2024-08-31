import numpy as np
import pandas as pd
import pytest

from feature_engine.outliers import ArbitraryOutlierCapper


def test_right_end_capping(df_normal_dist):
    # test case 1: right end capping
    transformer = ArbitraryOutlierCapper(
        max_capping_dict={"var": 0.10727677848029868}, min_capping_dict=None
    )
    X = transformer.fit_transform(df_normal_dist)

    # expected output
    df_transf = df_normal_dist.copy()
    df_transf["var"] = np.where(
        df_transf["var"] > 0.10727677848029868, 0.10727677848029868, df_transf["var"]
    )

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
    pd.testing.assert_frame_equal(X, df_transf)
    assert np.round(X["var"].max(), 3) <= np.round(0.10727677848029868, 3)
    assert np.round(df_normal_dist["var"].max(), 3) > np.round(0.10727677848029868, 3)


def test_both_ends_capping(df_normal_dist):
    # test case 2: both tails
    transformer = ArbitraryOutlierCapper(
        max_capping_dict={"var": 0.20857275540714884},
        min_capping_dict={"var": -0.19661115230025186},
    )
    X = transformer.fit_transform(df_normal_dist)

    # expected output
    df_transf = df_normal_dist.copy()
    df_transf["var"] = np.where(
        df_transf["var"] > 0.20857275540714884, 0.20857275540714884, df_transf["var"]
    )
    df_transf["var"] = np.where(
        df_transf["var"] < -0.19661115230025186, -0.19661115230025186, df_transf["var"]
    )

    # test fit params
    assert np.round(transformer.right_tail_caps_["var"], 3) == np.round(
        0.20857275540714884, 3
    )
    assert np.round(transformer.left_tail_caps_["var"], 3) == np.round(
        -0.19661115230025186, 3
    )
    # test transform output
    pd.testing.assert_frame_equal(X, df_transf)
    assert np.round(X["var"].max(), 3) <= np.round(0.20857275540714884, 3)
    assert np.round(X["var"].min(), 3) >= np.round(-0.19661115230025186, 3)
    assert np.round(df_normal_dist["var"].max(), 3) > np.round(0.20857275540714884, 3)
    assert np.round(df_normal_dist["var"].min(), 3) < np.round(-0.19661115230025186, 3)


def test_left_tail_capping(df_normal_dist):
    # test case 3: left tail
    transformer = ArbitraryOutlierCapper(
        max_capping_dict=None, min_capping_dict={"var": -0.17486039103044}
    )
    X = transformer.fit_transform(df_normal_dist)

    # expected output
    df_transf = df_normal_dist.copy()
    df_transf["var"] = np.where(
        df_transf["var"] < -0.17486039103044, -0.17486039103044, df_transf["var"]
    )

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
    pd.testing.assert_frame_equal(X, df_transf)
    assert np.round(X["var"].min(), 3) >= np.round(-0.17486039103044, 3)
    assert np.round(df_normal_dist["var"].min(), 3) < np.round(-0.17486039103044, 3)


def test_ignores_na_in_input_df(df_na):
    # test case 4: dataset contains na and transformer is asked to ignore them
    transformer = ArbitraryOutlierCapper(
        max_capping_dict=None, min_capping_dict={"Age": 20}, missing_values="ignore"
    )
    X = transformer.fit_transform(df_na)

    # expected output
    df_transf = df_na.copy()
    df_transf["Age"] = np.where(df_transf["Age"] < 20, 20, df_transf["Age"])

    # test fit params
    assert transformer.max_capping_dict is None
    assert transformer.min_capping_dict == {"Age": 20}
    assert transformer.n_features_in_ == 6
    # test transform output
    pd.testing.assert_frame_equal(X, df_transf)
    assert X["Age"].min() >= 20
    assert df_na["Age"].min() < 20


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


def test_fit_and_transform_raise_error_if_df_contains_na(df_normal_dist):
    df_na = df_normal_dist.copy()
    df_na.loc[1, "var"] = np.nan

    # test case 5: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = ArbitraryOutlierCapper(
            min_capping_dict={"var": -0.17486039103044}
        )
        transformer.fit(df_na)

    # test case 6: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = ArbitraryOutlierCapper(
            min_capping_dict={"var": -0.17486039103044}
        )
        transformer.fit(df_normal_dist)
        transformer.transform(df_na)


@pytest.mark.parametrize(
    "missing_values",
    ["HOLA", 1, True, {"key1": "value1", "key2": "value2", "key3": "value3"}],
)
def test_error_if_missing_values_wrong_type(missing_values):
    msg = "missing_values takes only values 'raise' or 'ignore'"
    with pytest.raises(ValueError) as record:
        ArbitraryOutlierCapper(
            min_capping_dict={"var": -0.17486039103044}, missing_values="missing_values"
        )
    # check that error message matches
    assert str(record.value) == msg
