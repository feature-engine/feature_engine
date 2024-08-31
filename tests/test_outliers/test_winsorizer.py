import math

import numpy as np
import pandas as pd
import pytest

from feature_engine.outliers import Winsorizer


def test_gaussian_capping_right_tail_with_fold_1(df_normal_dist):
    # test case 1: mean and std, right tail
    transformer = Winsorizer(capping_method="gaussian", tail="right", fold=1)
    X = transformer.fit_transform(df_normal_dist)

    # expected output
    df_transf = df_normal_dist.copy()
    df_transf["var"] = df_transf["var"].clip(upper=0.1067690260251065)

    # test init params
    assert transformer.capping_method == "gaussian"
    assert transformer.tail == "right"
    assert transformer.fold == 1
    # test fit attr
    assert math.isclose(transformer.right_tail_caps_["var"], 0.1067690260251065)
    assert transformer.left_tail_caps_ == {}
    assert transformer.n_features_in_ == 1
    # test transform outputs
    pd.testing.assert_frame_equal(X, df_transf)
    assert math.isclose(X["var"].max(), 0.10676902602510658)
    assert math.isclose(df_transf["var"].max(), 0.1067690260251065)


def test_gaussian_capping_both_tails_with_fold_2(df_normal_dist):
    # test case 2: mean and std, both tails, different fold value
    transformer = Winsorizer(capping_method="gaussian", tail="both", fold=2)
    X = transformer.fit_transform(df_normal_dist)

    # expected output
    df_transf = df_normal_dist.copy()
    df_transf["var"] = df_transf["var"].clip(-0.1955956473898675, 0.2075572504967645)

    # test fit params
    assert math.isclose(transformer.right_tail_caps_["var"], 0.2075572504967645)
    assert math.isclose(transformer.left_tail_caps_["var"], -0.1955956473898675)
    # test transform output
    pd.testing.assert_frame_equal(X, df_transf)
    assert math.isclose(X["var"].max(), 0.2075572504967645)
    assert math.isclose(X["var"].min(), -0.1955956473898675)
    assert math.isclose(df_transf["var"].max(), 0.2075572504967645)
    assert math.isclose(df_transf["var"].min(), -0.1955956473898675)


def test_iqr_capping_both_tails_with_fold_1(df_normal_dist):
    # test case 3: IQR, both tails, fold 1
    transformer = Winsorizer(capping_method="iqr", tail="both", fold=1)
    X = transformer.fit_transform(df_normal_dist)

    # expected output
    df_transf = df_normal_dist.copy()
    df_transf["var"] = df_transf["var"].clip(-0.20247907173293223, 0.21180113880445128)

    # test fit params
    assert math.isclose(transformer.right_tail_caps_["var"], 0.21180113880445128)
    assert math.isclose(transformer.left_tail_caps_["var"], -0.20247907173293223)
    # test transform output
    pd.testing.assert_frame_equal(X, df_transf)
    assert math.isclose(X["var"].max(), 0.21180113880445128)
    assert math.isclose(X["var"].min(), -0.20247907173293223)
    assert math.isclose(df_transf["var"].max(), 0.21180113880445128)
    assert math.isclose(df_transf["var"].min(), -0.20247907173293223)


def test_iqr_capping_left_tail_with_fold_2(df_normal_dist):
    # test case 4: IQR, left tail, fold 2
    transformer = Winsorizer(capping_method="iqr", tail="left", fold=0.8)
    X = transformer.fit_transform(df_normal_dist)

    # expected output
    df_transf = df_normal_dist.copy()
    df_transf["var"] = df_transf["var"].clip(lower=-0.17486039103044)

    # test fit params
    assert transformer.right_tail_caps_ == {}
    assert math.isclose(transformer.left_tail_caps_["var"], -0.17486039103044)
    # test transform output
    pd.testing.assert_frame_equal(X, df_transf)
    assert math.isclose(X["var"].min(), -0.17486039103044)
    assert math.isclose(df_transf["var"].min(), -0.17486039103044)


def test_quantile_capping_both_tails_with_fold_10_percent(df_normal_dist):
    # test case 5: quantiles, both tails, fold 10%
    transformer = Winsorizer(capping_method="quantiles", tail="both", fold=0.1)
    X = transformer.fit_transform(df_normal_dist)

    # expected output
    df_transf = df_normal_dist.copy()
    df_transf["var"] = df_transf["var"].clip(-0.12366227743232801, 0.14712481122898166)

    # test fit params
    assert math.isclose(transformer.right_tail_caps_["var"], 0.14712481122898166)
    assert math.isclose(transformer.left_tail_caps_["var"], -0.12366227743232801)
    # test transform output
    pd.testing.assert_frame_equal(X, df_transf)
    assert math.isclose(X["var"].max(), 0.14712481122898166)
    assert math.isclose(X["var"].min(), -0.12366227743232801)
    assert math.isclose(df_transf["var"].max(), 0.14712481122898166)
    assert math.isclose(df_transf["var"].min(), -0.12366227743232801)


def test_quantile_capping_right_tail_with_fold_15_percent(df_normal_dist):
    # test case 6: quantiles, right tail, fold 15%
    transformer = Winsorizer(capping_method="quantiles", tail="right", fold=0.15)
    X = transformer.fit_transform(df_normal_dist)

    # expected output
    df_transf = df_normal_dist.copy()
    df_transf["var"] = df_transf["var"].clip(upper=0.11823196128033647)

    # test fit params
    assert math.isclose(transformer.right_tail_caps_["var"], 0.11823196128033647)
    assert transformer.left_tail_caps_ == {}
    # test transform output
    pd.testing.assert_frame_equal(X, df_transf)
    assert math.isclose(X["var"].max(), 0.11823196128033647)
    assert math.isclose(df_transf["var"].max(), 0.11823196128033647)


@pytest.mark.parametrize(
    "strings,expected",
    [("gaussian", 3), ("iqr", 1.5), ("mad", 3.29), ("quantiles", 0.05)],
)
def test_auto_fold_default_value(strings, expected, df_normal_dist):
    transformer = Winsorizer(capping_method=strings, fold="auto")
    transformer.fit(df_normal_dist)
    assert transformer.fold_ == expected


def test_mad_capping_right_tail_with_fold_1(df_normal_dist):
    # test case 1: median and mad, right tail
    transformer = Winsorizer(capping_method="mad", tail="right", fold=1)
    X = transformer.fit_transform(df_normal_dist)

    # expected output
    df_transf = df_normal_dist.copy()
    df_transf["var"] = df_transf["var"].clip(upper=0.10995521088494983)

    # test init params
    assert transformer.capping_method == "mad"
    assert transformer.tail == "right"
    assert transformer.fold == 1
    # test fit attr
    assert math.isclose(transformer.right_tail_caps_["var"], 0.10995521088494983)
    assert transformer.left_tail_caps_ == {}
    assert transformer.n_features_in_ == 1
    # test transform outputs
    pd.testing.assert_frame_equal(X, df_transf)
    assert math.isclose(X["var"].max(), 0.10995521088494983)
    assert math.isclose(df_transf["var"].max(), 0.10995521088494983)


def test_mad_capping_both_tails_with_fold_2(df_normal_dist):
    # test case 2: mean and std, both tails, different fold value
    transformer = Winsorizer(capping_method="mad", tail="both", fold=2)
    X = transformer.fit_transform(df_normal_dist)

    # expected output
    df_transf = df_normal_dist.copy()
    df_transf["var"] = df_transf["var"].clip(-0.1916815859385002, 0.21050080982609987)

    # test fit params
    assert math.isclose(transformer.right_tail_caps_["var"], 0.21050080982609987)
    assert math.isclose(transformer.left_tail_caps_["var"], -0.1916815859385002)
    # test transform output
    pd.testing.assert_frame_equal(X, df_transf)
    assert math.isclose(X["var"].max(), 0.21050080982609987)
    assert math.isclose(X["var"].min(), -0.1916815859385002)
    assert math.isclose(df_transf["var"].max(), 0.21050080982609987)
    assert math.isclose(df_transf["var"].min(), -0.1916815859385002)


def test_indicators_are_added(df_normal_dist):
    transformer = Winsorizer(
        tail="both", capping_method="quantiles", fold=0.1, add_indicators=True
    )
    X = transformer.fit_transform(df_normal_dist)
    # test that the number of output variables is correct
    assert X.shape[1] == 3 * df_normal_dist.shape[1]
    assert np.all(X.iloc[:, df_normal_dist.shape[1]:].sum(axis=0) > 0)

    transformer = Winsorizer(
        tail="left", capping_method="quantiles", fold=0.1, add_indicators=True
    )
    X = transformer.fit_transform(df_normal_dist)
    assert X.shape[1] == 2 * df_normal_dist.shape[1]
    assert np.all(X.iloc[:, df_normal_dist.shape[1]:].sum(axis=0) > 0)

    transformer = Winsorizer(
        tail="right", capping_method="quantiles", fold=0.1, add_indicators=True
    )
    X = transformer.fit_transform(df_normal_dist)
    assert X.shape[1] == 2 * df_normal_dist.shape[1]
    assert np.all(X.iloc[:, df_normal_dist.shape[1]:].sum(axis=0) > 0)


def test_indicators_filter_variables(df_vartypes):
    transformer = Winsorizer(
        variables=["Age", "Marks"],
        tail="both",
        capping_method="quantiles",
        fold=0.1,
        add_indicators=True,
    )
    X = transformer.fit_transform(df_vartypes)
    assert X.shape[1] == df_vartypes.shape[1] + 4

    transformer.set_params(tail="left")
    X = transformer.fit_transform(df_vartypes)
    assert X.shape[1] == df_vartypes.shape[1] + 2

    transformer.set_params(tail="right")
    X = transformer.fit_transform(df_vartypes)
    assert X.shape[1] == df_vartypes.shape[1] + 2


def test_indicators_are_correct():
    transformer = Winsorizer(
        tail="left", capping_method="quantiles", fold=0.1, add_indicators=True
    )
    df = pd.DataFrame({"col": np.arange(100).astype(np.float64)})
    df_out = transformer.fit_transform(df)
    expected_ind = np.r_[np.repeat(True, 10), np.repeat(False, 90)].astype(np.float64)
    pd.testing.assert_frame_equal(
        df_out.drop("col", axis=1), df.assign(col_left=expected_ind).drop("col", axis=1)
    )

    transformer.set_params(tail="right")
    df_out = transformer.fit_transform(df)
    expected_ind = np.r_[np.repeat(False, 90), np.repeat(True, 10)].astype(np.float64)
    pd.testing.assert_frame_equal(
        df_out.drop("col", axis=1),
        df.assign(col_right=expected_ind).drop("col", axis=1),
    )

    transformer.set_params(tail="both")
    df_out = transformer.fit_transform(df)
    expected_ind_left = np.r_[np.repeat(True, 10), np.repeat(False, 90)].astype(
        np.float64
    )
    expected_ind_right = np.r_[np.repeat(False, 90), np.repeat(True, 10)].astype(
        np.float64
    )
    pd.testing.assert_frame_equal(
        df_out.drop("col", axis=1),
        df.assign(col_left=expected_ind_left, col_right=expected_ind_right).drop(
            "col", axis=1
        ),
    )


def test_transformer_ignores_na_in_df(df_na):
    # test case 7: dataset contains na and transformer is asked to ignore them
    transformer = Winsorizer(
        capping_method="gaussian",
        tail="right",
        fold=1,
        variables=["Age", "Marks"],
        missing_values="ignore",
    )
    X = transformer.fit_transform(df_na)

    # expected output
    df_transf = df_na.copy()
    df_transf["Age"] = df_transf["Age"].clip(upper=38.04494616731882)
    df_transf["Marks"] = df_transf["Marks"].clip(upper=0.8784116651786605)

    # test fit params
    assert math.isclose(transformer.right_tail_caps_["Age"], 38.04494616731882)
    assert math.isclose(transformer.right_tail_caps_["Marks"], 0.8784116651786605)
    assert transformer.left_tail_caps_ == {}
    assert transformer.n_features_in_ == 6
    # test transform output
    pd.testing.assert_frame_equal(X, df_transf)
    assert math.isclose(X["Age"].max(), 38.04494616731882)
    assert math.isclose(X["Age"].max(), 38.04494616731882)
    assert math.isclose(X["Marks"].max(), 0.8784116651786605)
    assert math.isclose(df_transf["Marks"].max(), 0.8784116651786605)


def test_error_if_capping_method_not_permitted():
    # test error raises
    with pytest.raises(ValueError):
        Winsorizer(capping_method="other")


def test_error_if_tail_value_not_permitted():
    with pytest.raises(ValueError):
        Winsorizer(tail="other")


def test_error_if_missing_values_not_permited():
    with pytest.raises(ValueError):
        Winsorizer(missing_values="other")


def test_error_if_fold_value_not_permitted():
    with pytest.raises(ValueError):
        Winsorizer(fold=-1)


def test_error_if_capping_method_quantiles_and_fold_value_not_permitted():
    with pytest.raises(ValueError):
        Winsorizer(capping_method="quantiles", fold=0.3)


def test_error_if_add_incators_not_permitted():
    with pytest.raises(ValueError):
        Winsorizer(add_indicators=-1)
    with pytest.raises(ValueError):
        Winsorizer(add_indicators=())
    with pytest.raises(ValueError):
        Winsorizer(add_indicators=[True])


def test_fit_raises_error_if_na_in_inut_df(df_na):
    # test case 8: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = Winsorizer()
        transformer.fit(df_na)


def test_transform_raises_error_if_na_in_input_df(df_vartypes, df_na):
    # test case 9: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = Winsorizer()
        transformer.fit(df_vartypes)
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])


def test_get_feature_names_out(df_na):
    original_features = df_na.columns.to_list()
    input_features = ["Age", "Marks"]

    # when indicators is false, we've got the generic check.
    # We need to test only when true
    tr = Winsorizer(tail="left", add_indicators=True, missing_values="ignore")
    tr.fit(df_na)

    out = [f + "_left" for f in input_features]
    assert tr.get_feature_names_out() == original_features + out
    assert tr.get_feature_names_out(original_features) == original_features + out

    tr = Winsorizer(tail="right", add_indicators=True, missing_values="ignore")
    tr.fit(df_na)

    out = [f + "_right" for f in input_features]
    assert tr.get_feature_names_out() == original_features + out
    assert tr.get_feature_names_out(original_features) == original_features + out

    tr = Winsorizer(tail="both", add_indicators=True, missing_values="ignore")
    tr.fit(df_na)

    out = ["Age_left", "Age_right", "Marks_left", "Marks_right"]
    assert tr.get_feature_names_out() == original_features + out
    assert tr.get_feature_names_out(original_features) == original_features + out


def test_low_variation(df_normal_dist):
    transformer = Winsorizer(capping_method="mad")
    with pytest.raises(ValueError):
        transformer.fit(df_normal_dist // 10)
