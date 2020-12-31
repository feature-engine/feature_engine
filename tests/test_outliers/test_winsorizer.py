import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.outliers import Winsorizer


def test_gaussian_capping_right_tail_with_fold_1(df_normal_dist):
    # test case 1: mean and std, right tail
    transformer = Winsorizer(capping_method="gaussian", tail="right", fold=1)
    X = transformer.fit_transform(df_normal_dist)

    # expected output
    df_transf = df_normal_dist.copy()
    df_transf["var"] = np.where(
        df_transf["var"] > 0.10727677848029868, 0.10727677848029868, df_transf["var"]
    )

    # test init params
    assert transformer.capping_method == "gaussian"
    assert transformer.tail == "right"
    assert transformer.fold == 1
    # test fit attr
    assert transformer.right_tail_caps_ == {"var": 0.10727677848029868}
    assert transformer.left_tail_caps_ == {}
    assert transformer.input_shape_ == (100, 1)
    # test transform outputs
    pd.testing.assert_frame_equal(X, df_transf)
    assert X["var"].max() <= 0.10727677848029868
    assert df_normal_dist["var"].max() > 0.10727677848029868


def test_gaussian_capping_both_tails_with_fold_2(df_normal_dist):
    # test case 2: mean and std, both tails, different fold value
    transformer = Winsorizer(capping_method="gaussian", tail="both", fold=2)
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
    assert transformer.right_tail_caps_ == {"var": 0.20857275540714884}
    assert transformer.left_tail_caps_ == {"var": -0.19661115230025186}
    # test transform output
    pd.testing.assert_frame_equal(X, df_transf)
    assert X["var"].max() <= 0.20857275540714884
    assert X["var"].min() >= -0.19661115230025186
    assert df_normal_dist["var"].max() > 0.20857275540714884
    assert df_normal_dist["var"].min() < -0.19661115230025186


def test_iqr_capping_both_tails_with_fold_1(df_normal_dist):
    # test case 3: IQR, both tails, fold 1
    transformer = Winsorizer(capping_method="iqr", tail="both", fold=1)
    X = transformer.fit_transform(df_normal_dist)

    # expected output
    df_transf = df_normal_dist.copy()
    df_transf["var"] = np.where(
        df_transf["var"] > 0.21180113880445128, 0.21180113880445128, df_transf["var"]
    )
    df_transf["var"] = np.where(
        df_transf["var"] < -0.20247907173293223, -0.20247907173293223, df_transf["var"]
    )

    # test fit params
    assert transformer.right_tail_caps_ == {"var": 0.21180113880445128}
    assert transformer.left_tail_caps_ == {"var": -0.20247907173293223}
    # test transform output
    pd.testing.assert_frame_equal(X, df_transf)
    assert X["var"].max() <= 0.21180113880445128
    assert X["var"].min() >= -0.20247907173293223
    assert df_normal_dist["var"].max() > 0.21180113880445128
    assert df_normal_dist["var"].min() < -0.20247907173293223


def test_iqr_capping_left_tail_with_fold_2(df_normal_dist):
    # test case 4: IQR, left tail, fold 2
    transformer = Winsorizer(capping_method="iqr", tail="left", fold=0.8)
    X = transformer.fit_transform(df_normal_dist)

    # expected output
    df_transf = df_normal_dist.copy()
    df_transf["var"] = np.where(
        df_transf["var"] < -0.17486039103044, -0.17486039103044, df_transf["var"]
    )

    # test fit params
    assert transformer.right_tail_caps_ == {}
    assert transformer.left_tail_caps_ == {"var": -0.17486039103044}
    # test transform output
    pd.testing.assert_frame_equal(X, df_transf)
    assert X["var"].min() >= -0.17486039103044
    assert df_normal_dist["var"].min() < -0.17486039103044


def test_quantile_capping_both_tails_with_fold_10_percent(df_normal_dist):
    # test case 5: quantiles, both tails, fold 10%
    transformer = Winsorizer(capping_method="quantiles", tail="both", fold=0.1)
    X = transformer.fit_transform(df_normal_dist)

    # expected output
    df_transf = df_normal_dist.copy()
    df_transf["var"] = np.where(
        df_transf["var"] > 0.14712481122898166, 0.14712481122898166, df_transf["var"]
    )
    df_transf["var"] = np.where(
        df_transf["var"] < -0.12366227743232801, -0.12366227743232801, df_transf["var"]
    )

    # test fit params
    assert transformer.right_tail_caps_ == {"var": 0.14712481122898166}
    assert transformer.left_tail_caps_ == {"var": -0.12366227743232801}
    # test transform output
    pd.testing.assert_frame_equal(X, df_transf)
    assert X["var"].max() <= 0.14712481122898166
    assert X["var"].min() >= -0.12366227743232801
    assert df_normal_dist["var"].max() > 0.14712481122898166
    assert df_normal_dist["var"].min() < -0.12366227743232801


def test_quantile_capping_both_tails_with_fold_15_percent(df_normal_dist):
    # test case 6: quantiles, right tail, fold 15%
    transformer = Winsorizer(capping_method="quantiles", tail="right", fold=0.15)
    X = transformer.fit_transform(df_normal_dist)

    # expected output
    df_transf = df_normal_dist.copy()
    df_transf["var"] = np.where(
        df_transf["var"] > 0.11823196128033647, 0.11823196128033647, df_transf["var"]
    )

    # test fit params
    assert transformer.right_tail_caps_ == {"var": 0.11823196128033647}
    assert transformer.left_tail_caps_ == {}
    # test transform output
    pd.testing.assert_frame_equal(X, df_transf)
    assert X["var"].max() <= 0.11823196128033647
    assert df_normal_dist["var"].max() > 0.11823196128033647


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
    df_transf["Age"] = np.where(
        df_transf["Age"] > 38.79255087111844, 38.79255087111844, df_transf["Age"]
    )
    df_transf["Marks"] = np.where(
        df_transf["Marks"] > 0.8970309389976613, 0.8970309389976613, df_transf["Marks"]
    )

    # test fit params
    transformer.right_tail_caps_ = {
        key: round(value, 3) for (key, value) in transformer.right_tail_caps_.items()
    }

    assert transformer.right_tail_caps_ == {
        "Age": 38.793,
        "Marks": 0.897,
    }
    assert transformer.left_tail_caps_ == {}
    assert transformer.input_shape_ == (8, 6)
    # test transform output
    pd.testing.assert_frame_equal(X, df_transf)
    assert X["Age"].max() <= 38.79255087111844
    assert df_na["Age"].max() > 38.79255087111844


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


def test_non_fitted_error(df_vartypes):
    with pytest.raises(NotFittedError):
        transformer = Winsorizer()
        transformer.transform(df_vartypes)
