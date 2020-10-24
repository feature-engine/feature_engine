# Authors: Soledad Galli <solegalli@gprotonmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd

from feature_engine.outliers import OutlierTrimmer


def test_gaussian_right_tail_capping_when_fold_is_1(df_normal_dist):
    # test case 1: mean and std, right tail
    transformer = OutlierTrimmer(capping_method="gaussian", tail="right", fold=1)
    X = transformer.fit_transform(df_normal_dist)

    # expected output
    df_transf = df_normal_dist.copy()
    outliers = np.where(df_transf["var"] > 0.10727677848029868, True, False)
    df_transf = df_transf.loc[~outliers]

    # test transform output
    pd.testing.assert_frame_equal(X, df_transf)
    assert len(X) == 83


def test_gaussian_both_tails_capping_with_fold_2(df_normal_dist):
    # test case 2: mean and std, both tails, different fold value
    transformer = OutlierTrimmer(capping_method="gaussian", tail="both", fold=2)
    X = transformer.fit_transform(df_normal_dist)
    assert len(X) == 96


def test_iqr_left_tail_capping_with_fold_2(df_normal_dist):
    # test case 3: IQR, left tail, fold 2
    transformer = OutlierTrimmer(capping_method="iqr", tail="left", fold=0.8)
    X = transformer.fit_transform(df_normal_dist)

    df_transf = df_normal_dist.copy()
    outliers = np.where(df_transf["var"] < -0.17486039103044, True, False)
    df_transf = df_transf.loc[~outliers]

    pd.testing.assert_frame_equal(X, df_transf)
    assert len(X) == 98


def test_transformer_ignores_na_in_df(df_na):
    # test case 4: dataset contains na, and transformer is asked to ignore
    transformer = OutlierTrimmer(
        capping_method="gaussian",
        tail="right",
        fold=1,
        variables=["Age"],
        missing_values="ignore",
    )
    X = transformer.fit_transform(df_na)

    df_transf = df_na.copy()
    outliers = np.where(df_transf["Age"] > 38.79255087111844, True, False)
    df_transf = df_transf.loc[~outliers]

    pd.testing.assert_frame_equal(X, df_transf)
    assert len(X) == 6
