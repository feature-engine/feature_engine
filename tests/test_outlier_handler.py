# Authors: Soledad Galli <solegalli@gprotonmail.com>
# License: BSD 3 clause

import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from feature_engine.outlier_removers import Winsorizer, ArbitraryOutlierCapper, OutlierTrimmer


def test_Windsorizer(dataframe_normal_dist, dataframe_na, dataframe_vartypes):
    # test case 1: mean and std, right tail
    transformer = Winsorizer(distribution='gaussian', tail='right', fold=1)
    X = transformer.fit_transform(dataframe_normal_dist)

    df_transf = dataframe_normal_dist.copy()
    df_transf['var'] = np.where(df_transf['var'] > 0.10727677848029868, 0.10727677848029868, df_transf['var'])

    # init params
    assert transformer.distribution == 'gaussian'
    assert transformer.tail == 'right'
    assert transformer.fold == 1
    # fit params
    assert transformer.right_tail_caps_ == {'var': 0.10727677848029868}
    assert transformer.left_tail_caps_ == {}
    assert transformer.input_shape_ == (100, 1)
    # transform params
    pd.testing.assert_frame_equal(X, df_transf)
    assert X['var'].max() <= 0.10727677848029868
    assert dataframe_normal_dist['var'].max() > 0.10727677848029868

    # test case 2: mean and std, both tails, different fold value
    transformer = Winsorizer(distribution='gaussian', tail='both', fold=2)
    X = transformer.fit_transform(dataframe_normal_dist)

    df_transf = dataframe_normal_dist.copy()
    df_transf['var'] = np.where(df_transf['var'] > 0.20857275540714884, 0.20857275540714884, df_transf['var'])
    df_transf['var'] = np.where(df_transf['var'] < -0.19661115230025186, -0.19661115230025186, df_transf['var'])

    # fit params
    assert transformer.right_tail_caps_ == {'var': 0.20857275540714884}
    assert transformer.left_tail_caps_ == {'var': -0.19661115230025186}
    # transform params
    pd.testing.assert_frame_equal(X, df_transf)
    assert X['var'].max() <= 0.20857275540714884
    assert X['var'].min() >= -0.19661115230025186
    assert dataframe_normal_dist['var'].max() > 0.20857275540714884
    assert dataframe_normal_dist['var'].min() < -0.19661115230025186

    # test case 3: IQR, both tails, fold 1
    transformer = Winsorizer(distribution='skewed', tail='both', fold=1)
    X = transformer.fit_transform(dataframe_normal_dist)

    df_transf = dataframe_normal_dist.copy()
    df_transf['var'] = np.where(df_transf['var'] > 0.21180113880445128, 0.21180113880445128, df_transf['var'])
    df_transf['var'] = np.where(df_transf['var'] < -0.20247907173293223, -0.20247907173293223, df_transf['var'])

    # fit params
    assert transformer.right_tail_caps_ == {'var': 0.21180113880445128}
    assert transformer.left_tail_caps_ == {'var': -0.20247907173293223}
    # transform params
    pd.testing.assert_frame_equal(X, df_transf)
    assert X['var'].max() <= 0.21180113880445128
    assert X['var'].min() >= -0.20247907173293223
    assert dataframe_normal_dist['var'].max() > 0.21180113880445128
    assert dataframe_normal_dist['var'].min() < -0.20247907173293223

    # test case 4: IQR, left tail, fold 2
    transformer = Winsorizer(distribution='skewed', tail='left', fold=0.8)
    X = transformer.fit_transform(dataframe_normal_dist)

    df_transf = dataframe_normal_dist.copy()
    df_transf['var'] = np.where(df_transf['var'] < -0.17486039103044, -0.17486039103044, df_transf['var'])

    # fit params
    assert transformer.right_tail_caps_ == {}
    assert transformer.left_tail_caps_ == {'var': -0.17486039103044}
    # transform params
    pd.testing.assert_frame_equal(X, df_transf)
    assert X['var'].min() >= -0.17486039103044
    assert dataframe_normal_dist['var'].min() < -0.17486039103044

    # test case 5: quantiles, both tails, fold 10%
    transformer = Winsorizer(distribution='quantiles', tail='both', fold=0.1)
    X = transformer.fit_transform(dataframe_normal_dist)

    df_transf = dataframe_normal_dist.copy()
    df_transf['var'] = np.where(df_transf['var'] > 0.14712481122898166, 0.14712481122898166, df_transf['var'])
    df_transf['var'] = np.where(df_transf['var'] < -0.12366227743232801, -0.12366227743232801, df_transf['var'])

    # fit params
    assert transformer.right_tail_caps_ == {'var': 0.14712481122898166}
    assert transformer.left_tail_caps_ == {'var': -0.12366227743232801}
    # transform params
    pd.testing.assert_frame_equal(X, df_transf)
    assert X['var'].max() <= 0.14712481122898166
    assert X['var'].min() >= -0.12366227743232801
    assert dataframe_normal_dist['var'].max() > 0.14712481122898166
    assert dataframe_normal_dist['var'].min() < -0.12366227743232801

    # test case 6: quantiles, right tail, fold 15%
    transformer = Winsorizer(distribution='quantiles', tail='right', fold=0.15)
    X = transformer.fit_transform(dataframe_normal_dist)

    df_transf = dataframe_normal_dist.copy()
    df_transf['var'] = np.where(df_transf['var'] > 0.11823196128033647, 0.11823196128033647, df_transf['var'])

    # fit params
    assert transformer.right_tail_caps_ == {'var': 0.11823196128033647}
    assert transformer.left_tail_caps_ == {}
    # transform params
    pd.testing.assert_frame_equal(X, df_transf)
    assert X['var'].max() <= 0.11823196128033647
    assert dataframe_normal_dist['var'].max() > 0.11823196128033647

    # test case 7: dataset contains na and transformer is asked to ignore them
    transformer = Winsorizer(distribution='gaussian', tail='right', fold=1,
                             variables=['Age', 'Marks'],
                             missing_values='ignore')
    X = transformer.fit_transform(dataframe_na)

    df_transf = dataframe_na.copy()
    df_transf['Age'] = np.where(df_transf['Age'] > 38.79255087111844, 38.79255087111844, df_transf['Age'])
    df_transf['Marks'] = np.where(df_transf['Marks'] > 0.8970309389976613, 0.8970309389976613, df_transf['Marks'])

    # fit params
    assert transformer.right_tail_caps_ == {'Age': 38.79255087111844, 'Marks': 0.8970309389976613}
    assert transformer.left_tail_caps_ == {}
    assert transformer.input_shape_ == (8, 6)
    # transform params
    pd.testing.assert_frame_equal(X, df_transf)
    assert X['Age'].max() <= 38.79255087111844
    assert dataframe_na['Age'].max() > 38.79255087111844

    # test error raises
    with pytest.raises(ValueError):
        Winsorizer(distribution='other')

    with pytest.raises(ValueError):
        Winsorizer(tail='other')

    with pytest.raises(ValueError):
        Winsorizer(missing_values='other')

    with pytest.raises(ValueError):
        Winsorizer(fold=-1)

    with pytest.raises(ValueError):
        Winsorizer(distribution='quantiles', fold=0.3)

    # test case 8: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = Winsorizer()
        transformer.fit(dataframe_na)

    # test case 9: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = Winsorizer()
        transformer.fit(dataframe_vartypes)
        transformer.transform(dataframe_na[['Name', 'City', 'Age', 'Marks', 'dob']])

    with pytest.raises(NotFittedError):
        transformer = Winsorizer()
        transformer.transform(dataframe_vartypes)


def test_ArbitraryOutlierCapper(dataframe_normal_dist, dataframe_na, dataframe_vartypes):
    # test case 1: right end capping
    transformer = ArbitraryOutlierCapper(max_capping_dict={'var': 0.10727677848029868}, min_capping_dict=None)
    X = transformer.fit_transform(dataframe_normal_dist)

    df_transf = dataframe_normal_dist.copy()
    df_transf['var'] = np.where(df_transf['var'] > 0.10727677848029868, 0.10727677848029868, df_transf['var'])

    # init params
    assert transformer.max_capping_dict == {'var': 0.10727677848029868}
    assert transformer.min_capping_dict is None
    assert transformer.variables == ['var']
    # fit params
    assert transformer.right_tail_caps_ == {'var': 0.10727677848029868}
    assert transformer.left_tail_caps_ == {}
    assert transformer.input_shape_ == (100, 1)
    # transform params
    pd.testing.assert_frame_equal(X, df_transf)
    assert X['var'].max() <= 0.10727677848029868
    assert dataframe_normal_dist['var'].max() > 0.10727677848029868

    # test case 2: both tails
    transformer = ArbitraryOutlierCapper(max_capping_dict={'var': 0.20857275540714884},
                                         min_capping_dict={'var': -0.19661115230025186})
    X = transformer.fit_transform(dataframe_normal_dist)

    df_transf = dataframe_normal_dist.copy()
    df_transf['var'] = np.where(df_transf['var'] > 0.20857275540714884, 0.20857275540714884, df_transf['var'])
    df_transf['var'] = np.where(df_transf['var'] < -0.19661115230025186, -0.19661115230025186, df_transf['var'])

    # fit params
    assert transformer.right_tail_caps_ == {'var': 0.20857275540714884}
    assert transformer.left_tail_caps_ == {'var': -0.19661115230025186}
    # transform params
    pd.testing.assert_frame_equal(X, df_transf)
    assert X['var'].max() <= 0.20857275540714884
    assert X['var'].min() >= -0.19661115230025186
    assert dataframe_normal_dist['var'].max() > 0.20857275540714884
    assert dataframe_normal_dist['var'].min() < -0.19661115230025186

    # test case 3: left tail
    transformer = ArbitraryOutlierCapper(max_capping_dict=None, min_capping_dict={'var': -0.17486039103044})
    X = transformer.fit_transform(dataframe_normal_dist)

    df_transf = dataframe_normal_dist.copy()
    df_transf['var'] = np.where(df_transf['var'] < -0.17486039103044, -0.17486039103044, df_transf['var'])

    # init param
    assert transformer.max_capping_dict is None
    assert transformer.min_capping_dict == {'var': -0.17486039103044}
    # fit params
    assert transformer.right_tail_caps_ == {}
    assert transformer.left_tail_caps_ == {'var': -0.17486039103044}
    # transform params
    pd.testing.assert_frame_equal(X, df_transf)
    assert X['var'].min() >= -0.17486039103044
    assert dataframe_normal_dist['var'].min() < -0.17486039103044

    # test case 4: dataset contains na and transformer is asked to ignore them
    transformer = ArbitraryOutlierCapper(max_capping_dict=None, min_capping_dict={'Age': 20},
                                         missing_values='ignore')
    X = transformer.fit_transform(dataframe_na)

    df_transf = dataframe_na.copy()
    df_transf['Age'] = np.where(df_transf['Age'] < 20, 20, df_transf['Age'])

    # fit params
    assert transformer.max_capping_dict is None
    assert transformer.min_capping_dict == {'Age': 20}
    assert transformer.input_shape_ == (8, 6)
    # transform params
    pd.testing.assert_frame_equal(X, df_transf)
    assert X['Age'].min() >= 20
    assert dataframe_na['Age'].min() < 20

    with pytest.raises(ValueError):
        ArbitraryOutlierCapper(max_capping_dict='other')

    with pytest.raises(ValueError):
        ArbitraryOutlierCapper(min_capping_dict='other')

    with pytest.raises(ValueError):
        ArbitraryOutlierCapper(min_capping_dict=None, max_capping_dict=None)

    with pytest.raises(ValueError):
        ArbitraryOutlierCapper(missing_values='other')

    df_na = dataframe_normal_dist.copy()
    df_na.loc[1, 'var'] = np.nan

    # test case 5: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = ArbitraryOutlierCapper(min_capping_dict={'var': -0.17486039103044})
        transformer.fit(df_na)

    # test case 6: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = ArbitraryOutlierCapper(min_capping_dict={'var': -0.17486039103044})
        transformer.fit(dataframe_normal_dist)
        transformer.transform(df_na)

    with pytest.raises(NotFittedError):
        transformer = ArbitraryOutlierCapper(min_capping_dict={'var': -0.17486039103044})
        transformer.transform(dataframe_vartypes)


def test_OutlierTrimmer(dataframe_normal_dist, dataframe_na):
    # test case 1: mean and std, right tail
    transformer = OutlierTrimmer(distribution='gaussian', tail='right', fold=1)
    X = transformer.fit_transform(dataframe_normal_dist)

    df_transf = dataframe_normal_dist.copy()
    outliers = np.where(df_transf['var'] > 0.10727677848029868, True, False)
    df_transf = df_transf.loc[~outliers]

    # transform params
    pd.testing.assert_frame_equal(X, df_transf)
    assert len(X) == 83

    # test case 2: mean and std, both tails, different fold value
    transformer = OutlierTrimmer(distribution='gaussian', tail='both', fold=2)
    X = transformer.fit_transform(dataframe_normal_dist)
    assert len(X) == 96

    # test case 3: IQR, left tail, fold 2
    transformer = OutlierTrimmer(distribution='skewed', tail='left', fold=0.8)
    X = transformer.fit_transform(dataframe_normal_dist)

    df_transf = dataframe_normal_dist.copy()
    outliers = np.where(df_transf['var'] < -0.17486039103044, True, False)
    df_transf = df_transf.loc[~outliers]

    pd.testing.assert_frame_equal(X, df_transf)
    assert len(X) == 98

    # test case 4: dataset contains na, and transformer is asked to ignore
    transformer = OutlierTrimmer(distribution='gaussian', tail='right', fold=1,
                                 variables=['Age'], missing_values='ignore')
    X = transformer.fit_transform(dataframe_na)

    df_transf = dataframe_na.copy()
    outliers = np.where(df_transf['Age'] > 38.79255087111844, True, False)
    df_transf = df_transf.loc[~outliers]

    pd.testing.assert_frame_equal(X, df_transf)
    assert len(X) == 6
