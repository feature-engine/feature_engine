import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.outliers import Winsorizer


def test_Windsorizer(dataframe_normal_dist, dataframe_na, dataframe_vartypes):
    # test case 1: mean and std, right tail
    transformer = Winsorizer(capping_method='gaussian', tail='right', fold=1)
    X = transformer.fit_transform(dataframe_normal_dist)

    df_transf = dataframe_normal_dist.copy()
    df_transf['var'] = np.where(df_transf['var'] > 0.10727677848029868, 0.10727677848029868, df_transf['var'])

    # init params
    assert transformer.capping_method == 'gaussian'
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
    transformer = Winsorizer(capping_method='gaussian', tail='both', fold=2)
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
    transformer = Winsorizer(capping_method='iqr', tail='both', fold=1)
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
    transformer = Winsorizer(capping_method='iqr', tail='left', fold=0.8)
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
    transformer = Winsorizer(capping_method='quantiles', tail='both', fold=0.1)
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
    transformer = Winsorizer(capping_method='quantiles', tail='right', fold=0.15)
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
    transformer = Winsorizer(capping_method='gaussian', tail='right', fold=1,
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
        Winsorizer(capping_method='other')

    with pytest.raises(ValueError):
        Winsorizer(tail='other')

    with pytest.raises(ValueError):
        Winsorizer(missing_values='other')

    with pytest.raises(ValueError):
        Winsorizer(fold=-1)

    with pytest.raises(ValueError):
        Winsorizer(capping_method='quantiles', fold=0.3)

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