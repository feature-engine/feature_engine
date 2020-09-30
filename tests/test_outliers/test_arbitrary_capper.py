import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.outliers import ArbitraryOutlierCapper


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