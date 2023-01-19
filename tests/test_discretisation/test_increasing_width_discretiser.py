import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.discretisation import IncreasingWidthDiscretiser


def test_init_and_fit_params(df_normal_dist):
    transformer = IncreasingWidthDiscretiser(
        bins=10, variables=None, return_object=False
    )
    transformer.fit(df_normal_dist)
    # init params
    assert transformer.bins == 10
    assert transformer.variables is None
    assert transformer.return_object is False
    # fit params
    assert transformer.variables_ == ["var"]
    assert transformer.n_features_in_ == 1


def test_fit_and_transform_methods(df_normal_dist):
    transformer = IncreasingWidthDiscretiser(
        bins=10, variables=None, return_object=False
    )
    X = transformer.fit_transform(df_normal_dist)

    # manual calculation
    min_, max_ = df_normal_dist["var"].min(), df_normal_dist["var"].max()
    increment = np.power(max_ - min_, 1.0 / 10)
    bins = np.r_[-np.inf, min_ + np.power(increment, np.arange(1, 10)), np.inf]
    bins = np.sort(bins)

    # fit params
    assert (transformer.binner_dict_["var"] == bins).all()
    assert (X["var"].unique() == np.arange(0, 10)).all()

    # transform params
    assert (X == np.digitize(df_normal_dist, bins)).all()


def test_automatically_find_variables_and_return_as_object(df_normal_dist):
    transformer = IncreasingWidthDiscretiser(
        bins=10, variables=None, return_object=True
    )
    X = transformer.fit_transform(df_normal_dist)
    assert X["var"].dtypes == "O"


def test_error_when_bins_not_number():
    with pytest.raises(ValueError):
        IncreasingWidthDiscretiser(bins="other")


def test_error_if_return_object_not_bool():
    with pytest.raises(ValueError):
        IncreasingWidthDiscretiser(return_object="other")


def test_error_if_input_df_contains_na_in_fit(df_na):
    # test case 3: when dataset contains na, fit method
    transformer = IncreasingWidthDiscretiser()
    with pytest.raises(ValueError):
        transformer.fit(df_na)


def test_error_if_input_df_contains_na_in_transform(df_vartypes, df_na):
    # test case 4: when dataset contains na, transform method
    transformer = IncreasingWidthDiscretiser()
    transformer.fit(df_vartypes)
    with pytest.raises(ValueError):
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])


def test_non_fitted_error(df_vartypes):
    transformer = IncreasingWidthDiscretiser()
    with pytest.raises(NotFittedError):
        transformer.transform(df_vartypes)
