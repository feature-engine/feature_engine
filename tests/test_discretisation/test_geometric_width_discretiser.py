import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.discretisation import GeometricWidthDiscretiser


# test init params
@pytest.mark.parametrize("param", [0.1, "hola", (True, False), {"a": True}, 2])
def test_raises_error_when_return_object_not_bool(param):
    with pytest.raises(ValueError):
        GeometricWidthDiscretiser(return_object=param)


@pytest.mark.parametrize("param", [0.1, "hola", (True, False), {"a": True}, 2])
def test_raises_error_when_return_boundaries_not_bool(param):
    with pytest.raises(ValueError):
        GeometricWidthDiscretiser(return_boundaries=param)


@pytest.mark.parametrize("param", [0.1, "hola", (True, False), {"a": True}, 0, -1])
def test_raises_error_when_precision_not_int(param):
    with pytest.raises(ValueError):
        GeometricWidthDiscretiser(precision=param)


@pytest.mark.parametrize("param", [0.1, "hola", (True, False), {"a": True}])
def test_raises_error_when_bins_not_int(param):
    with pytest.raises(ValueError):
        GeometricWidthDiscretiser(bins=param)


@pytest.mark.parametrize("params", [(False, 1), (True, 10)])
def test_correct_param_assignment_at_init(params):
    param1, param2 = params
    t = GeometricWidthDiscretiser(
        return_object=param1, return_boundaries=param1, precision=param2, bins=param2
    )
    assert t.return_object is param1
    assert t.return_boundaries is param1
    assert t.precision == param2
    assert t.bins == param2


def test_fit_and_transform_methods(df_normal_dist):
    transformer = GeometricWidthDiscretiser(
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

    # transform params
    assert (
        X["var"] == pd.cut(df_normal_dist["var"], bins=bins, precision=7).cat.codes
    ).all()


def test_automatically_find_variables_and_return_as_object(df_normal_dist):
    transformer = GeometricWidthDiscretiser(bins=10, variables=None, return_object=True)
    X = transformer.fit_transform(df_normal_dist)
    assert X["var"].dtypes == "O"


def test_error_if_input_df_contains_na_in_fit(df_na):
    # test case 3: when dataset contains na, fit method
    transformer = GeometricWidthDiscretiser()
    with pytest.raises(ValueError):
        transformer.fit(df_na)


def test_error_if_input_df_contains_na_in_transform(df_vartypes, df_na):
    # test case 4: when dataset contains na, transform method
    transformer = GeometricWidthDiscretiser()
    transformer.fit(df_vartypes)
    with pytest.raises(ValueError):
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])


def test_non_fitted_error(df_vartypes):
    transformer = GeometricWidthDiscretiser()
    with pytest.raises(NotFittedError):
        transformer.transform(df_vartypes)
