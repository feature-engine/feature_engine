import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.discretisation import BinaryDiscretiser


def test_automatically_find_variables_and_return_as_numeric(df_normal_dist):
    # test case 1: automatically select variables, return_object=False
    transformer = BinaryDiscretiser(threshold=0, variables=None, return_object=False)
    X = transformer.fit_transform(df_normal_dist)

    # transform input
    Xt = np.where(df_normal_dist["var"] > 0, 1, 0)
    bins = [float("-inf"), np.float64(0), float("inf")]

    # init params
    assert transformer.threshold == 0
    assert transformer.variables is None
    assert transformer.return_object is False
    # fit params
    assert transformer.variables_ == ["var"]
    assert transformer.n_features_in_ == 1
    assert transformer.binner_dict_["var"] == bins
    # check transformed output against Xt
    assert all(x == y for x, y in zip(X["var"].values, Xt))


def test_automatically_find_variables_and_return_as_object(df_normal_dist):
    transformer = BinaryDiscretiser(threshold=0, variables=None, return_object=True)
    X = transformer.fit_transform(df_normal_dist)
    assert X["var"].dtypes == "O"


def test_error_when_threshold_not_int_or_float():
    with pytest.raises(TypeError):
        BinaryDiscretiser(threshold="other")


def test_error_when_threshold_not_supplied():
    with pytest.raises(TypeError):
        BinaryDiscretiser()


def test_error_if_return_object_not_bool():
    with pytest.raises(ValueError):
        BinaryDiscretiser(threshold=0, return_object="other")


def test_error_if_input_df_contains_na_in_fit(df_na):
    # test case 3: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = BinaryDiscretiser(threshold=0)
        transformer.fit(df_na)


def test_error_if_input_df_contains_na_in_transform(df_vartypes, df_na):
    # test case 4: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = BinaryDiscretiser(threshold=0)
        transformer.fit(df_vartypes)
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])


def test_non_fitted_error(df_vartypes):
    with pytest.raises(NotFittedError):
        transformer = BinaryDiscretiser(threshold=0)
        transformer.transform(df_vartypes)


def test_stout_threshold_out_of_range(df_vartypes, capsys):
    transformer = BinaryDiscretiser(threshold=20, variables=None, return_object=False)
    _ = transformer.fit_transform(df_vartypes[["Age", "Marks"]])
    captured = capsys.readouterr()
    assert (
        captured.out
        == "threshold outside of range for one or more variables. Features ['Marks'] have not been transformed.\n"
    )


def test_return_boundaries(df_normal_dist):
    transformer = BinaryDiscretiser(threshold=0, return_boundaries=True)
    Xt = transformer.fit_transform(df_normal_dist)
    assert all(x for x in df_normal_dist["var"].unique() if x not in Xt)
