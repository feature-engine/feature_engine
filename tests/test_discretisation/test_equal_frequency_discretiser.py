import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.discretisation import EqualFrequencyDiscretiser


def test_automatically_find_variables_and_return_as_numeric(df_normal_dist):
    # test case 1: automatically select variables, return_object=False
    transformer = EqualFrequencyDiscretiser(q=10, variables=None, return_object=False)
    X = transformer.fit_transform(df_normal_dist)

    # output expected for fit attr
    _, bins = pd.qcut(x=df_normal_dist["var"], q=10, retbins=True, duplicates="drop")
    bins[0] = float("-inf")
    bins[len(bins) - 1] = float("inf")

    # expected transform output
    X_t = [x for x in range(0, 10)]

    # test init params
    assert transformer.q == 10
    assert transformer.variables is None
    assert transformer.return_object is False
    # test fit attr
    assert transformer.variables_ == ["var"]
    assert transformer.n_features_in_ == 1
    # test transform output
    assert (transformer.binner_dict_["var"] == bins).all()
    assert all(x for x in X["var"].unique() if x not in X_t)
    # in equal frequency discretisation, all intervals get same proportion of values
    assert len((X["var"].value_counts()).unique()) == 1


def test_automatically_find_variables_and_return_as_object(df_normal_dist):
    # test case 2: return variables cast as object
    transformer = EqualFrequencyDiscretiser(q=10, variables=None, return_object=True)
    X = transformer.fit_transform(df_normal_dist)
    assert X["var"].dtypes == "O"


def test_error_when_q_not_number():
    with pytest.raises(ValueError):
        EqualFrequencyDiscretiser(q="other")


def test_error_if_return_object_not_bool():
    with pytest.raises(ValueError):
        EqualFrequencyDiscretiser(return_object="other")


def test_error_if_input_df_contains_na_in_fit(df_na):
    # test case 3: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = EqualFrequencyDiscretiser(q=10, variables=None)
        transformer.fit(df_na)


def test_error_if_input_df_contains_na_in_transform(df_vartypes, df_na):
    # test case 4: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = EqualFrequencyDiscretiser(q=10, variables=None)
        transformer.fit(df_vartypes)
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])


def test_error_when_nan_introduced_during_transform(df_vartypes, df_na):
    # test case 5: when NA is introduced by the transformation
    msg = "During the discretisation, NaN values were introduced " \
          "in the feature(s) var_A."

    # check for warning when errors equals 'ignore'
    with pytest.warns(UserWarning) as record:
        transformer = EqualFrequencyDiscretiser(q=10, variables=None, errors="ignore")
        transformer.fit(df_vartypes)
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])

    # check that only one warning was returned
    assert len(record) == 1
    # check that message matches
    assert record[0].value.args[0] == msg

    # check for error when errors equals 'raise'
    with pytest.raises(ValueError) as record:
        transformer = EqualFrequencyDiscretiser(q=10, variables=None, errors="raise")
        transformer.fit(df_vartypes)
        transformer.transform(df_na["Name", "City", "Age", "Marks", "dob"])

    # check that message matches
    assert str(record.value) == msg


def test_non_fitted_error(df_vartypes):
    with pytest.raises(NotFittedError):
        transformer = EqualFrequencyDiscretiser()
        transformer.transform(df_vartypes)


def test_error_if_not_permitted_value_is_errors():
    with pytest.raises(ValueError):
        EqualFrequencyDiscretiser(errors="medialuna")

