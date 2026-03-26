import re
import warnings

import numpy as np
import pandas as pd
import pytest

from feature_engine.imputation import CategoricalImputer


@pytest.fixture
def multimodal_df():
    return pd.DataFrame(
        {
            "city": ["London", "London", "Paris", "Paris", "Berlin", "Berlin", "Madrid"],
            "country": ["UK", "UK", "FR", "FR", "DE", "DE", "ES"],
            "one_mode": ["London", "London", "London", "Paris", "Paris", "Berlin", "Berlin"],
        }
    )


def test_impute_with_string_missing_and_automatically_find_variables(df_na):
    # set up transformer
    imputer = CategoricalImputer(imputation_method="missing", variables=None)
    X_transformed = imputer.fit_transform(df_na)

    # set up expected output
    X_reference = df_na.copy()
    X_reference["Name"] = X_reference["Name"].fillna("Missing")
    X_reference["City"] = X_reference["City"].fillna("Missing")
    X_reference["Studies"] = X_reference["Studies"].fillna("Missing")

    # test init params
    assert imputer.imputation_method == "missing"
    assert imputer.variables is None

    # test fit attributes
    assert imputer.variables_ == ["Name", "City", "Studies"]
    assert imputer.n_features_in_ == 6
    assert imputer.imputer_dict_ == {
        "Name": "Missing",
        "City": "Missing",
        "Studies": "Missing",
    }

    # test transform output
    # selected columns should have no NA
    # non selected columns should still have NA
    assert X_transformed[["Name", "City", "Studies"]].isnull().sum().sum() == 0
    assert X_transformed[["Age", "Marks"]].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def test_user_defined_string_and_automatically_find_variables(df_na):
    # set up imputer
    imputer = CategoricalImputer(
        imputation_method="missing", fill_value="Unknown", variables=None
    )
    X_transformed = imputer.fit_transform(df_na)

    # set up expected output
    X_reference = df_na.copy()
    X_reference["Name"] = X_reference["Name"].fillna("Unknown")
    X_reference["City"] = X_reference["City"].fillna("Unknown")
    X_reference["Studies"] = X_reference["Studies"].fillna("Unknown")

    # test init params
    assert imputer.imputation_method == "missing"
    assert imputer.fill_value == "Unknown"
    assert imputer.variables is None

    # tes fit attributes
    assert imputer.variables_ == ["Name", "City", "Studies"]
    assert imputer.n_features_in_ == 6
    assert imputer.imputer_dict_ == {
        "Name": "Unknown",
        "City": "Unknown",
        "Studies": "Unknown",
    }

    # test transform output:
    assert X_transformed[["Name", "City", "Studies"]].isnull().sum().sum() == 0
    assert X_transformed[["Age", "Marks"]].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def test_mode_imputation_and_single_variable(df_na):
    # set up imputer
    imputer = CategoricalImputer(imputation_method="frequent", variables="City")
    X_transformed = imputer.fit_transform(df_na)

    # set up expected result
    X_reference = df_na.copy()
    X_reference["City"] = X_reference["City"].fillna("London")

    # test init, fit and transform params, attr and output
    assert imputer.imputation_method == "frequent"
    assert imputer.variables == "City"
    assert imputer.variables_ == ["City"]
    assert imputer.n_features_in_ == 6
    assert imputer.imputer_dict_ == {"City": "London"}
    assert X_transformed["City"].isnull().sum() == 0
    assert X_transformed[["Age", "Marks"]].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def test_mode_imputation_with_multiple_variables(df_na):
    # set up imputer
    imputer = CategoricalImputer(
        imputation_method="frequent", variables=["Studies", "City"]
    )
    X_transformed = imputer.fit_transform(df_na)

    # set up expected output
    X_reference = df_na.copy()
    X_reference["City"] = X_reference["City"].fillna("London")
    X_reference["Studies"] = X_reference["Studies"].fillna("Bachelor")

    # test fit attr and transform output
    assert imputer.imputer_dict_ == {"Studies": "Bachelor", "City": "London"}
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def test_imputation_of_numerical_vars_cast_as_object_and_returned_as_numerical(df_na):
    # test case: imputing of numerical variables cast as object + return numeric
    df_na = df_na.copy()
    df_na["Marks"] = df_na["Marks"].astype("O")
    imputer = CategoricalImputer(
        imputation_method="frequent", variables=["City", "Studies", "Marks"]
    )
    X_transformed = imputer.fit_transform(df_na)

    X_reference = df_na.copy()
    X_reference["Marks"] = X_reference["Marks"].astype(float).fillna(0.8)
    X_reference["City"] = X_reference["City"].fillna("London")
    X_reference["Studies"] = X_reference["Studies"].fillna("Bachelor")
    assert imputer.variables == ["City", "Studies", "Marks"]
    assert imputer.variables_ == ["City", "Studies", "Marks"]
    assert imputer.imputer_dict_ == {
        "Studies": "Bachelor",
        "City": "London",
        "Marks": 0.8,
    }
    assert X_transformed["Marks"].dtype == "float"
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def test_imputation_of_numerical_vars_cast_as_object_and_returned_as_object(df_na):
    # test case 6: imputing of numerical variables cast as object + return as object
    # after imputation
    df_na = df_na.copy()
    df_na["Marks"] = df_na["Marks"].astype("O")
    imputer = CategoricalImputer(
        imputation_method="frequent",
        variables=["City", "Studies", "Marks"],
        return_object=True,
    )
    X_transformed = imputer.fit_transform(df_na)
    assert X_transformed["Marks"].dtype == "O"


def test_error_when_imputation_method_not_frequent_or_missing():
    msg = "imputation_method takes only values 'missing' or 'frequent'"
    with pytest.raises(ValueError, match=msg):
        CategoricalImputer(imputation_method="arbitrary")


def test_error_when_variable_contains_multiple_modes(df_na):
    msg = (
        "The variable Name contains multiple frequent categories. "
        "Set errors='warn' or errors='ignore' to allow imputation "
        "using the first most frequent category found."
    )
    imputer = CategoricalImputer(imputation_method="frequent", variables="Name")
    with pytest.raises(ValueError, match=re.escape(msg)):
        imputer.fit(df_na)

    msg = (
        "The variable(s) Name contain(s) multiple frequent categories. "
        "Set errors='warn' or errors='ignore' to allow imputation "
        "using the first most frequent category found."
    )
    imputer = CategoricalImputer(imputation_method="frequent")
    with pytest.raises(ValueError, match=re.escape(msg)):
        imputer.fit(df_na)

    df_ = df_na.copy()
    df_["Name_dup"] = df_["Name"]
    msg = (
        "The variable(s) Name, Name_dup contain(s) multiple frequent categories. "
        "Set errors='warn' or errors='ignore' to allow imputation "
        "using the first most frequent category found."
    )
    imputer = CategoricalImputer(imputation_method="frequent")
    with pytest.raises(ValueError, match=re.escape(msg)):
        imputer.fit(df_)


def test_impute_numerical_variables(df_na):
    # set up transformer
    imputer = CategoricalImputer(
        imputation_method="missing",
        fill_value=0,
        variables=["Name", "City", "Studies", "Age", "Marks"],
        ignore_format=True,
    )
    X_transformed = imputer.fit_transform(df_na)

    # set up expected output
    X_reference = df_na.copy()
    X_reference = X_reference.fillna(0)

    # test init params
    assert imputer.imputation_method == "missing"
    assert imputer.variables == ["Name", "City", "Studies", "Age", "Marks"]

    # test fit attributes
    assert imputer.variables_ == ["Name", "City", "Studies", "Age", "Marks"]
    assert imputer.n_features_in_ == 6

    # test transform params
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def test_impute_numerical_variables_with_mode(df_na):
    # set up transformer
    imputer = CategoricalImputer(
        imputation_method="frequent",
        variables=["City", "Studies", "Marks"],
        ignore_format=True,
    )
    X_transformed = imputer.fit_transform(df_na)

    # set up expected output
    X_reference = df_na.copy()
    X_reference["City"] = X_reference["City"].fillna("London")
    X_reference["Studies"] = X_reference["Studies"].fillna("Bachelor")
    X_reference["Marks"] = X_reference["Marks"].fillna(0.8)

    # test init params
    assert imputer.variables == ["City", "Studies", "Marks"]

    # test fit attributes
    assert imputer.variables_ == ["City", "Studies", "Marks"]
    assert imputer.n_features_in_ == 6
    assert imputer.imputer_dict_ == {
        "City": "London",
        "Studies": "Bachelor",
        "Marks": 0.8,
    }

    # test transform output
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def test_variables_cast_as_category_missing(df_na):
    # string missing
    df_na = df_na.copy()
    df_na["City"] = df_na["City"].astype("category")

    imputer = CategoricalImputer(imputation_method="missing", variables=None)
    X_transformed = imputer.fit_transform(df_na)

    # set up expected output
    X_reference = df_na.copy()
    X_reference["Name"] = X_reference["Name"].fillna("Missing")
    X_reference["Studies"] = X_reference["Studies"].fillna("Missing")

    X_reference["City"] = (
        X_reference["City"].cat.add_categories("Missing").fillna("Missing")
    )

    # test fit attributes
    assert imputer.variables_ == ["Name", "City", "Studies"]
    assert imputer.imputer_dict_ == {
        "Name": "Missing",
        "City": "Missing",
        "Studies": "Missing",
    }

    # test transform output
    # selected columns should have no NA
    # non selected columns should still have NA
    assert X_transformed[["Name", "City", "Studies"]].isnull().sum().sum() == 0
    assert X_transformed[["Age", "Marks"]].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def test_variables_cast_as_category_frequent(df_na):
    df_na = df_na.copy()
    df_na["City"] = df_na["City"].astype("category")

    # this variable does not have a mode, so drop
    df_na.drop(labels=["Name"], axis=1, inplace=True)

    imputer = CategoricalImputer(imputation_method="frequent", variables=None)
    X_transformed = imputer.fit_transform(df_na)

    # set up expected output
    X_reference = df_na.copy()
    X_reference["Studies"] = X_reference["Studies"].fillna("Bachelor")
    X_reference["City"] = X_reference["City"].fillna("London")

    # test fit attributes
    assert imputer.variables_ == ["City", "Studies"]
    assert imputer.imputer_dict_ == {
        "City": "London",
        "Studies": "Bachelor",
    }

    # test transform output
    # selected columns should have no NA
    # non selected columns should still have NA
    assert X_transformed[["City", "Studies"]].isnull().sum().sum() == 0
    assert X_transformed[["Age", "Marks"]].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, X_reference)


@pytest.mark.parametrize(
    "ignore_format",
    [22.3, 1, "HOLA", {"key1": "value1", "key2": "value2", "key3": "value3"}],
)
def test_error_when_ignore_format_is_not_boolean(ignore_format):
    msg = "ignore_format takes only booleans True and False"
    with pytest.raises(ValueError, match=msg):
        CategoricalImputer(imputation_method="missing", ignore_format=ignore_format)


def test_multimodal_raises_errors(multimodal_df):
    imputer = CategoricalImputer(imputation_method="frequent")
    msg = (
        "The variable(s) city, country contain(s) multiple frequent categories. "
        "Set errors='warn' or errors='ignore' to allow imputation "
        "using the first most frequent category found."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        imputer.fit(multimodal_df)


@pytest.mark.parametrize("errors", ["warn", "ignore"])
def test_multimodal_imputation_result(multimodal_df, errors):
    """Check that result is the same when errors='warn' or 'ignore'."""
    imputer = CategoricalImputer(imputation_method="frequent", errors=errors)
    if errors == "warn":
        with pytest.warns(UserWarning, match="multiple frequent categories"):
            imputer.fit(multimodal_df)
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            imputer.fit(multimodal_df)
            # Check that no warnings with the specific message were raised
            matching_warnings = [
                msg for msg in w if "multiple frequent categories" in str(msg.message)
            ]
            assert len(matching_warnings) == 0


@pytest.mark.parametrize("errors", ["bad_value", 1, True])
def test_errors_invalid_value_raises(errors):
    """Passing an unsupported value for errors should raise ValueError at init."""
    with pytest.raises(ValueError, match="errors takes only values"):
        CategoricalImputer(imputation_method="frequent", errors=errors)


def test_errors_param_ignored_when_imputation_method_is_missing():
    """errors param has no effect for imputation_method='missing'."""
    df = pd.DataFrame({"city": ["London", np.nan, "Paris"]})
    imputer = CategoricalImputer(imputation_method="missing", errors="warn")
    # Should fit without warnings since there's no mode computation
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        imputer.fit(df)
        matching_warnings = [
            msg for msg in w if "multiple frequent categories" in str(msg.message)
        ]
        assert len(matching_warnings) == 0


def test_errors_warn_single_variable_emits_userwarning():
    X = pd.DataFrame(
        {"city": ["London", "London", "Paris", "Paris", "Berlin", "Berlin"]}
    )
    imputer = CategoricalImputer(
        imputation_method="frequent", variables="city", errors="warn"
    )
    with pytest.warns(UserWarning, match="multiple frequent categories"):
        imputer.fit(X)
    # First mode is used
    assert imputer.imputer_dict_["city"] == X["city"].mode()[0]


def test_errors_raise_one_multimodal_among_multiple_variables():
    """
    Covers the `varnames_str = varnames[0]` else-branch in the MULTI-VARIABLE block.

    This branch is reached when multiple variables are selected but only ONE of them
    turns out to have multiple modes. The existing tests either raise on all-multimodal
    datasets (len(varnames) > 1) or use errors='ignore'/'warn' (skipping the raise).
    Here we select two variables where only 'city' is multimodal, triggering the
    singular else-branch before the ValueError is raised.
    """
    X = pd.DataFrame(
        {
            # 'city': 3 equally frequent values → multimodal
            "city": ["London", "London", "Paris", "Paris", "Berlin", "Berlin"],
            # 'country': clear single mode (UK appears 3×, others once)
            "country": ["UK", "UK", "UK", "FR", "DE", "SE"],
        }
    )
    imputer = CategoricalImputer(imputation_method="frequent", errors="raise")
    with pytest.raises(ValueError, match="city"):
        imputer.fit(X)
