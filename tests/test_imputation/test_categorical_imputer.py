import re

import pandas as pd
import polars as pl
import pytest

from feature_engine.imputation import CategoricalImputer
from tests.backend_helpers import frame_to_dict, null_count


# init parameters
@pytest.mark.parametrize("imputation_method", ["arbitrary", "mean", 1])
def test_error_when_imputation_method_not_frequent_or_missing(imputation_method):
    msg = (
        "imputation_method takes only values 'missing' or 'frequent'. "
        f"Got {imputation_method} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        CategoricalImputer(imputation_method=imputation_method)


@pytest.mark.parametrize(
    "ignore_format",
    [22.3, 1, "HOLA", {"key1": "value1", "key2": "value2", "key3": "value3"}],
)
def test_error_when_ignore_format_is_not_boolean(ignore_format):
    msg = (
        "ignore_format takes only booleans True and False. "
        f"Got {ignore_format} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        CategoricalImputer(imputation_method="missing", ignore_format=ignore_format)


@pytest.mark.parametrize(
    "return_object",
    [22.3, 1, "HOLA", {"key1": "value1", "key2": "value2", "key3": "value3"}],
)
def test_error_when_return_object_is_not_boolean(return_object):
    msg = (
        "return_object takes only booleans True and False. "
        f"Got {return_object} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        CategoricalImputer(imputation_method="missing", return_object=return_object)


@pytest.mark.parametrize(
    "imputation_method, fill_value, return_object, ignore_format",
    [
        ("missing", "Missing", False, False),
        ("missing", 0, True, True),
        ("frequent", "Unknown", False, True),
        ("frequent", 1.5, True, False),
    ],
)
def test_init_param_assignment(
    imputation_method, fill_value, return_object, ignore_format
):
    imputer = CategoricalImputer(
        imputation_method=imputation_method,
        fill_value=fill_value,
        return_object=return_object,
        ignore_format=ignore_format,
    )
    assert imputer.imputation_method == imputation_method
    assert imputer.fill_value == fill_value
    assert imputer.return_object is return_object
    assert imputer.ignore_format is ignore_format


# fit and transform
def test_impute_with_string_missing_and_automatically_find_variables(
    make_df, data_na
):
    imputer = CategoricalImputer(imputation_method="missing", variables=None)
    X_transformed = imputer.fit_transform(make_df(data_na))

    # test fit attributes
    assert imputer.variables_ == ["Name", "City", "Studies"]
    assert imputer.n_features_in_ == 5
    assert imputer.imputer_dict_ == {
        "Name": "Missing",
        "City": "Missing",
        "Studies": "Missing",
    }

    # test transform output
    # selected columns should have no NA
    # non selected columns should still have NA
    assert isinstance(X_transformed, make_df)
    assert null_count(X_transformed, "Name") == 0
    assert null_count(X_transformed, "City") == 0
    assert null_count(X_transformed, "Studies") == 0
    assert null_count(X_transformed, "Age") > 0
    assert null_count(X_transformed, "Marks") > 0
    result = frame_to_dict(X_transformed)
    assert result["Name"] == [
        "tom", "nick", "krish", "Missing", "peter", "Missing", "fred", "sam",
    ]
    assert result["City"] == [
        "London", "Manchester", "Missing", "Missing", "London", "London",
        "Bristol", "Manchester",
    ]
    assert result["Studies"] == [
        "Bachelor", "Bachelor", "Missing", "Missing", "Bachelor", "PhD",
        "None", "Masters",
    ]


def test_user_defined_string_and_automatically_find_variables(make_df, data_na):
    imputer = CategoricalImputer(
        imputation_method="missing", fill_value="Unknown", variables=None
    )
    X_transformed = imputer.fit_transform(make_df(data_na))

    # test fit attributes
    assert imputer.variables_ == ["Name", "City", "Studies"]
    assert imputer.n_features_in_ == 5
    assert imputer.imputer_dict_ == {
        "Name": "Unknown",
        "City": "Unknown",
        "Studies": "Unknown",
    }

    # test transform output
    assert isinstance(X_transformed, make_df)
    assert null_count(X_transformed, "Name") == 0
    assert null_count(X_transformed, "City") == 0
    assert null_count(X_transformed, "Studies") == 0
    assert null_count(X_transformed, "Age") > 0
    assert null_count(X_transformed, "Marks") > 0
    assert frame_to_dict(X_transformed)["City"] == [
        "London", "Manchester", "Unknown", "Unknown", "London", "London",
        "Bristol", "Manchester",
    ]


def test_mode_imputation_and_single_variable(make_df, data_na):
    imputer = CategoricalImputer(imputation_method="frequent", variables="City")
    X_transformed = imputer.fit_transform(make_df(data_na))

    # test fit attr and transform output
    assert imputer.variables_ == ["City"]
    assert imputer.n_features_in_ == 5
    assert imputer.imputer_dict_ == {"City": "London"}
    assert isinstance(X_transformed, make_df)
    assert null_count(X_transformed, "City") == 0
    assert null_count(X_transformed, "Age") > 0
    assert null_count(X_transformed, "Marks") > 0
    assert frame_to_dict(X_transformed)["City"] == [
        "London", "Manchester", "London", "London", "London", "London",
        "Bristol", "Manchester",
    ]


def test_mode_imputation_with_multiple_variables(make_df, data_na):
    imputer = CategoricalImputer(
        imputation_method="frequent", variables=["Studies", "City"]
    )
    X_transformed = imputer.fit_transform(make_df(data_na))

    # test fit attr and transform output
    assert imputer.imputer_dict_ == {"Studies": "Bachelor", "City": "London"}
    assert isinstance(X_transformed, make_df)
    result = frame_to_dict(X_transformed)
    assert result["Studies"] == [
        "Bachelor", "Bachelor", "Bachelor", "Bachelor", "Bachelor", "PhD",
        "None", "Masters",
    ]
    assert result["City"] == [
        "London", "Manchester", "London", "London", "London", "London",
        "Bristol", "Manchester",
    ]


def test_imputation_of_numerical_vars_cast_as_object_and_returned_as_numerical(
    data_na,
):
    # casting a numeric column to pandas' "object" dtype while keeping
    # numeric values is a pandas quirk with no polars equivalent.
    df_na = pd.DataFrame(data_na)
    df_na["Marks"] = df_na["Marks"].astype("O")
    imputer = CategoricalImputer(
        imputation_method="frequent", variables=["City", "Studies", "Marks"]
    )
    X_transformed = imputer.fit_transform(df_na)

    X_reference = df_na.copy()
    X_reference["Marks"] = X_reference["Marks"].astype(float).fillna(0.8)
    X_reference["City"] = X_reference["City"].fillna("London")
    X_reference["Studies"] = X_reference["Studies"].fillna("Bachelor")
    assert imputer.variables_ == ["City", "Studies", "Marks"]
    assert imputer.imputer_dict_ == {
        "Studies": "Bachelor",
        "City": "London",
        "Marks": 0.8,
    }
    assert X_transformed["Marks"].dtype == "float"
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def test_imputation_of_numerical_vars_cast_as_object_and_returned_as_object(
    data_na,
):
    # pandas only: see comment on the test above.
    df_na = pd.DataFrame(data_na)
    df_na["Marks"] = df_na["Marks"].astype("O")
    imputer = CategoricalImputer(
        imputation_method="frequent",
        variables=["City", "Studies", "Marks"],
        return_object=True,
    )
    X_transformed = imputer.fit_transform(df_na)
    assert X_transformed["Marks"].dtype == "O"


def test_polars_return_object_is_a_no_op():
    # polars never casts String back to numeric, so return_object has no effect
    df_na = pl.DataFrame(
        {"Marks": ["0.9", "0.8", "0.7", None, "0.3", None, "0.8", "0.6"]}
    )
    imputer = CategoricalImputer(
        imputation_method="frequent",
        variables=["Marks"],
        ignore_format=True,
        return_object=True,
    )
    X_transformed = imputer.fit_transform(df_na)
    assert X_transformed.schema["Marks"] == pl.String


def test_uses_smallest_mode_when_variable_has_multiple_modes(make_df, data_na):
    # every non-null value of "Name" is unique, so all are modes. The imputer
    # picks the sorted-smallest one ("fred") deterministically.
    df_na = make_df(data_na)

    # explicit variable
    imputer = CategoricalImputer(imputation_method="frequent", variables="Name")
    imputer.fit(df_na)
    assert imputer.imputer_dict_ == {"Name": "fred"}
    X_transformed = imputer.transform(df_na)
    assert isinstance(X_transformed, make_df)
    assert frame_to_dict(X_transformed)["Name"] == [
        "tom",
        "nick",
        "krish",
        "fred",
        "peter",
        "fred",
        "fred",
        "sam",
    ]

    # auto-selected: only "Name" is multi-mode; "City" has
    # a single mode and is unaffected.
    imputer = CategoricalImputer(imputation_method="frequent")
    imputer.fit(df_na)
    assert imputer.imputer_dict_["Name"] == "fred"
    assert imputer.imputer_dict_["City"] == "London"


def test_impute_numerical_variables(make_df, data_na):
    imputer = CategoricalImputer(
        imputation_method="missing",
        fill_value=0,
        variables=["Name", "City", "Studies", "Age", "Marks"],
        ignore_format=True,
    )
    X_transformed = imputer.fit_transform(make_df(data_na))

    # test fit attributes
    assert imputer.variables_ == ["Name", "City", "Studies", "Age", "Marks"]
    assert imputer.n_features_in_ == 5

    # test transform params: no nulls left anywhere
    assert isinstance(X_transformed, make_df)
    for col in ["Name", "City", "Studies", "Age", "Marks"]:
        assert null_count(X_transformed, col) == 0


def test_impute_numerical_variables_with_mode(make_df, data_na):
    imputer = CategoricalImputer(
        imputation_method="frequent",
        variables=["City", "Studies", "Marks"],
        ignore_format=True,
    )
    X_transformed = imputer.fit_transform(make_df(data_na))

    # test fit attributes
    assert imputer.variables_ == ["City", "Studies", "Marks"]
    assert imputer.n_features_in_ == 5
    assert imputer.imputer_dict_ == {
        "City": "London",
        "Studies": "Bachelor",
        "Marks": 0.8,
    }

    # test transform output
    assert isinstance(X_transformed, make_df)
    for col in ["City", "Studies", "Marks"]:
        assert null_count(X_transformed, col) == 0


def test_variables_cast_as_category_missing(data_na):
    # pandas only
    df_na = pd.DataFrame(data_na)
    df_na["City"] = df_na["City"].astype("category")

    imputer = CategoricalImputer(imputation_method="missing", variables=None)
    X_transformed = imputer.fit_transform(df_na)

    X_reference = df_na.copy()
    X_reference["Name"] = X_reference["Name"].fillna("Missing")
    X_reference["Studies"] = X_reference["Studies"].fillna("Missing")
    X_reference["City"] = (
        X_reference["City"].cat.add_categories("Missing").fillna("Missing")
    )

    assert imputer.variables_ == ["Name", "City", "Studies"]
    assert imputer.imputer_dict_ == {
        "Name": "Missing",
        "City": "Missing",
        "Studies": "Missing",
    }
    assert X_transformed[["Name", "City", "Studies"]].isnull().sum().sum() == 0
    assert X_transformed[["Age", "Marks"]].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def test_variables_cast_as_category_frequent(data_na):
    # pandas only
    df_na = pd.DataFrame(data_na)
    df_na["City"] = df_na["City"].astype("category")
    df_na = df_na.drop(columns=["Name"])  # this variable has no mode

    imputer = CategoricalImputer(imputation_method="frequent", variables=None)
    X_transformed = imputer.fit_transform(df_na)

    X_reference = df_na.copy()
    X_reference["Studies"] = X_reference["Studies"].fillna("Bachelor")
    X_reference["City"] = X_reference["City"].fillna("London")

    assert imputer.variables_ == ["City", "Studies"]
    assert imputer.imputer_dict_ == {
        "City": "London",
        "Studies": "Bachelor",
    }
    assert X_transformed[["City", "Studies"]].isnull().sum().sum() == 0
    assert X_transformed[["Age", "Marks"]].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def test_polars_categorical_dtype_widens_on_missing_fill(data_na):
    # polars only.
    df_na = pl.DataFrame(data_na).with_columns(pl.col("City").cast(pl.Categorical))

    imputer = CategoricalImputer(
        imputation_method="missing", fill_value="Missing", variables=["City"]
    )
    X_transformed = imputer.fit_transform(df_na)

    assert X_transformed.schema["City"] == pl.Categorical
    assert null_count(X_transformed, "City") == 0
    assert frame_to_dict(X_transformed)["City"] == [
        "London", "Manchester", "Missing", "Missing", "London", "London",
        "Bristol", "Manchester",
    ]


def test_polars_enum_fixed_categories_raises_on_missing_fill(data_na):
    # polars only.
    enum_dtype = pl.Enum(["London", "Manchester", "Bristol"])
    df_na = pl.DataFrame(data_na).with_columns(pl.col("City").cast(enum_dtype))

    imputer = CategoricalImputer(
        imputation_method="missing", fill_value="Missing", variables=["City"]
    )
    with pytest.raises(ValueError, match="polars Enum with fixed categories"):
        imputer.fit_transform(df_na)

    # a fill value that is already a member of the fixed category set works
    imputer_ok = CategoricalImputer(
        imputation_method="missing", fill_value="London", variables=["City"]
    )
    X_transformed = imputer_ok.fit_transform(df_na)
    assert null_count(X_transformed, "City") == 0
