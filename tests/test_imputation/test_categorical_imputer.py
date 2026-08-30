import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from feature_engine.imputation import CategoricalImputer

DATA = {
    "Name": ["tom", "nick", "krish", None, "peter", None, "fred", "sam"],
    "City": [
        "London",
        "Manchester",
        None,
        None,
        "London",
        "London",
        "Bristol",
        "Manchester",
    ],
    "Studies": [
        "Bachelor",
        "Bachelor",
        None,
        None,
        "Bachelor",
        "PhD",
        "None",
        "Masters",
    ],
    "Age": [20, 21, 19, None, 23, 40, 41, 37],
    "Marks": [0.9, 0.8, 0.7, None, 0.3, None, 0.8, 0.6],
}


def _cols(X, columns):
    # to_dict(as_series=False) is a convenient, backend-agnostic way to read
    # values back out for comparison, regardless of pandas vs polars.
    result = nw.from_native(X, eager_only=True).to_dict(as_series=False)
    return {c: result[c] for c in columns}


def _null_count(X, col):
    return nw.from_native(X, eager_only=True)[col].null_count()


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_impute_with_string_missing_and_automatically_find_variables(make_df):
    df_na = make_df(DATA)
    imputer = CategoricalImputer(imputation_method="missing", variables=None)
    X_transformed = imputer.fit_transform(df_na)

    # test init params
    assert imputer.imputation_method == "missing"
    assert imputer.variables is None

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
    assert _null_count(X_transformed, "Name") == 0
    assert _null_count(X_transformed, "City") == 0
    assert _null_count(X_transformed, "Studies") == 0
    assert _null_count(X_transformed, "Age") > 0
    assert _null_count(X_transformed, "Marks") > 0
    assert _cols(X_transformed, ["Name", "City", "Studies"]) == {
        "Name": [
            "tom", "nick", "krish", "Missing", "peter", "Missing", "fred", "sam",
        ],
        "City": [
            "London", "Manchester", "Missing", "Missing", "London", "London",
            "Bristol", "Manchester",
        ],
        "Studies": [
            "Bachelor", "Bachelor", "Missing", "Missing", "Bachelor", "PhD",
            "None", "Masters",
        ],
    }


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_user_defined_string_and_automatically_find_variables(make_df):
    df_na = make_df(DATA)
    imputer = CategoricalImputer(
        imputation_method="missing", fill_value="Unknown", variables=None
    )
    X_transformed = imputer.fit_transform(df_na)

    # test init params
    assert imputer.imputation_method == "missing"
    assert imputer.fill_value == "Unknown"
    assert imputer.variables is None

    # test fit attributes
    assert imputer.variables_ == ["Name", "City", "Studies"]
    assert imputer.n_features_in_ == 5
    assert imputer.imputer_dict_ == {
        "Name": "Unknown",
        "City": "Unknown",
        "Studies": "Unknown",
    }

    # test transform output
    assert _null_count(X_transformed, "Name") == 0
    assert _null_count(X_transformed, "City") == 0
    assert _null_count(X_transformed, "Studies") == 0
    assert _null_count(X_transformed, "Age") > 0
    assert _null_count(X_transformed, "Marks") > 0
    assert _cols(X_transformed, ["City"]) == {
        "City": [
            "London", "Manchester", "Unknown", "Unknown", "London", "London",
            "Bristol", "Manchester",
        ],
    }


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_mode_imputation_and_single_variable(make_df):
    df_na = make_df(DATA)
    imputer = CategoricalImputer(imputation_method="frequent", variables="City")
    X_transformed = imputer.fit_transform(df_na)

    # test init, fit and transform params, attr and output
    assert imputer.imputation_method == "frequent"
    assert imputer.variables == "City"
    assert imputer.variables_ == ["City"]
    assert imputer.n_features_in_ == 5
    assert imputer.imputer_dict_ == {"City": "London"}
    assert _null_count(X_transformed, "City") == 0
    assert _null_count(X_transformed, "Age") > 0
    assert _null_count(X_transformed, "Marks") > 0
    assert _cols(X_transformed, ["City"]) == {
        "City": [
            "London", "Manchester", "London", "London", "London", "London",
            "Bristol", "Manchester",
        ],
    }


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_mode_imputation_with_multiple_variables(make_df):
    df_na = make_df(DATA)
    imputer = CategoricalImputer(
        imputation_method="frequent", variables=["Studies", "City"]
    )
    X_transformed = imputer.fit_transform(df_na)

    # test fit attr and transform output
    assert imputer.imputer_dict_ == {"Studies": "Bachelor", "City": "London"}
    assert _cols(X_transformed, ["Studies", "City"]) == {
        "Studies": [
            "Bachelor", "Bachelor", "Bachelor", "Bachelor", "Bachelor", "PhD",
            "None", "Masters",
        ],
        "City": [
            "London", "Manchester", "London", "London", "London", "London",
            "Bristol", "Manchester",
        ],
    }


def test_imputation_of_numerical_vars_cast_as_object_and_returned_as_numerical():
    # Backend-specific: casting a numeric column to pandas' "object" dtype
    # while keeping numeric values (Option 1 in the docstring) is a pandas
    # dtype quirk with no polars equivalent - polars stays typed, so
    # fillna+infer_objects' auto-revert-to-numeric never happens there
    # (see test_polars_return_object_is_a_no_op below).
    df_na = pd.DataFrame(DATA)
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


def test_imputation_of_numerical_vars_cast_as_object_and_returned_as_object():
    # Backend-specific: see comment on the test above - return_object only
    # has an effect on pandas, where infer_objects() silently upcasts.
    df_na = pd.DataFrame(DATA)
    df_na["Marks"] = df_na["Marks"].astype("O")
    imputer = CategoricalImputer(
        imputation_method="frequent",
        variables=["City", "Studies", "Marks"],
        return_object=True,
    )
    X_transformed = imputer.fit_transform(df_na)
    assert X_transformed["Marks"].dtype == "O"


def test_polars_return_object_is_a_no_op():
    # Documents the backend difference: polars never silently upcasts a
    # String-typed column back to numeric (no infer_objects equivalent),
    # so return_object has nothing to do there, unlike on pandas above.
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


def test_error_when_imputation_method_not_frequent_or_missing():
    with pytest.raises(ValueError):
        CategoricalImputer(imputation_method="arbitrary")


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_uses_smallest_mode_when_variable_has_multiple_modes(make_df):
    # every non-null value of "Name" is unique, so all are modes. The imputer
    # picks the sorted-smallest one ("fred") - deterministically and
    # identically for pandas and polars - instead of raising.
    df_na = make_df(DATA)

    # explicit variable
    imputer = CategoricalImputer(imputation_method="frequent", variables="Name")
    imputer.fit(df_na)
    assert imputer.imputer_dict_ == {"Name": "fred"}
    assert _cols(imputer.transform(df_na), ["Name"])["Name"] == [
        "tom",
        "nick",
        "krish",
        "fred",
        "peter",
        "fred",
        "fred",
        "sam",
    ]

    # auto-selected: only "Name" is multi-mode; "City" and "Studies" each have
    # a single mode and are unaffected.
    imputer = CategoricalImputer(imputation_method="frequent")
    imputer.fit(df_na)
    assert imputer.imputer_dict_["Name"] == "fred"
    assert imputer.imputer_dict_["City"] == "London"


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_impute_numerical_variables(make_df):
    df_na = make_df(DATA)
    imputer = CategoricalImputer(
        imputation_method="missing",
        fill_value=0,
        variables=["Name", "City", "Studies", "Age", "Marks"],
        ignore_format=True,
    )
    X_transformed = imputer.fit_transform(df_na)

    # test init params
    assert imputer.imputation_method == "missing"
    assert imputer.variables == ["Name", "City", "Studies", "Age", "Marks"]

    # test fit attributes
    assert imputer.variables_ == ["Name", "City", "Studies", "Age", "Marks"]
    assert imputer.n_features_in_ == 5

    # test transform params: no nulls left anywhere
    for col in ["Name", "City", "Studies", "Age", "Marks"]:
        assert _null_count(X_transformed, col) == 0


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_impute_numerical_variables_with_mode(make_df):
    df_na = make_df(DATA)
    imputer = CategoricalImputer(
        imputation_method="frequent",
        variables=["City", "Studies", "Marks"],
        ignore_format=True,
    )
    X_transformed = imputer.fit_transform(df_na)

    # test init params
    assert imputer.variables == ["City", "Studies", "Marks"]

    # test fit attributes
    assert imputer.variables_ == ["City", "Studies", "Marks"]
    assert imputer.n_features_in_ == 5
    assert imputer.imputer_dict_ == {
        "City": "London",
        "Studies": "Bachelor",
        "Marks": 0.8,
    }

    # test transform output
    for col in ["City", "Studies", "Marks"]:
        assert _null_count(X_transformed, col) == 0


def test_variables_cast_as_category_missing():
    # Backend-specific: pandas' category dtype needs an explicit
    # cat.add_categories() step before fillna, or it raises TypeError -
    # polars' Categorical widens itself automatically on fill_null (see
    # test_polars_categorical_dtype_widens_on_missing_fill below), so
    # there is no shared behaviour to parametrize here.
    df_na = pd.DataFrame(DATA)
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


def test_variables_cast_as_category_frequent():
    # Backend-specific: see comment on test_variables_cast_as_category_missing.
    # The frequent-mode fill value is always an existing category, so this
    # particular case wouldn't actually exercise a real pandas-vs-polars
    # difference - it is kept pandas-only to match the "missing" test above.
    df_na = pd.DataFrame(DATA)
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


def test_polars_categorical_dtype_widens_on_missing_fill():
    # Correctness risk called out for this migration: polars' Categorical
    # (unlike pandas' category dtype) accepts a brand-new value directly on
    # fill_null - no add_categories-equivalent step is needed.
    df_na = pl.DataFrame(DATA).with_columns(pl.col("City").cast(pl.Categorical))

    imputer = CategoricalImputer(
        imputation_method="missing", fill_value="Missing", variables=["City"]
    )
    X_transformed = imputer.fit_transform(df_na)

    assert X_transformed.schema["City"] == pl.Categorical
    assert X_transformed["City"].null_count() == 0
    assert X_transformed["City"].to_list() == [
        "London", "Manchester", "Missing", "Missing", "London", "London",
        "Bristol", "Manchester",
    ]


def test_polars_enum_fixed_categories_raises_on_missing_fill():
    # Correctness risk called out for this migration: polars' Enum has a
    # *fixed* category set. Filling with a value outside it would otherwise
    # silently write null (no error) instead of the intended fill value -
    # we raise a clear error instead of corrupting data silently.
    enum_dtype = pl.Enum(["London", "Manchester", "Bristol"])
    df_na = pl.DataFrame(DATA).with_columns(pl.col("City").cast(enum_dtype))

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
    assert X_transformed["City"].null_count() == 0


@pytest.mark.parametrize(
    "ignore_format",
    [22.3, 1, "HOLA", {"key1": "value1", "key2": "value2", "key3": "value3"}],
)
def test_error_when_ignore_format_is_not_boolean(ignore_format):
    msg = "ignore_format takes only booleans True and False"
    with pytest.raises(ValueError, match=msg):
        CategoricalImputer(imputation_method="missing", ignore_format=ignore_format)
