import warnings

import pandas as pd
import pytest
from sklearn.base import clone

from feature_engine.imputation import CategoricalImputer


@pytest.fixture
def multimodal_df():
    return pd.DataFrame(
        {
            "city": [
                "London",
                "London",
                "Paris",
                "Paris",
                "Rome",
                "Rome",
                "Oslo",
                None,
                None,
            ],
            "country": ["UK", "UK", "UK", "FR", "FR", "FR", "DE", None, None],
            "unique": ["a", "a", "a", "a", "b", "b", "c", None, None],
        },
        index=range(10, 19),
        dtype="object",
    )


@pytest.mark.parametrize("multimodal", ["other", 1, True, None, ["warn"]])
def test_invalid_multimodal(multimodal):
    with pytest.raises(ValueError, match="multimodal"):
        CategoricalImputer(multimodal=multimodal)


@pytest.mark.parametrize("multimodal", ["warn", "ignore"])
@pytest.mark.parametrize("variables", ["city", ["city", "country", "unique"]])
@pytest.mark.parametrize("categorical", [False, True])
def test_multimodal_imputes_first_mode(
    multimodal_df, multimodal, variables, categorical
):
    X = multimodal_df.astype("category") if categorical else multimodal_df.copy()
    X_original = X.copy(deep=True)
    imputer = CategoricalImputer(
        imputation_method="frequent", variables=variables, multimodal=multimodal
    )

    if multimodal == "warn":
        with pytest.warns(UserWarning, match="first frequent category") as record:
            result = imputer.fit_transform(X)
        assert len(record) == 1
        assert "city" in str(record[0].message)
        if isinstance(variables, list):
            assert "country" in str(record[0].message)
            assert "unique" not in str(record[0].message)
    else:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = imputer.fit_transform(X)
        assert len(record) == 0

    expected_modes = {"city": "London"}
    if isinstance(variables, list):
        expected_modes.update({"country": "FR", "unique": "a"})
    assert imputer.imputer_dict_ == expected_modes
    assert clone(imputer).get_params()["multimodal"] == multimodal
    pd.testing.assert_frame_equal(result, X.fillna(expected_modes).infer_objects())
    pd.testing.assert_frame_equal(X, X_original)


@pytest.mark.parametrize("multimodal", ["raise", "warn", "ignore"])
def test_multimodal_with_integer_variable_names(multimodal_df, multimodal):
    X = multimodal_df.rename(columns={"city": 0, "country": 1, "unique": 2})
    imputer = CategoricalImputer(
        imputation_method="frequent", variables=[0, 1, 2], multimodal=multimodal
    )
    message = r"0, 1 contain\(s\) multiple frequent categories"
    if multimodal == "raise":
        with pytest.raises(ValueError, match=message):
            imputer.fit(X)
        return
    if multimodal == "warn":
        with pytest.warns(UserWarning, match=message) as record:
            result = imputer.fit_transform(X)
        assert len(record) == 1
    else:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = imputer.fit_transform(X)
        assert len(record) == 0
    pd.testing.assert_frame_equal(
        result, X.fillna({0: "London", 1: "FR", 2: "a"}).infer_objects()
    )


def test_multimodal_raise_reports_different_mode_counts(multimodal_df):
    imputer = CategoricalImputer(imputation_method="frequent")
    with pytest.raises(ValueError, match=r"city, country contain\(s\)"):
        imputer.fit(multimodal_df)


@pytest.mark.parametrize("multimodal", ["raise", "warn", "ignore"])
def test_multimodal_does_not_warn_for_unique_mode(df_na, multimodal):
    imputer = CategoricalImputer(
        imputation_method="frequent", variables="City", multimodal=multimodal
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = imputer.fit_transform(df_na)
    assert len(record) == 0
    pd.testing.assert_series_equal(result["City"], df_na["City"].fillna("London"))


@pytest.mark.parametrize("multimodal", ["raise", "warn", "ignore"])
def test_multimodal_does_not_affect_missing_imputation(multimodal_df, multimodal):
    imputer = CategoricalImputer(
        variables=["city", "country", "unique"], multimodal=multimodal
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = imputer.fit_transform(multimodal_df)
    assert len(record) == 0
    pd.testing.assert_frame_equal(result, multimodal_df.fillna("Missing"))


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
    with pytest.raises(ValueError):
        CategoricalImputer(imputation_method="arbitrary")


def test_error_when_variable_contains_multiple_modes(df_na):
    msg = "The variable Name contains multiple frequent categories."
    imputer = CategoricalImputer(imputation_method="frequent", variables="Name")
    with pytest.raises(ValueError) as record:
        imputer.fit(df_na)
    # check that error message matches
    assert str(record.value) == msg

    msg = "The variable(s) Name contain(s) multiple frequent categories."
    imputer = CategoricalImputer(imputation_method="frequent")
    with pytest.raises(ValueError) as record:
        imputer.fit(df_na)
    # check that error message matches
    assert str(record.value) == msg

    df_ = df_na.copy()
    df_["Name_dup"] = df_["Name"]
    msg = "The variable(s) Name, Name_dup contain(s) multiple frequent categories."
    imputer = CategoricalImputer(imputation_method="frequent")
    with pytest.raises(ValueError) as record:
        imputer.fit(df_)
    # check that error message matches
    assert str(record.value) == msg


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
    with pytest.raises(ValueError) as record:
        CategoricalImputer(imputation_method="missing", ignore_format=ignore_format)

    # check that error message matches
    assert str(record.value) == msg
