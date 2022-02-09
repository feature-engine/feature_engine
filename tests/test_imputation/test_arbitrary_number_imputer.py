import pandas as pd
import pytest

from feature_engine.imputation import ArbitraryNumberImputer


def test_impute_with_99_and_automatically_select_variables(df_na):
    # set up the transformer
    imputer = ArbitraryNumberImputer(arbitrary_number=99, variables=None)
    X_transformed = imputer.fit_transform(df_na)

    # set up output reference
    X_reference = df_na.copy()
    X_reference["Age"] = X_reference["Age"].fillna(99)
    X_reference["Marks"] = X_reference["Marks"].fillna(99)

    # test init params
    assert imputer.arbitrary_number == 99
    assert imputer.variables is None

    # test fit attributes
    assert imputer.variables_ == ["Age", "Marks"]
    assert imputer.n_features_in_ == 6
    assert imputer.imputer_dict_ == {"Age": 99, "Marks": 99}

    # test transform output
    # selected variables should not contain NA
    # non selected variables should still contain NA
    assert X_transformed[["Age", "Marks"]].isnull().sum().sum() == 0
    assert X_transformed[["Name", "City"]].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def test_impute_with_1_and_single_variable_entered_by_user(df_na):
    # set up transformer
    imputer = ArbitraryNumberImputer(arbitrary_number=-1, variables=["Age"])
    X_transformed = imputer.fit_transform(df_na)

    # set up output reference
    X_reference = df_na.copy()
    X_reference["Age"] = X_reference["Age"].fillna(-1)

    # test init params
    assert imputer.arbitrary_number == -1
    assert imputer.variables == ["Age"]

    # test fit attributes
    assert imputer.variables_ == ["Age"]
    assert imputer.n_features_in_ == 6
    assert imputer.imputer_dict_ == {"Age": -1}

    # test transform output
    assert X_transformed["Age"].isnull().sum() == 0
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def test_error_when_arbitrary_number_is_string():
    with pytest.raises(ValueError):
        ArbitraryNumberImputer(arbitrary_number="arbitrary")


def test_dictionary_of_imputation_values(df_na):
    # set up transformer
    imputer = ArbitraryNumberImputer(imputer_dict={"Age": -42, "Marks": -999})
    X_transformed = imputer.fit_transform(df_na)

    # set up expected output
    X_reference = df_na.copy()
    X_reference["Age"] = X_reference["Age"].fillna(-42)
    X_reference["Marks"] = X_reference["Marks"].fillna(-999)

    # test fit params
    assert imputer.n_features_in_ == 6
    assert imputer.imputer_dict_ == {"Age": -42, "Marks": -999}

    # test transform params
    assert X_transformed[["Age", "Marks"]].isnull().sum().sum() == 0
    assert X_transformed[["Name", "City"]].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def imputer_error_when_dictionary_value_is_string():
    with pytest.raises(ValueError):
        ArbitraryNumberImputer(imputer_dict={"Age": "arbitrary_number"})
