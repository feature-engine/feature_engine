import pandas as pd
import pytest

from feature_engine.imputation import MeanMedianImputer


def test_mean_imputation_and_automatically_select_variables(df_na):
    # set up transformer
    imputer = MeanMedianImputer(imputation_method="mean", variables=None)
    X_transformed = imputer.fit_transform(df_na)

    # set up reference result
    X_reference = df_na.copy()
    X_reference["Age"] = X_reference["Age"].fillna(28.714285714285715)
    X_reference["Marks"] = X_reference["Marks"].fillna(0.6833333333333332)

    # test init params
    assert imputer.imputation_method == "mean"
    assert imputer.variables is None

    # test fit attributes
    assert imputer.variables_ == ["Age", "Marks"]
    imputer.imputer_dict_ = {
        key: round(value, 3) for (key, value) in imputer.imputer_dict_.items()
    }
    assert imputer.imputer_dict_ == {
        "Age": 28.714,
        "Marks": 0.683,
    }
    assert imputer.n_features_in_ == 6

    # test transform output:
    # selected variables should have no NA
    # not selected variables should still have NA
    assert X_transformed[["Age", "Marks"]].isnull().sum().sum() == 0
    assert X_transformed[["Name", "City"]].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def test_median_imputation_when_user_enters_single_variables(df_na):
    # set up trasnformer
    imputer = MeanMedianImputer(imputation_method="median", variables=["Age"])
    X_transformed = imputer.fit_transform(df_na)

    # set up reference output
    X_reference = df_na.copy()
    X_reference["Age"] = X_reference["Age"].fillna(23.0)

    # test init params
    assert imputer.imputation_method == "median"
    assert imputer.variables == ["Age"]

    # test fit attributes
    assert imputer.n_features_in_ == 6
    assert imputer.imputer_dict_ == {"Age": 23.0}

    # test transform output
    assert X_transformed["Age"].isnull().sum() == 0
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def test_error_with_wrong_imputation_method():
    with pytest.raises(ValueError):
        MeanMedianImputer(imputation_method="arbitrary")
