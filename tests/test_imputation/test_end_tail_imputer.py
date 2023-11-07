import numpy as np
import pandas as pd
import pytest

from feature_engine.imputation import EndTailImputer


def test_automatically_find_variables_and_gaussian_imputation_on_right_tail(df_na):
    # set up transformer
    imputer = EndTailImputer(
        imputation_method="gaussian", tail="right", fold=3, variables=None
    )
    X_transformed = imputer.fit_transform(df_na)

    # set up expected output
    X_reference = df_na.copy()
    X_reference["Age"] = X_reference["Age"].fillna(58.94908118478389)
    X_reference["Marks"] = X_reference["Marks"].fillna(1.3244261503263175)

    # test init params
    assert imputer.imputation_method == "gaussian"
    assert imputer.tail == "right"
    assert imputer.fold == 3
    assert imputer.variables is None
    # test fit attr
    assert imputer.variables_ == ["Age", "Marks"]
    assert imputer.n_features_in_ == 6
    imputer.imputer_dict_ = {
        key: round(value, 3) for (key, value) in imputer.imputer_dict_.items()
    }
    assert imputer.imputer_dict_ == {
        "Age": 58.949,
        "Marks": 1.324,
    }
    # transform output: indicated vars ==> no NA, not indicated vars with NA
    assert X_transformed[["Age", "Marks"]].isnull().sum().sum() == 0
    assert X_transformed[["City", "Name"]].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def test_user_enters_variables_and_iqr_imputation_on_right_tail(df_na):
    # set up transformer
    imputer = EndTailImputer(
        imputation_method="iqr", tail="right", fold=1.5, variables=["Age", "Marks"]
    )
    X_transformed = imputer.fit_transform(df_na)

    # set up expected result
    X_reference = df_na.copy()
    X_reference["Age"] = X_reference["Age"].fillna(65.5)
    X_reference["Marks"] = X_reference["Marks"].fillna(1.0625)

    # test fit  and transform attr and output
    assert imputer.imputer_dict_ == {"Age": 65.5, "Marks": 1.0625}
    assert X_transformed[["Age", "Marks"]].isnull().sum().sum() == 0
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def test_user_enters_variables_and_max_value_imputation(df_na):
    imputer = EndTailImputer(
        imputation_method="max", tail="right", fold=2, variables=["Age", "Marks"]
    )
    imputer.fit(df_na)
    assert imputer.imputer_dict_ == {"Age": 82.0, "Marks": 1.8}


def test_automatically_select_variables_and_gaussian_imputation_on_left_tail(df_na):
    imputer = EndTailImputer(imputation_method="gaussian", tail="left", fold=3)
    imputer.fit(df_na)
    imputer.imputer_dict_ = {
        key: round(value, 3) for (key, value) in imputer.imputer_dict_.items()
    }
    assert imputer.imputer_dict_ == {
        "Age": -1.521,
        "Marks": 0.042,
    }


def test_user_enters_variables_and_iqr_imputation_on_left_tail(df_na):
    # test case 5: IQR + left tail
    imputer = EndTailImputer(
        imputation_method="iqr", tail="left", fold=1.5, variables=["Age", "Marks"]
    )
    imputer.fit(df_na)
    assert imputer.imputer_dict_["Age"] == -6.5
    assert np.round(imputer.imputer_dict_["Marks"], 3) == np.round(
        0.36249999999999993, 3
    )


def test_error_when_imputation_method_is_not_permitted():
    with pytest.raises(ValueError):
        EndTailImputer(imputation_method="arbitrary")


def test_error_when_tail_is_string():
    with pytest.raises(ValueError):
        EndTailImputer(tail="arbitrary")


def test_error_when_fold_is_1():
    with pytest.raises(ValueError):
        EndTailImputer(fold=-1)

#TODO: Upddate unit taste
def test_transformation_when_missing_only_is_true(df_na):
    # Case 1: imputation method is IQR
    imputer = EndTailImputer(
        imputation_method="iqr",
        tail="right",
        fold=1.5,
        missing_only=True,
    )
    X = df_na.copy()
    X["Var_No_Nulls"] = [1984] * X.shape[0]
    X_transformed = imputer.fit_transform(X)

    # prepare expected results
    expected_results_df= df_na.copy()
    expected_results_df["Age"] = expected_results_df["Age"].fillna(65.5)
    expected_results_df["Marks"] = expected_results_df["Marks"].fillna(1.0625)
    expected_results_df["Var_No_Nulls"] = [1984] * X.shape[0]

    # test variables being saved and transformed
    assert imputer.variables_ == ["Age", "Marks"]

    # test transform output
    pd.testing.assert_frame_equal(X_transformed, expected_results_df)

    # Case 2: imputation method is Gaussian
    imputer = EndTailImputer(
        imputation_method="gaussian",
        tail="left",
        fold=3,
        missing_only=True,
    )
    X = df_na.copy()
    X["Var_No_Nulls"] = [1984] * X.shape[0]
    X_transformed = imputer.fit_transform(X)

    # prepare expected results
    expected_results_df= df_na.copy()
    expected_results_df["Age"] = expected_results_df["Age"].fillna(-1.521)
    expected_results_df["Marks"] = expected_results_df["Marks"].fillna(0.042)
    expected_results_df["Var_No_Nulls"] = [1984] * X.shape[0]

    # test variables being saved and transformed
    assert imputer.variables_ == ["Age", "Marks"]

    # test transform output
    pd.testing.assert_frame_equal(X_transformed.round(3), expected_results_df)

    # Case 3: imputation method is Max
    imputer = EndTailImputer(
        imputation_method="max",
        tail="right",
        fold=1,
        missing_only=True,
    )
    X = df_na.copy()
    X["Var_No_Nulls"] = [1984] * X.shape[0]
    X_transformed = imputer.fit_transform(X)

    # prepare expected results
    expected_results_df= df_na.copy()
    expected_results_df["Age"] = expected_results_df["Age"].fillna(41.0)
    expected_results_df["Marks"] = expected_results_df["Marks"].fillna(0.9)
    expected_results_df["Var_No_Nulls"] = [1984] * X.shape[0]

    # test variables being saved and transformed
    assert imputer.variables_ == ["Age", "Marks"]

    # test transform output
    pd.testing.assert_frame_equal(X_transformed, expected_results_df)