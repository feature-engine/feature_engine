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
