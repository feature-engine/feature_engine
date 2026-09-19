import re

import numpy as np
import pytest

from feature_engine.imputation import EndTailImputer
from tests.backend_helpers import frame_to_dict, null_count


# init parameters
@pytest.mark.parametrize(
    "imputation_method", ["arbitrary", "mean", 1, ("iqr",), ["iqr"]]
)
def test_error_when_imputation_method_is_not_permitted(imputation_method):
    msg = (
        "imputation_method takes only values 'gaussian', 'iqr' or 'max'. "
        f"Got {imputation_method} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        EndTailImputer(imputation_method=imputation_method)


@pytest.mark.parametrize("tail", ["arbitrary", "both", 1, ("right",), ["right"]])
def test_error_when_tail_is_not_permitted(tail):
    msg = f"tail takes only values 'right' or 'left'. Got {tail} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        EndTailImputer(tail=tail)


@pytest.mark.parametrize("fold", [-1, 0, -0.5, "3", None, [3], True])
def test_error_when_fold_is_not_positive_number(fold):
    msg = f"fold takes only positive numbers. Got {fold} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        EndTailImputer(fold=fold)


@pytest.mark.parametrize(
    "imputation_method, tail, fold",
    [("gaussian", "right", 3), ("iqr", "left", 1.5), ("max", "right", 2)],
)
def test_init_param_assignment(imputation_method, tail, fold):
    imputer = EndTailImputer(imputation_method=imputation_method, tail=tail, fold=fold)
    assert imputer.imputation_method == imputation_method
    assert imputer.tail == tail
    assert imputer.fold == fold


# fit and transform
def test_automatically_find_variables_and_gaussian_imputation_on_right_tail(
    make_df, data_na
):
    imputer = EndTailImputer(
        imputation_method="gaussian", tail="right", fold=3, variables=None
    )
    X_transformed = imputer.fit_transform(make_df(data_na))

    # test fit attr
    assert imputer.variables_ == ["Age", "Marks"]
    assert imputer.n_features_in_ == 5
    rounded = {k: round(v, 3) for k, v in imputer.imputer_dict_.items()}
    assert rounded == {"Age": 58.949, "Marks": 1.324}

    # transform output: indicated vars ==> no NA, not indicated vars with NA
    assert isinstance(X_transformed, make_df)
    assert null_count(X_transformed, "Age") == 0
    assert null_count(X_transformed, "Marks") == 0
    assert null_count(X_transformed, "City") > 0
    assert null_count(X_transformed, "Name") > 0

    expected = dict(data_na)
    expected["Age"] = pytest.approx([20, 21, 19, 58.94908118478389, 23, 40, 41, 37])
    expected["Marks"] = pytest.approx(
        [0.9, 0.8, 0.7, 1.3244261503263175, 0.3, 1.3244261503263175, 0.8, 0.6]
    )
    assert frame_to_dict(X_transformed) == expected


def test_user_enters_variables_and_iqr_imputation_on_right_tail(make_df, data_na):
    imputer = EndTailImputer(
        imputation_method="iqr", tail="right", fold=1.5, variables=["Age", "Marks"]
    )
    X_transformed = imputer.fit_transform(make_df(data_na))

    assert imputer.imputer_dict_ == {"Age": 65.5, "Marks": 1.0625}
    assert isinstance(X_transformed, make_df)
    assert null_count(X_transformed, "Age") == 0
    assert null_count(X_transformed, "Marks") == 0

    expected = dict(data_na)
    expected["Age"] = pytest.approx([20, 21, 19, 65.5, 23, 40, 41, 37])
    expected["Marks"] = pytest.approx([0.9, 0.8, 0.7, 1.0625, 0.3, 1.0625, 0.8, 0.6])
    assert frame_to_dict(X_transformed) == expected


def test_user_enters_variables_and_max_value_imputation(make_df, data_na):
    imputer = EndTailImputer(
        imputation_method="max", tail="right", fold=2, variables=["Age", "Marks"]
    )
    imputer.fit(make_df(data_na))
    assert imputer.imputer_dict_ == {"Age": 82.0, "Marks": 1.8}


def test_automatically_select_variables_and_gaussian_imputation_on_left_tail(
    make_df, data_na
):
    imputer = EndTailImputer(imputation_method="gaussian", tail="left", fold=3)
    imputer.fit(make_df(data_na))
    rounded = {k: round(v, 3) for k, v in imputer.imputer_dict_.items()}
    assert rounded == {"Age": -1.521, "Marks": 0.042}


def test_user_enters_variables_and_iqr_imputation_on_left_tail(make_df, data_na):
    imputer = EndTailImputer(
        imputation_method="iqr", tail="left", fold=1.5, variables=["Age", "Marks"]
    )
    imputer.fit(make_df(data_na))
    assert imputer.imputer_dict_["Age"] == -6.5
    assert np.round(imputer.imputer_dict_["Marks"], 3) == np.round(
        0.36249999999999993, 3
    )
