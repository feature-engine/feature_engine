import re

import pytest

from feature_engine.imputation import ArbitraryImputer, ArbitraryNumberImputer
from tests.backend_helpers import frame_to_dict, null_count


# init parameters
@pytest.mark.parametrize("arbitrary_number", ["arbitrary", [1], None])
def test_error_when_arbitrary_number_not_numeric(arbitrary_number):
    msg = (
        "arbitrary_number must be numeric of type int or float. "
        f"Got {arbitrary_number} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        ArbitraryImputer(arbitrary_number=arbitrary_number)


@pytest.mark.parametrize(
    "imputer_dict", [{"Age": "arbitrary_number"}, {"Age": 1, "Marks": [2]}]
)
def test_error_when_imputer_dict_values_not_numeric(imputer_dict):
    msg = (
        "All values in the dictionary must be integer or float. "
        f"Got {imputer_dict} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        ArbitraryImputer(imputer_dict=imputer_dict)


@pytest.mark.parametrize("imputer_dict", ["Age", ["Age", 1], 1])
def test_error_when_imputer_dict_not_dict(imputer_dict):
    msg = (
        "The parameter can only take a dictionary or None. "
        f"Got {imputer_dict} instead."
    )
    with pytest.raises(TypeError, match=re.escape(msg)):
        ArbitraryImputer(imputer_dict=imputer_dict)


@pytest.mark.parametrize(
    "arbitrary_number, imputer_dict",
    [
        (999, None),
        (-1, None),
        (0.5, {"Age": -42, "Marks": -999}),
        (99, {"Age": 1.5}),
    ],
)
def test_init_param_assignment(arbitrary_number, imputer_dict):
    imputer = ArbitraryImputer(
        arbitrary_number=arbitrary_number, imputer_dict=imputer_dict
    )
    assert imputer.arbitrary_number == arbitrary_number
    assert imputer.imputer_dict == imputer_dict


# fit and transform
def test_impute_with_99_and_automatically_select_variables(make_df, data_na):
    imputer = ArbitraryImputer(arbitrary_number=99, variables=None)
    X_transformed = imputer.fit_transform(make_df(data_na))

    # test fit attributes
    assert imputer.variables_ == ["Age", "Marks"]
    assert imputer.n_features_in_ == 5
    assert imputer.imputer_dict_ == {"Age": 99, "Marks": 99}

    # selected variables should not contain NA, non-selected should still
    assert isinstance(X_transformed, make_df)
    assert null_count(X_transformed, "Age") == 0
    assert null_count(X_transformed, "Marks") == 0
    assert null_count(X_transformed, "Name") > 0
    assert null_count(X_transformed, "City") > 0

    result = frame_to_dict(X_transformed)
    assert result["Age"] == [20, 21, 19, 99, 23, 40, 41, 37]
    assert result["Marks"] == [0.9, 0.8, 0.7, 99, 0.3, 99, 0.8, 0.6]


def test_impute_with_1_and_single_variable_entered_by_user(make_df, data_na):
    imputer = ArbitraryImputer(arbitrary_number=-1, variables=["Age"])
    X_transformed = imputer.fit_transform(make_df(data_na))

    # test fit attributes
    assert imputer.variables_ == ["Age"]
    assert imputer.n_features_in_ == 5
    assert imputer.imputer_dict_ == {"Age": -1}

    assert isinstance(X_transformed, make_df)
    assert null_count(X_transformed, "Age") == 0
    assert frame_to_dict(X_transformed)["Age"] == [20, 21, 19, -1, 23, 40, 41, 37]


def test_dictionary_of_imputation_values(make_df, data_na):
    imputer = ArbitraryImputer(imputer_dict={"Age": -42, "Marks": -999})
    X_transformed = imputer.fit_transform(make_df(data_na))

    # test fit params
    assert imputer.n_features_in_ == 5
    assert imputer.imputer_dict_ == {"Age": -42, "Marks": -999}

    assert isinstance(X_transformed, make_df)
    assert null_count(X_transformed, "Age") == 0
    assert null_count(X_transformed, "Marks") == 0
    assert null_count(X_transformed, "Name") > 0
    assert null_count(X_transformed, "City") > 0

    result = frame_to_dict(X_transformed)
    assert result["Age"] == [20, 21, 19, -42, 23, 40, 41, 37]
    assert result["Marks"] == [0.9, 0.8, 0.7, -999, 0.3, -999, 0.8, 0.6]


def test_arbitrary_number_imputer_is_deprecated():
    """ArbitraryNumberImputer should emit a FutureWarning and still work."""
    with pytest.warns(FutureWarning, match="ArbitraryNumberImputer was deprecated"):
        imputer = ArbitraryNumberImputer(arbitrary_number=99)
    assert isinstance(imputer, ArbitraryImputer)
