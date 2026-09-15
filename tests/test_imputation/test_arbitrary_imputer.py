import pytest

from feature_engine.imputation import ArbitraryImputer, ArbitraryNumberImputer
from tests.backend_helpers import null_count, frame_to_dict


def test_impute_with_99_and_automatically_select_variables(make_df, data_na):
    imputer = ArbitraryImputer(arbitrary_number=99, variables=None)
    X_transformed = imputer.fit_transform(make_df(data_na))

    # test init params
    assert imputer.arbitrary_number == 99
    assert imputer.variables is None

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

    # test init params
    assert imputer.arbitrary_number == -1
    assert imputer.variables == ["Age"]

    # test fit attributes
    assert imputer.variables_ == ["Age"]
    assert imputer.n_features_in_ == 5
    assert imputer.imputer_dict_ == {"Age": -1}

    assert isinstance(X_transformed, make_df)
    assert null_count(X_transformed, "Age") == 0
    assert frame_to_dict(X_transformed)["Age"] == [20, 21, 19, -1, 23, 40, 41, 37]


def test_error_when_arbitrary_number_is_string():
    with pytest.raises(ValueError):
        ArbitraryImputer(arbitrary_number="arbitrary")


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


def test_imputer_error_when_dictionary_value_is_string():
    with pytest.raises(ValueError):
        ArbitraryImputer(imputer_dict={"Age": "arbitrary_number"})


def test_arbitrary_number_imputer_is_deprecated():
    """ArbitraryNumberImputer should emit a FutureWarning and still work."""
    with pytest.warns(FutureWarning, match="ArbitraryNumberImputer was deprecated"):
        imputer = ArbitraryNumberImputer(arbitrary_number=99)
    assert isinstance(imputer, ArbitraryImputer)
    assert imputer.arbitrary_number == 99
