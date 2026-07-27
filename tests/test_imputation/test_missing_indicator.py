import re
import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.pipeline import Pipeline

from feature_engine.imputation import AddMissingIndicator, MissingIndicator

DEPRECATION_WARNING = (
    "AddMissingIndicator was deprecated in favour of MissingIndicator in "
    "version 2.0.0 and will be removed in version 2.1.0. To silence this "
    "warning, use MissingIndicator instead."
)


@pytest.fixture(
    params=[MissingIndicator, AddMissingIndicator],
    ids=["MissingIndicator", "AddMissingIndicator"],
)
def indicator_class(request):
    return request.param


def make_indicator(indicator_class, **kwargs):
    if indicator_class is AddMissingIndicator:
        with pytest.warns(FutureWarning, match=re.escape(DEPRECATION_WARNING)):
            return indicator_class(**kwargs)
    return indicator_class(**kwargs)


def test_add_missing_indicator_raises_future_warning():
    with pytest.warns(FutureWarning, match=re.escape(DEPRECATION_WARNING)):
        AddMissingIndicator()


def test_detect_variables_with_missing_data_when_variables_is_none(
    df_na, indicator_class
):
    # test case 1: automatically detect variables with missing data
    imputer = make_indicator(indicator_class, missing_only=True, variables=None)
    X_transformed = imputer.fit_transform(df_na)

    # init params
    assert imputer.missing_only is True
    assert imputer.variables is None
    # fit params
    assert imputer.variables_ == ["Name", "City", "Studies", "Age", "Marks"]
    assert imputer.n_features_in_ == 6
    # transform outputs
    assert X_transformed.shape == (8, 11)
    assert "Name_na" in X_transformed.columns
    assert X_transformed["Name_na"].sum() == 2


def test_add_indicators_to_all_variables_when_variables_is_none(df_na, indicator_class):
    imputer = make_indicator(indicator_class, missing_only=False, variables=None)
    X_transformed = imputer.fit_transform(df_na)
    assert imputer.variables_ == ["Name", "City", "Studies", "Age", "Marks", "dob"]
    assert X_transformed.shape == (8, 12)
    assert "dob_na" in X_transformed.columns
    assert X_transformed["dob_na"].sum() == 0


def test_add_indicators_to_one_variable(df_na, indicator_class):
    imputer = make_indicator(indicator_class, variables="Name")
    X_transformed = imputer.fit_transform(df_na)
    assert imputer.variables_ == ["Name"]
    assert X_transformed.shape == (8, 7)
    assert "Name_na" in X_transformed.columns
    assert X_transformed["Name_na"].sum() == 2


def test_detect_variables_with_missing_data_in_variables_entered_by_user(
    df_na, indicator_class
):
    imputer = make_indicator(
        indicator_class,
        missing_only=True,
        variables=["City", "Studies", "Age", "dob"],
    )
    X_transformed = imputer.fit_transform(df_na)
    assert imputer.variables == ["City", "Studies", "Age", "dob"]
    assert imputer.variables_ == ["City", "Studies", "Age"]
    assert X_transformed.shape == (8, 9)
    assert "City_na" in X_transformed.columns
    assert "dob_na" not in X_transformed.columns
    assert X_transformed["City_na"].sum() == 2


def test_error_when_missing_only_not_bool(indicator_class):
    with pytest.raises(ValueError):
        make_indicator(indicator_class, missing_only="missing_only")


def test_get_feature_names_out(df_na, indicator_class):
    original_features = df_na.columns.to_list()

    tr = make_indicator(indicator_class, missing_only=False)
    tr.fit(df_na)

    out = [f + "_na" for f in original_features]
    feat_out = original_features + out

    assert tr.get_feature_names_out(input_features=None) == feat_out
    assert tr.get_feature_names_out(input_features=original_features) == feat_out

    tr = make_indicator(indicator_class, missing_only=True)
    tr.fit(df_na)

    out = [f + "_na" for f in original_features[0:-1]]
    feat_out = original_features + out

    assert tr.get_feature_names_out(input_features=None) == feat_out
    assert tr.get_feature_names_out(input_features=original_features) == feat_out

    with pytest.raises(ValueError):
        tr.get_feature_names_out("Name")

    with pytest.raises(ValueError):
        tr.get_feature_names_out(["Name", "hola"])


def test_get_feature_names_out_from_pipeline(df_na, indicator_class):
    original_features = df_na.columns.to_list()

    tr = Pipeline(
        [("transformer", make_indicator(indicator_class, missing_only=False))]
    )
    tr.fit(df_na)

    out = [f + "_na" for f in original_features]
    feat_out = original_features + out

    assert tr.get_feature_names_out(input_features=None) == feat_out
    assert tr.get_feature_names_out(input_features=original_features) == feat_out


def test_no_performance_warning_with_many_variables(indicator_class):
    n_cols = 101
    df = pd.DataFrame(
        np.random.randn(10, n_cols),
        columns=[f"col_{i}" for i in range(n_cols)],
    )

    # Introduce missing values
    df.iloc[0, :] = np.nan

    ami = make_indicator(indicator_class, missing_only=False)
    ami.fit(df)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        ami.transform(df)

    assert not any(
        issubclass(w.category, pd.errors.PerformanceWarning)
        for w in captured
    ), "PerformanceWarning was raised during transform"
