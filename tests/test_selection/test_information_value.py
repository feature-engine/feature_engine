import pandas as pd
import pytest

from feature_engine.selection import SelectByInformationValue


@pytest.mark.parametrize(
    "_threshold", ["python", (True, False), [4.3, 3]]
)
def test_error_when_not_permitted_threshold(_threshold):
    with pytest.raises(ValueError):
        SelectByInformationValue(
            variables=None,
            threshold=_threshold,
            ignore_format=False,
            confirm_variables=False,
        )


def test_error_when_more_than_two_classes(df_enc_numeric):
    transformer = SelectByInformationValue(
        variables=None,
        threshold=0.2,
        ignore_format=False,
        confirm_variables=False
    )
    with pytest.raises(ValueError):
        transformer.fit(
            df_enc_numeric[["var_A", "target"]], df_enc_numeric["var_B"]
        )


def test_error_when_no_categorical_values(df_test):
    X, y = df_test

    transformer = SelectByInformationValue(
        variables=None,
        threshold=0.2,
        ignore_format=False,
        confirm_variables=False
    )
    with pytest.raises(ValueError):
        transformer.fit_transform(X, y)


def test_transformer_with_default_params(df_enc):
    X = df_enc.drop("target", axis=1).copy()
    y = df_enc["target"].copy()

    transformer = SelectByInformationValue(
        variables=None,
        threshold=0.2,
        ignore_format=False,
        confirm_variables=False
    )
    X_tr = transformer.fit_transform(X, y)
    expected_results_df = X[["var_E"]].copy()

    assert X_tr.equals(expected_results_df)


def test_transformer_with_selected_variables(df_enc):
    X = df_enc.drop("target", axis=1).copy()
    y = df_enc["target"].copy()

    transformer = SelectByInformationValue(
        variables=["var_A", "var_C", "var_E"],
        threshold=-0.5,
        ignore_format=False,
        confirm_variables=False
    )
    X_tr = transformer.fit_transform(X, y)
    expected_results_df = X[["var_B", "var_D", "var_E"]].copy()

    assert X_tr.equals(expected_results_df)