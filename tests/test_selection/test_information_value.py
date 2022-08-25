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
    X = df_enc.drop("target", axis=1)
    y = df_enc["target"].copy()

    transformer = SelectByInformationValue(
        variables=None,
        threshold=0.2,
        ignore_format=False,
        confirm_variables=False
    )
    X_tr = transformer.fit_transform(X, y)

    expected_results = {
        'var_E': {
            0: 'R',
            1: 'R',
            2: 'R',
            3: 'R',
            4: 'R',
            5: 'R',
            6: 'R',
            7: 'S',
            8: 'S',
            9: 'S',
            10: 'S',
            11: 'T',
            12: 'T',
            13: 'T',
            14: 'T',
            15: 'T',
            16: 'T',
            17: 'T',
            18: 'T',
            19: 'T'
        }
    }
    expected_results_df = pd.DataFrame(expected_results)

    assert X_tr.equals(expected_results_df)
