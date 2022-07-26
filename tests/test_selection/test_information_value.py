import pandas as pd
import pytest

from feature_engine.selection import InformationValue


def _round_dict_of_dict_values(data, ndigits):
    """
    Rounds the values of the encoder dictionaries to a given precision
    in decimal digits.

    Parameters:
    -----------
    data: dict
        Dictionary of dictionary. Each variables encoder dictionary.

    ndigits: int
        Precision of decimal digits to round

    Returns:
    --------
    dict_rounded: dict
        Dictionary of dictionaries with rounded values.
    """
    dict_rounded = {var_name: {} for var_name in data.keys()}
    variables = list(data.keys())
    encoder_dict_values = list(data.values())
    for var, temp_dict in zip(variables, encoder_dict_values):
        for key, val in temp_dict.items():
            dict_rounded[var][key] = round(val, ndigits)
    return dict_rounded


def test_class_diff_encoder_dict_calc(df_enc):
    transformer = InformationValue(
        variables=None,
        sort_values=False,
        ignore_format=False,
        errors="ignore"
    )
    transformer.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    encoder_dict = transformer.class_diff_encoder_dict_

    # create dictionary with rounded values for assert statement
    dict_rounded = _round_dict_of_dict_values(encoder_dict, 5)

    expected_results = {
        "var_A": {"A": 0.04762, "B": -0.2381, "C": 0.19048},
        "var_B": {"A": -0.2381, "B": 0.04762, "C": 0.19048},
    }

    assert dict_rounded == expected_results


def test_information_value_calc(df_enc):
    transformer = InformationValue(
        variables=None,
        sort_values=False,
        ignore_format=False,
        errors="ignore"
    )
    transformer.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    information_values = transformer.information_values_

    # create dictionary with rounded values for assert statement
    info_vals_rounded = {
        var: round(val, 5) for var, val in information_values.items()
    }

    expected_results = {"var_A": 0.29706, "var_B": 0.29706}

    assert info_vals_rounded == expected_results


def test_error_when_more_than_two_classes(df_enc_numeric):
    transformer = InformationValue(
        variables=None,
        sort_values=True,
        ignore_format=False,
        errors="ignore"
    )
    with pytest.raises(ValueError):
        transformer.fit(
            df_enc_numeric[["var_A", "target"]], df_enc_numeric["var_B"]
        )


@pytest.mark.parametrize(
    "_sort_values", [3, "python", (True, False), ["var", "fun"]]
)
def test_error_when_not_permitted_param_sort_values(_sort_values):
    with pytest.raises(ValueError):
        InformationValue(
            variables=None,
            sort_values=_sort_values,
            ignore_format=True,
            errors="ignore"
        )