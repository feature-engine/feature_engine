import pandas as pd
import pytest

from feature_engine.encoding import InformationValue


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
        confirm_variables=False,
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


def test_information_value_encoder_dict(df_enc):
    transformer = InformationValue(
        variables=None,
        confirm_variables=False,
        ignore_format=False,
        errors="ignore"
    )
    transformer.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    encoder_dict = transformer.info_value_encoder_dict_

    # create dictionary with rounded values for assert statement
    dict_rounded = _round_dict_of_dict_values(encoder_dict, 5)

    expected_results = {
        "var_A": {"A": 0.00734, "B": 0.12833, "C": 0.16139},
        "var_B": {"A": 0.12833, "B": 0.00734, "C": 0.16139},
    }

    assert dict_rounded == expected_results
