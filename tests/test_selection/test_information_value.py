import pandas as pd
import pytest

from feature_engine.selection import InformationValue


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
    dict_rounded = {var_name:{} for var_name in encoder_dict.keys()}
    variables = list(encoder_dict.keys())
    encoder_dict_values = list(encoder_dict.values())
    for var, temp_dict in zip(variables, encoder_dict_values):
        for key, val in temp_dict.items():
            dict_rounded[var][key] = round(val, 5)

    expected_results = {
        "var_A": {"A": 0.04762, "B": -0.2381, "C": 0.19048},
        "var_B": {"A": -0.2381, "B": 0.04762, "C": 0.19048},
    }

    assert dict_rounded == expected_results


