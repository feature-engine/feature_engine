import numpy as np
import pandas as pd
import pytest

from feature_engine.creation import DecisionTreeCreation


def test_create_variable_combinations(df_creation):
    # output_features is None
    transformer = DecisionTreeCreation(
        variables=["Age", "Studies", "Avg_5k_run_minutes", "Height_cm"],
        output_features=None,
        regression=True,
        max_depth=3,
        missing_value="raise",
        drop_original=False
    )
    expected_results = [
        ['Age'],
        ['Studies'],
        ['Avg_5k_run_minutes'],
        ['Height_cm'],
        ['Age', 'Studies'],
        ['Age', 'Avg_5k_run_minutes'],
        ['Age', 'Height_cm'],
        ['Studies', 'Avg_5k_run_minutes'],
        ['Studies', 'Height_cm'],
        ['Avg_5k_run_minutes', 'Height_cm'],
        ['Age', 'Studies', 'Avg_5k_run_minutes'],
        ['Age', 'Studies', 'Height_cm'],
        ['Age', 'Avg_5k_run_minutes', 'Height_cm'],
        ['Studies', 'Avg_5k_run_minutes', 'Height_cm'],
        ['Age', 'Studies', 'Avg_5k_run_minutes', 'Height_cm'],
    ]

    results = transformer._create_variable_combinations()
    assert results == expected_results

    # output_features is an integer
    transformer = DecisionTreeCreation(
        variables=["Age", "Studies", "Avg_5k_run_minutes", "Height_cm"],
        output_features=4,
        regression=True,
        max_depth=3,
        missing_value="raise",
        drop_original=False
    )
    expected_results = [
        ['Age'],
        ['Studies'],
        ['Avg_5k_run_minutes'],
        ['Height_cm'],
        ['Age', 'Studies'],
        ['Age', 'Avg_5k_run_minutes'],
        ['Age', 'Height_cm'],
        ['Studies', 'Avg_5k_run_minutes'],
        ['Studies', 'Height_cm'],
        ['Avg_5k_run_minutes', 'Height_cm'],
        ['Age', 'Studies', 'Avg_5k_run_minutes'],
        ['Age', 'Studies', 'Height_cm'],
        ['Age', 'Avg_5k_run_minutes', 'Height_cm'],
        ['Studies', 'Avg_5k_run_minutes', 'Height_cm'],
        ['Age', 'Studies', 'Avg_5k_run_minutes', 'Height_cm'],
    ]

    results = transformer._create_variable_combinations()
    assert results == expected_results

    # output_features is a list of integers
    transformer = DecisionTreeCreation(
        variables=["Age", "Studies", "Avg_5k_run_minutes", "Height_cm"],
        output_features=[1, 3],
        regression=True,
        max_depth=3,
        missing_value="raise",
        drop_original=False
    )
    expected_results = [
        ['Age'],
        ['Studies'],
        ['Avg_5k_run_minutes'],
        ['Height_cm'],
        ['Age', 'Studies', 'Avg_5k_run_minutes'],
        ['Age', 'Studies', 'Height_cm'],
        ['Age', 'Avg_5k_run_minutes', 'Height_cm'],
        ['Studies', 'Avg_5k_run_minutes', 'Height_cm'],
    ]

    results = transformer._create_variable_combinations()
    assert results == expected_results

    # output_features
    transformer = DecisionTreeCreation(
        variables=["Age", "Studies", "Avg_5k_run_minutes", "Height_cm"],
        output_features=(
            "Height_cm",
            ("Avg_5k_run_minutes", "Height_cm"),
            "Age",
            ("Age", "Studies", "Height_cm")
        ),
        regression=True,
        max_depth=3,
        missing_value="raise",
        drop_original=False
    )
    expected_results = [
        ['Height_cm'],
        ['Avg_5k_run_minutes', 'Height_cm'],
        ['Age'],
        ['Age', 'Studies', 'Height_cm'],
    ]
    results = transformer._create_variable_combinations()
    assert results == expected_results
