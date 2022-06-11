import numpy as np
import pandas as pd
import pytest

from feature_engine.creation import DecisionTreeFeatures


def test_create_variable_combinations(df_creation):
    # output_features is None
    transformer = DecisionTreeFeatures(
        variables=["Age", "Marks", "Avg_5k_run_minutes", "Height_cm"],
        output_features=None,
        regression=True,
        max_depth=3,
        drop_original=False
    )
    expected_results = [
        ['Age'],
        ['Marks'],
        ['Avg_5k_run_minutes'],
        ['Height_cm'],
        ['Age', 'Marks'],
        ['Age', 'Avg_5k_run_minutes'],
        ['Age', 'Height_cm'],
        ['Marks', 'Avg_5k_run_minutes'],
        ['Marks', 'Height_cm'],
        ['Avg_5k_run_minutes', 'Height_cm'],
        ['Age', 'Marks', 'Avg_5k_run_minutes'],
        ['Age', 'Marks', 'Height_cm'],
        ['Age', 'Avg_5k_run_minutes', 'Height_cm'],
        ['Marks', 'Avg_5k_run_minutes', 'Height_cm'],
        ['Age', 'Marks', 'Avg_5k_run_minutes', 'Height_cm'],
    ]

    results = transformer._create_variable_combinations()
    assert results == expected_results

    # output_features is an integer
    transformer = DecisionTreeFeatures(
        variables=["Age", "Marks", "Avg_5k_run_minutes", "Height_cm"],
        output_features=3,
        regression=True,
        max_depth=3,
        drop_original=False
    )
    expected_results = [
        ['Age'],
        ['Marks'],
        ['Avg_5k_run_minutes'],
        ['Height_cm'],
        ['Age', 'Marks'],
        ['Age', 'Avg_5k_run_minutes'],
        ['Age', 'Height_cm'],
        ['Marks', 'Avg_5k_run_minutes'],
        ['Marks', 'Height_cm'],
        ['Avg_5k_run_minutes', 'Height_cm'],
        ['Age', 'Marks', 'Avg_5k_run_minutes'],
        ['Age', 'Marks', 'Height_cm'],
        ['Age', 'Avg_5k_run_minutes', 'Height_cm'],
        ['Marks', 'Avg_5k_run_minutes', 'Height_cm'],
    ]

    results = transformer._create_variable_combinations()
    assert results == expected_results

    # output_features is a list of integers
    transformer = DecisionTreeFeatures(
        variables=["Age", "Marks", "Avg_5k_run_minutes", "Height_cm"],
        output_features=[1, 3],
        regression=True,
        max_depth=3,
        drop_original=False
    )
    expected_results = [
        ['Age'],
        ['Marks'],
        ['Avg_5k_run_minutes'],
        ['Height_cm'],
        ['Age', 'Marks', 'Avg_5k_run_minutes'],
        ['Age', 'Marks', 'Height_cm'],
        ['Age', 'Avg_5k_run_minutes', 'Height_cm'],
        ['Marks', 'Avg_5k_run_minutes', 'Height_cm'],
    ]

    results = transformer._create_variable_combinations()
    assert results == expected_results

    # output_features
    transformer = DecisionTreeFeatures(
        variables=["Age", "Marks", "Avg_5k_run_minutes", "Height_cm"],
        output_features=(
            "Height_cm",
            ("Avg_5k_run_minutes", "Height_cm"),
            "Age",
            ("Age", "Studies", "Height_cm")
        ),
        regression=True,
        max_depth=3,
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


@pytest.mark.parametrize(
    "_output_features",
    [
        6,
        [1, 3, 5],
        [1, 2, "Marks"],
        (("Age", "Marks"), 3),
        "Height_cm",
        ("Avg_5k_run_minutes", ("Height_cm", "Banana"), ("Marks", "Height_cm"))

    ],
)
def test_error_when_output_features_not_permitted(_output_features):
    with pytest.raises(ValueError):
        transformer = DecisionTreeFeatures(
            variables=["Age", "Marks", "Avg_5k_run_minutes", "Height_cm"],
            output_features=_output_features,
            regression=True,
            max_depth=3,
            drop_original=True
        )
