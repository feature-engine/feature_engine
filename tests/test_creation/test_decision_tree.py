import numpy as np
import pandas as pd
import pytest

from feature_engine.creation import DecisionTreeFeatures


def test_create_variable_combinations(df_creation):
    # TEST 1: output_features is None
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

    transformer.fit(
        df_creation.drop("Best_40m_dash_seconds", axis=1),
        df_creation["Best_40m_dash_seconds"],
    )
    results = transformer.variable_combinations_
    assert results == expected_results

    # TEST 2: output_features is an integer
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

    transformer.fit(
        df_creation.drop("Best_40m_dash_seconds", axis=1),
        df_creation["Best_40m_dash_seconds"],
    )
    results = transformer.variable_combinations_
    assert results == expected_results

    # TEST 3: output_features is a list of integers
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

    transformer.fit(
        df_creation.drop("Best_40m_dash_seconds", axis=1),
        df_creation["Best_40m_dash_seconds"],
    )
    results = transformer.variable_combinations_
    assert results == expected_results

    # TEST 4: output_features is a tuple
    transformer = DecisionTreeFeatures(
        variables=["Age", "Marks", "Avg_5k_run_minutes", "Height_cm"],
        output_features=(
            "Height_cm",
            ("Avg_5k_run_minutes", "Height_cm"),
            "Age",
            ("Age", "Height_cm")
        ),
        regression=True,
        max_depth=3,
        drop_original=False
    )
    expected_results = [
        ['Height_cm'],
        ['Avg_5k_run_minutes', 'Height_cm'],
        ['Age'],
        ['Age', 'Height_cm'],
    ]
    transformer.fit(
        df_creation.drop("Best_40m_dash_seconds", axis=1),
        df_creation["Best_40m_dash_seconds"],
    )
    results = transformer.variable_combinations_
    assert results == expected_results


def test_get_distinct_from_output_features():
    output_features = (
        "Age",
        ("Avg_5k_run_minutes", "Plays_Football"),
        ("Age", "Marks", "Avg_5k_run_minutes", "Height_cm"),
        ("City", "Studies", "Plays_Football"),
        ("Height_cm", "Marks", "Studies", "City")
    )

    transformer = DecisionTreeFeatures(
        variables=["Age", "Marks", "Avg_5k_run_minutes", "Height_cm", "Plays_Football"],
        output_features=output_features,
        regression=False,
        max_depth=3,
        drop_original=False
    )

    unique_values = sorted(transformer._get_distinct_output_features())
    expected_results = [
        "Age",
        "Avg_5k_run_minutes",
        "City",
        "Height_cm",
        "Marks",
        "Plays_Football",
        "Studies",
    ]

    assert unique_values == expected_results


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
def test_error_when_output_features_not_permitted(_output_features, df_creation):
    with pytest.raises(ValueError):
        transformer = DecisionTreeFeatures(
            variables=["Age", "Marks", "Height_cm"],
            output_features=_output_features,
            regression=True,
            max_depth=3,
            drop_original=True
        )
        transformer.fit(
            df_creation[["Age", "Marks", "Avg_5k_run_minutes", "Height_cm"]],
            df_creation["Best_40m_dash_seconds"]
        )

@pytest.mark.parametrize("_regression",
                            [3, "summer", [3, 4], ("che", "si")],
                          )
def test_error_when_regression_not_permitted(_regression):

    with pytest.raises(ValueError):
        transformer = DecisionTreeFeatures(
            variables=["Age", "Marks", "Avg_5k_run_minutes", "Height_cm"],
            output_features=3,
            regression=_regression,
            max_depth=3,
            drop_original=True
        )
