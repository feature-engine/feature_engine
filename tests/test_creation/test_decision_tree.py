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
        random_state=0,
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
        random_state=0,
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
        random_state=0,
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
        random_state=0,
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
        random_state=0,
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
            random_state=0,
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
            random_state=0,
            drop_original=True
        )


@pytest.mark.parametrize("_max_depth",
                         ["copado", (3, 5), [1, 2, 3], False],
                         )
def test_error_when_max_depth_not_permitted(_max_depth):
    with pytest.raises(ValueError):
        transformer = DecisionTreeFeatures(
            variables=["Age", "Marks", "Avg_5k_run_minutes", "Height_cm"],
            output_features=3,
            regression=True,
            max_depth=_max_depth,
            random_state=0,
            drop_original=True
        )


@pytest.mark.parametrize("_random_state",
                         ["mountain", (9, 9, 9), [4, 2], True],
                         )
def test_error_when_random_state_not_permitted(_random_state):
    with pytest.raises(ValueError):
        transformer = DecisionTreeFeatures(
            variables=["Age", "Marks", "Avg_5k_run_minutes", "Height_cm"],
            output_features=3,
            regression=True,
            max_depth=3,
            random_state=_random_state,
            drop_original=True
        )


@pytest.mark.parametrize("_drop_original",
                         [36, "playa", [False, True], (True, True)],
                         )
def test_error_when_drop_original_not_permitted(_drop_original):
    with pytest.raises(ValueError):
        transformer = DecisionTreeFeatures(
            variables=["Age", "Marks", "Avg_5k_run_minutes", "Height_cm"],
            output_features=3,
            regression=True,
            max_depth=3,
            random_state=33,
            drop_original=_drop_original
        )


def test_output_features_as_integer_and_is_classification(df_creation):
    transformer = DecisionTreeFeatures(
        variables=["Age", "Marks", "Avg_5k_run_minutes"],
        output_features=3,
        regression=False,
        max_depth=3,
        random_state=0,
        drop_original=False
    )
    X = df_creation.drop("Plays_Football", axis=1)
    y = df_creation["Plays_Football"]
    transformer.fit(X, y)
    df_transformed = transformer.transform(X)
    results = df_transformed.head().round(1)

    expected_results = {
        "Name": [
            "tom",
            "nick",
            "krish",
            "megan",
            "peter"
        ],
        "City": [
            "London",
            "Manchester",
            "Liverpool",
            "Bristol",
            "Manchester",
        ],
        "Studies": [
            "Bachelor",
            "Bachelor",
            "PhD",
            "Masters",
            "Bachelor",
        ],
        "Age": [20, 44, 19, 33, 51],
        "Height_cm": [164, 150, 178, 158, 188],
        "Marks": [1.0, 0.8, 0.6, 0.1, 0.3],
        "Avg_5k_run_minutes": [22.5, 16.2, 18.3, 24.2, 20],
        "Best_40m_dash_seconds": [4.1, 5.8, 3.9, 6.2, 4.3],
        "Play_Baseball": [1, 0, 1, 0, 1],
        "Avg_100m_swim_seconds": [101, 153, 123, 201, 200],
        "Age_tree": [1, 0, 1, 0, 0],
        "Marks_tree": [1, 1, 1, 0, 0],
        "Avg_5k_run_minutes_tree": [1, 1, 1, 0, 0],
        "Age_Marks_tree": [1, 1, 1, 0, 0],
        "Age_Avg_5k_run_minutes_tree": [1, 1, 1, 0, 0],
        "Marks_Avg_5k_run_minutes_tree": [0, 1, 1, 0, 0],
        "Age_Marks_Avg_5k_run_minutes_tree": [1, 1, 1, 0, 0],
    }
    expected_results_df = pd.DataFrame(expected_results)

    assert results.equals(expected_results_df)


def test_output_features_as_list_and_is_regression(df_creation):
    X = df_creation.drop("Best_40m_dash_seconds", axis=1)
    y = df_creation["Best_40m_dash_seconds"]

    transformer = DecisionTreeFeatures(
        variables=["Age", "Avg_5k_run_minutes", "Plays_Football"],
        output_features=[2, 3],
        regression=True,
        max_depth=2,
        random_state=0,
        drop_original=True
    )
    transformer.fit(X, y)
    df_transformed = transformer.transform(X)
    results = df_transformed.head().round(1)

    expected_results = {
        "Name": [
            "tom",
            "nick",
            "krish",
            "megan",
            "peter",
        ],
        "City": [
            "London",
            "Manchester",
            "Liverpool",
            "Bristol",
            "Manchester",
        ],
        "Studies": [
            "Bachelor",
            "Bachelor",
            "PhD",
            "Masters",
            "Bachelor",
        ],
        "Height_cm": [164, 150, 178, 158, 188],
        "Marks": [1.0, 0.8, 0.6, 0.1, 0.3],
        "Play_Baseball": [1, 0, 1, 0, 1],
        "Avg_100m_swim_seconds": [101, 153, 123, 201, 200],
        "Age_Avg_5k_run_minutes_tree": [4.1, 5.8, 4.2, 6.7, 4.2],
        "Age_Plays_Football_tree": [4.0, 5.6, 4.0, 5.0, 5.6],
        "Avg_5k_run_minutes_Plays_Football_tree": [
            4.1,
            5.8,
            4.2,
            6.7,
            4.2
        ],
        "Age_Avg_5k_run_minutes_Plays_Football_tree": [
            4.1,
            5.8,
            4.2,
            6.7,
            4.2
        ]
    }
    expected_results_df = pd.DataFrame(expected_results)

    assert results.equals(expected_results_df)


def test_output_features_as_tuple_and_is_regression(df_creation):
    X = df_creation.drop("Best_40m_dash_seconds", axis=1)
    y = df_creation["Best_40m_dash_seconds"]

    output_features = (
        "Age",
        ("Avg_5k_run_minutes", "Plays_Football"),
        ("Age", "Marks", "Avg_5k_run_minutes", "Height_cm"),
        ("Plays_Football"),
        ("Height_cm", "Marks"),
        ("Play_Baseball", "Avg_100m_swim_seconds"),
    )

    transformer = DecisionTreeFeatures(
        variables=None,
        output_features=output_features,
        regression=True,
        max_depth=3,
        random_state=0,
        drop_original=True
    )
    transformer.fit(X, y)
    df_transformed = transformer.transform(X)
    results = df_transformed.head().round(1)

    expected_results = {
        "Name": [
            "tom",
            "nick",
            "krish",
            "megan",
            "peter",
        ],
        "City": [
            "London",
            "Manchester",
            "Liverpool",
            "Bristol",
            "Manchester",
        ],
        "Studies": [
            "Bachelor",
            "Bachelor",
            "PhD",
            "Masters",
            "Bachelor",
        ],
        "Age_tree": [4.1, 5.0, 3.9, 6.2, 5.0],
        'Avg_5k_run_minutes_Plays_Football_tree': [
            4.1,
            5.8,
            4.2,
            6.4,
            4.2
        ],
        "Age_Marks_Avg_5k_run_minutes_Height_cm_tree": [
            4.1,
            6.0,
            3.9,
            6.0,
            4.3
        ],
        "Plays_Football_tree": [4.5, 4.5, 4.5, 5.7, 5.7],
        "Height_cm_Marks_tree": [4.1, 6.0, 4.0, 6.0, 4.3],
        "Play_Baseball_Avg_100m_swim_seconds_tree": [
            4.1,
            6.0,
            3.9,
            6.0,
            4.4
        ],
    }
    expected_results_df = pd.DataFrame(expected_results)

    assert results.equals(expected_results_df)
