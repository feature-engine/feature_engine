import pandas as pd
import pytest
from sklearn import datasets

from feature_engine.discretisation import ChiMergeDiscretiser

# TODO: Should we create the df here on in conftest?

# create dataset for unit tests
col_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
iris_data = datasets.load_iris().data
iris = pd.DataFrame(iris_data, columns=col_names)
iris["flower"] = datasets.load_iris().target


def test_create_frequency_matrix():
    transformer = ChiMergeDiscretiser(
        variables="sepal_length",
        threshold=1.4,
        min_intervals=2,
        max_intervals=10,
        return_object=False,
        return_boundaries=False,
    )

    frequency_matrix = transformer._create_frequency_matrix(
        X=iris[["sepal_length", "sepal_width", "petal_length"]],
        y=iris["flower"],
        variable="sepal_length"
    )

    # number of flowers included in contingency table
    table_flower_count = 0
    for count_arr in frequency_matrix.values():
        table_flower_count += sum(count_arr)

    # expected results
    expected_results = {
        4.3: [1, 0, 0],
        4.4: [3, 0, 0],
        4.5: [1, 0, 0],
        4.6: [4, 0, 0],
        4.7: [2, 0, 0],
        4.8: [5, 0, 0],
        4.9: [4, 1, 1],
        5.0: [8, 2, 0],
        5.1: [8, 1, 0],
        5.2: [3, 1, 0],
        5.3: [1, 0, 0],
        5.4: [5, 1, 0],
        5.5: [2, 5, 0],
        5.6: [0, 5, 1],
        5.7: [2, 5, 1],
        5.8: [1, 3, 3],
        5.9: [0, 2, 1],
        6.0: [0, 4, 2],
        6.1: [0, 4, 2],
        6.2: [0, 2, 2],
        6.3: [0, 3, 6],
        6.4: [0, 2, 5],
        6.5: [0, 1, 4],
        6.6: [0, 2, 0],
        6.7: [0, 3, 5],
        6.8: [0, 1, 2],
        6.9: [0, 1, 3],
        7.0: [0, 1, 0],
        7.1: [0, 0, 1],
        7.2: [0, 0, 3],
        7.3: [0, 0, 1],
        7.4: [0, 0, 1],
        7.6: [0, 0, 1],
        7.7: [0, 0, 4],
        7.9: [0, 0, 1]
    }
    num_flowers = iris.shape[0]

    # check results
    assert frequency_matrix == expected_results
    # confirm all flowers are included
    assert table_flower_count == num_flowers


def test_chi_merge():

    transformer = ChiMergeDiscretiser(
        variables="sepal_length",
        threshold=1.4,
        min_intervals=2,
        max_intervals=10,
        return_object=False,
        return_boundaries=False,
    )

    transformer.fit(
        iris[["sepal_length", "sepal_width", "petal_length"]], iris["flower"]
    )

    chi_test = transformer._perform_chi_merge()
    chi_scores_round = pd.Series(chi_test.keys()).round(1)
    expected_results = pd.Series(
        [4.1, 2.4, 8.6, 2.9, 1.7, 1.8, 2.2, 4.8, 4.1, 3.2, 1.5, 3.6]
    )

    assert (chi_scores_round == expected_results).all()