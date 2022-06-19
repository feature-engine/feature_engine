
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from feature_engine.discretisation.base_discretiser import BaseDiscretiser
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.fit_attributes import (
    _variables_attribute_docstring,
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.class_inputs import _variables_numerical_docstring
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_X_matches_training_df,
    check_X,
)
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_numerical_variables,
)



class ChiMergeDiscretiser(BaseDiscretiser):
    """"

    Chi-Squared test is a statistical hypothesis test that assumes (the null hypothesis)
    that the observed frequencies for a categorical variable match the expected frequencies
    for the categorical variable.





    """
    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        threshold: Union[float, int] = 1.4,
        min_intervals: int = 2,
        max_intervals: int = 10,
        return_object: bool = False,
        return_boundaries: bool = False,
    ) -> None:

        if not isinstance(threshold, (int, float)) or threshold < 0:
            raise ValueError(
                "threshold must be a positive integer or a float. "
                f"Got {threshold} instead."
            )

        if not isinstance(min_intervals, int) or min_intervals < 2:
            raise ValueError(
                "min_intervals must be an integer that is greater than or "
                f"equal to 2. Got {min_intervals} instead."
            )

        # TODO: Should we limit max_intervals? If so, how much?
        if not isinstance(max_intervals, int) or max_intervals > 15:
            raise ValueError(
                "max_intervals must be an integer that is less than or "
                f"equal to 15. Got {max_intervals} instead."
            )
        super().__init__(return_object, return_boundaries)

        self.variables = _check_input_parameter_variables(variables)
        self.threshold = threshold
        self.min_intervals = min_intervals
        self.max_intervals = max_intervals

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Learn the limits of the intervals using the chi-square test.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset. Can be the entire dataframe, not just the variables
            to be transformed.

        y: pd.Series
            y is the predicted variables.

        """
        # check input dataframe
        X = check_X(X)
        _check_contains_na(X, self.variables)
        _check_contains_inf(X, self.variables)

        # find or check for numerical variables
        # self.variables = _find_or_check_numerical_variables(X, self.variables)

        self.frequency_matrix_intervals_, self.frequency_matrix_ = (
            self._create_frequency_matrix(X, y, self.variables)
        )
        self._perform_chi_merge()


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Sort the variable values into the intervals.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features]
            The transformed data with the discrete variables.
        """
        # check that fit method has been called
        check_is_fitted(self)

        # check if X is a dataframe
        X = check_X(X)

    # TODO: How to type hint 2 numpy arrays
    def _create_frequency_matrix(self, X: pd.DataFrame, y: pd.Series, variable: str) -> [NDArray, NDArray]:
        """
        Generates a frequency table in which the labels organized into bins.

        Parameters
        ----------
        X: pandas series = [n_samples, ]
            The data to discretised.

        y: pandas series = [n_samples, ]
            The categorical data that will be arranged in the bins.

        variable: str
            The variable used to count the frequency of the class labels.

        Returns
        -------
        contingency_table: dict
            A frequency table of the tables for each unvariable feature value.


        """
        frequency_matrix_intervals = np.sort(np.unique(X[variable]))
        unique_class_values = np.sort(np.unique(y))
        frequency_matrix = np.zeros(
            (len(frequency_matrix_intervals), len(unique_class_values))
        )

        for value, label in zip(X[variable], y):
            row_idx = np.where(frequency_matrix_intervals == value)[0][0]
            col_idx = np.where(unique_class_values == label)[0][0]
            frequency_matrix[row_idx][col_idx] += 1

        return frequency_matrix_intervals, frequency_matrix


    def _calc_chi_square(self, array: np.array) -> float:
        """
        Calculates chi-squared. Using the following equation:

        # TODO: Add chi2 formula docstring

        Parameters
        ----------
        X: np.array = [2, n_features]
            Two sequential rows from the contingency table.

        Returns
        -------
        chi2: float
            Determines whether two sets of measurements are related.
        """

        shape = array.shape
        num_obs = float(array.sum())
        rows_sums = {}
        cols_sums = {}
        chi2 = 0

        # calculate row-wise summations
        for row_idx in range(shape[0]):
            rows_sums[row_idx] = array[row_idx, :].sum()

        # calculate column-wise summations
        for col_idx in range(shape[1]):
            cols_sums[col_idx] = array[:, col_idx].sum()

        # iterate through all expect and actual value pairs.
        for row_idx in range(shape[0]):
            for col_idx in range(shape[1]):
                expected_val = rows_sums[row_idx] * cols_sums[col_idx] / num_obs
                actual_val = array[row_idx, col_idx]

                if expected_val == 0:
                    # prevents NaN error
                    chi2 += 0
                else:
                    chi2 += (actual_val - expected_val) ** 2 / float(expected_val)

        return chi2

    def _perform_chi_merge(self) -> None:
        """
        Merge adjacent distributions until the the minimum chi-square is greater than
        the threshold or the number of frequency-matrix intervals.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        chi_test = {}

        while self.frequency_matrix_.shape[0] > self.min_intervals:

            chi_test = {}
            shape = self.frequency_matrix_.shape

            for row_idx in range(0, shape[0] - 1):
                row_idx_2 = row_idx + 1
                chi2 = self._calc_chi_square(
                    self.frequency_matrix_[row_idx: row_idx_2 + 1]
                )

                if chi2 not in chi_test:
                    chi_test[chi2] = []

                chi_test[chi2].append((row_idx, row_idx_2))
                smallest = min(chi_test.keys())
                biggest = max(chi_test.keys())

            if smallest < self.threshold:
                for lower_bound, upper_bound in list(reversed(chi_test[smallest])):
                    for col_idx in range(shape[1]):
                        self.frequency_matrix_[lower_bound, col_idx] += self.frequency_matrix_[upper_bound, col_idx]
                    self.frequency_matrix_ = np.delete(self.frequency_matrix_, upper_bound, 0)
                    self.frequency_matrix_intervals_ = np.delete(self.frequency_matrix_intervals_, upper_bound, 0)
            else:
                break

