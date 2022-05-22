
from typing import List, Optional, Union

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
        threshold: float = 0.9,
        min_intervals: int = 2,
        max_intervals: int = 10,
        return_object: bool = False,
        return_boundaries: bool = False,
    ) -> None:

        if not isinstance(threshold, float) or threshold >= 1:
            raise ValueError(
                "threshold must be a float and less than one. "
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
        super().__init(return_object, return_boundaries)

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
        y: None
            y is not needed in this encoder. You can pass y or None.

        """
        # check input dataframe
        X = check_X(X)
        _check_contains_na(X)
        _check_contains_inf(X)

        # find or check for numerical variables
        self.variables = _find_or_check_numerical_variables(X, self.variables)


         pass

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



        pass

    def _create_contingency_table(self, feature: pd.Series, class_labels: pd.Series):
        """
        Generates a frequency table in which the labels organized into bins.

        Parameters
        ----------
        feature: pandas series = [n_samples, ]
            The data to discretised.

        class_labels: pandas series = [n_samples, ]
            The categorical data that will be arranged in the bins.

        Returns
        -------
        TBD


        """

        unique_values = sorted(set(feature), reverse=False)
        unique_labels = sorted(set(class_labels))
        count_dict = {label: 0 for label in unique_labels}
        zeros = [0 for i in range(len(unique_labels))]
        frequency_table = {val: zeros for val in unique_values}

        for feature_val, label_val in zip(feature, class_labels):
            print(feature_val)
            for idx, interval_key in enumerate(frequency_table.keys()):
                min_interval = list(frequency_table.keys())[idx]
                max_interval = list(frequency_table.keys())[idx + 1]
                table_col_index = unique_labels.index(label_val)

                print(idx, min_interval, max_interval)
                if interval_key == max(unique_values):
                    frequency_table[interval_key][label_val] += 1
                    print(feature_val, label_val, min_interval, max_interval, table_col_index)
                    print(frequency_table)
                    break

                if min_interval <= feature_val and feature_val < max_interval:
                    print(feature_val, label_val, min_interval, max_interval, table_col_index)
                    frequency_table[min_interval][label_val] += 1
                    print(frequency_table)
                    break


    def _calc_chi_sqaure(self):


