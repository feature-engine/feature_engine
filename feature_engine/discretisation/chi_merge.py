
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

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series]):
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