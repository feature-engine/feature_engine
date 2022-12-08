from typing import List, Union

import numpy as np
import pandas as pd


from feature_engine.dataframe_checks import check_X_y
from feature_engine.selection.base_selector import BaseSelector, get_feature_importances


from feature_engine._variable_handling.variable_type_selection import (
    _find_all_variables,
    _find_categorical_and_numerical_variables,
)

from feature_engine.selection._docstring import (
    _cv_docstring,
    _features_to_drop_docstring,
    _fit_docstring,
    _get_support_docstring,
    _initial_model_performance_docstring,
    _scoring_docstring,
    _threshold_docstring,
    _transform_docstring,
    _variables_attribute_docstring,
    _variables_numerical_docstring,
)

class ProbeFeatureSelection(BaseSelector):
    """


    """
    def __init__(
        self,
        estimator,
        scoring: str = "roc_auc",
        n_iter: int = 10,
        seed: int = 0,
        # TODO: Do we need confirm_variable given that this selector will not be used in a pipeline?
        # TODO: Do we need the parameter because it is a param of BaseSelector?
        confirm_variables: bool = False,
    ):

        super().__init__(confirm_variables)
        self.estimator = estimator
        self.scoring = scoring
        self.n_iter = n_iter
        self.seed = seed


    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Create three random feature. Find initial model performance.
        Sort features by importance.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
           The input dataframe
        y: array-like of shape (n_samples)
           Target variable. Required to train the estimator.
        """
        # check input dataframe
        X, y = check_X_y(X, y)

        # if required excluded variables that not in the input dataframe
        self._confirm_variables(X)

        # find numerical and categorical variables
        self.variables_ = _find_categorical_and_numerical_variables(X, None)

        # check that there are more than 1 variable
        self._check_variable_number()

        # save input efatures
        pass


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return a tuple comprised of the variables and the number of times
        each variable was worse than all three random variables.
        """
        pass