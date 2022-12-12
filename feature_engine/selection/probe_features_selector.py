from typing import List, Union

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate

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

class ProbeFeaturesSelection(BaseSelector):
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

        # find numerical and categorical variables
        self.variables = _find_categorical_and_numerical_variables(X, None)

        # if required excluded variables that not in the input dataframe
        self._confirm_variables(X)

        # check that there are more than 1 variable
        self._check_variable_number()

        # save input features
        self._get_feature_names_in(X)

        # generate 3 random variables
        self.probe_features_ = self._generate_probe_features(X.shape[0])

        # get random variable names
        self.random_variables_ = self.probe_features_.columns

        X_new = pd.concat([X, self.probe_features_], axis=1)

        # train model with all variables including the random variables
        model = cross_validate(
            self.estimator,
            X_new,
            y,
            cv=self.n_iter,
            scoring=self.scoring,
            return_estimator=True,
        )

        # Initialize a dataframe that will contain the list of the feature/coeff
        # importance for each cross-validation fold
        feature_importances_cv = pd.DataFrame()

        # Populate the feature_importances_cv dataframe with columns containing
        # the feature importance values for each model returned by the cross
        # validation.
        # There are as many columns as folds.
        for i in range(len(model["estimator"])):
            m = model["estimator"][i]
            feature_importances_cv[i] = get_feature_importances(m)

        # add the variables as the index to feature_importances_cv
        feature_importances_cv.index = X_new.columns

        # save feature importances as an attribute
        self.feature_importances_ = feature_importances_cv

        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return a tuple comprised of the variables and the number of times
        each variable was worse than all three random variables.
        """
        pass

    def _generate_probe_features(self, n_obs: int) -> pd.DataFrame:
        """
        Returns a dataframe of 3 random variables.
        """
        # create 3 random variables
        binary_var = np.random.randint(0, 2, n_obs)
        uniform_var = np.random.uniform(0, 1, 1).tolist() * n_obs
        gaussian_var = np.random.normal(0, 3, n_obs)

        # create dataframe
        df = pd.DataFrame()
        df["rndm_binary_var"] = binary_var
        df["rndm_uniform_var"] = uniform_var
        df["rndm_gaussian_var"] = gaussian_var

        return df