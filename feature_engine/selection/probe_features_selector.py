from typing import List, Union, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate

from feature_engine.dataframe_checks import (
    check_X_y,
    _check_X_matches_training_df,
    check_X,
)
from feature_engine.selection.base_selector import BaseSelector, get_feature_importances

from feature_engine._variable_handling.variable_type_selection import (
    _find_or_check_numerical_variables,
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
        distribution: str = "normal",
        cut_rate: float = 0.2,
        cv: int = 5,
        n_iter: int = 5,
        random_state: int = 0,
        # TODO: Do we need confirm_variable given that this selector will not be used in a pipeline?
        # TODO: Do we need the parameter because it is a param of BaseSelector?
        confirm_variables: bool = False,
    ):

        if distribution not in ["normal", "binary", "unfirom"]:
            raise ValueError(
                "distribution takes on normal, binary, or uniform as values. "
                f"Got {distribution} instead."
            )

        if not (0 <= cut_rate <= 1):
            raise ValueError(
                "cut_rate must range from 0 to 1, inclusively. "
                f"Got {cut_rate} instead."
            )

        if not isinstance(cv, int) or not (cv > 0):
            raise ValueError(
                f"cv must be a positive integer. Got {cv} instead."
            )

        if not isinstance(n_iter, int) or not (n_iter > 0):
            raise ValueError(
                f"n_iter must be a positive integer. Got {n_iter} instead."
            )
        super().__init__(confirm_variables)
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state


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

        # find numerical variables
        self.variables_ = _find_or_check_numerical_variables(X, None)

        # if required excluded variables that not in the input dataframe
        self._confirm_variables(X)

        # check that there are more than 1 variable
        self._check_variable_number()

        # save input features
        self._get_feature_names_in(X)

        # generate random variable
        self.probe_feature_ = self._generate_probe_feature(X.shape[0])

        # get random variable names
        self.random_variables_ = self.probe_features_.columns

        X_new = pd.concat([X, self.probe_features_], axis=1)

        # train model with all variables including the random variables
        model = cross_validate(
            self.estimator,
            X_new,
            y,
            cv=self.cv,
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

        # count the frequencies in which the variables are less important than the probe feature
        self.vars_freq_low_importance_ = self._count_variables_importance_less_than_random_variables()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return a tuple comprised of the variables and the number of times
        each variable was worse than all three random variables.
        """
        # check if fit is performed prior to transform
        check_is_fitted(self)

        # check if input is a dataframe
        X = check_X(X)

        # check if number of columns in test dataset matches to train dataset
        _check_X_matches_training_df(X, self.n_features_in_)

        pass

    def _generate_probe_feature(self, n_obs: int) -> pd.DataFrame:
        """
        Returns a dataframe comprised of the probe feature using the user-selected distribution.
        """
        # create 3 random variables
        # create dataframe
        df = pd.DataFrame()

        if self.distribution == "normal":
            df["rndm_gaussian_var"] = np.random.normal(0, 3, n_obs)

        elif self.distribution == "binary":
            df["rndm_binary_var"] = np.random.randint(0, 2, n_obs)

        else:
            df["rndm_uniform_var"] = np.random.uniform(0, 1, 1).tolist() * n_obs

        return df

    def _count_variables_importance_less_than_random_variables(self) -> Dict:
        """
        Count how many times a variable is less than the maximum value
        of the random variables.
        """
        # create dictionary count frequency of a variable being less informative
        # than the maximum value of a random variable
        count_dict = {var: 0 for var in self.variables_}

        for col in range(0, self.cv):
            # get one iteration of the feature importances
            tmp_importances = self.feature_importances_[col]

            # identify max value of the random variables for this iteration
            max_val_rndm_vars = tmp_importances[self.random_variables_].max()

            for var in self.variables_:
                if tmp_importances < max_val_rndm_vars:
                    count_dict[var] += 1

        return count_dict

    def _get_variables_that_pass_cut_rate_threshold(self):
        """
        Returns list of variables that meet the cut-rate threshold.
        These variables have shown to be more informative than the probe feature
        based on the selected cut rate.
        """
        prcnt_freq_dict = {
            var: cnt / self.cv for var, cnt in self.vars_freq_low_importance_.items()
        }

        vars_pass_threshold = []

        for var, rate in prcnt_freq_dict:
            if rate < self.cut_rate:
                vars_pass_thershold.append(var)

        return vars_pass_thershold