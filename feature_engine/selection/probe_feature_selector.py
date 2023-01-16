from typing import List, Union, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate
from sklearn.utils.validation import check_is_fitted

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

class ProbeFeatureSelection(BaseSelector):
    """


    """
    def __init__(
        self,
        estimator,
        scoring: str = "roc_auc",
        distribution: str = "normal",
        threshold: float = 0.2,
        cv: int = 5,
        n_iter: int = 5,
        random_state: int = 0,
        # TODO: Do we need confirm_variable given that this selector will not be used in a pipeline?
        # TODO: Do we need the parameter because it is a param of BaseSelector?
        confirm_variables: bool = False,
    ):

        if distribution not in ["normal", "binary", "uniform"]:
            raise ValueError(
                "distribution takes on normal, binary, or uniform as values. "
                f"Got {distribution} instead."
            )

        if not (0 <= threshold <= 1):
            raise ValueError(
                "threshold must range from 0 to 1, inclusively. "
                f"Got {threshold} instead."
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
        self.distribution = distribution
        self.threshold = threshold
        self.cv = cv
        self.n_iter = n_iter
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
        self.variables = _find_or_check_numerical_variables(X, None)

        # if required exclude variables that are not in the input dataframe
        # instantiates the 'variables_' attribute
        self._confirm_variables(X)

        # check that there is more than 1 variable
        self._check_variable_number()

        # save input features
        self._get_feature_names_in(X)

        # create probe feature distribution
        self.probe_feature_data_ = self._generate_probe_feature(X.shape[0])

        # get probe feature name
        self.probe_feature_ = self.probe_feature_data_.columns

        # merge X and probe feature
        X_new = pd.concat([X, self.probe_feature_data_], axis=1)

        # train model with all variables including the probe feature
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
        self.vars_freq_low_importance_ = self._count_variables_importance_less_than_probe_feature()

        return self


    def _generate_probe_feature(self, n_obs: int) -> pd.DataFrame:
        """
        Returns a dataframe comprised of the probe feature using the selected distribution.
        """
        # create dataframe
        df = pd.DataFrame()

        # set random state
        np.random.seed(self.random_state)

        if self.distribution == "normal":
            df["rndm_gaussian_var"] = np.random.normal(0, 3, n_obs)

        elif self.distribution == "binary":
            df["rndm_binary_var"] = np.random.randint(0, 2, n_obs)

        else:
            df["rndm_uniform_var"] = np.random.uniform(0, 1, 1).tolist() * n_obs

        return df

    def _count_variables_importance_less_than_probe_feature(self) -> Dict:
        """
        Count the frequency in which the variables' importances are less than
        the probe feature's importance.
        """
        # create dictionary to count frequency of a variable being less informative
        # than the probe feature
        count_dict = {var: 0 for var in self.variables_}

        for col in range(0, self.n_iter):
            # get one iteration of the feature importances
            tmp_importances = self.feature_importances_[col]

            # Get the probe feature's importance
            probe_feature_importance = tmp_importances[self.probe_feature_].values

            for var in self.variables_:

                # exclude the probe feature
                if var != self.probe_feature_:

                    # count the frequency in which variables' importances are less
                    # than the probe feature's importance
                    if tmp_importances[var] < probe_feature_importance:
                        count_dict[var] += 1

        return count_dict

    def _get_variables_that_pass_threshold_threshold(self):
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
            if rate < self.threshold:
                vars_pass_threshold.append(var)

        return vars_pass_threshold