import pandas as pd
from sklearn.model_selection import cross_validate

from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.substitute import Substitution
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
from feature_engine.selection.base_recursive_selector import BaseRecursiveSelector


@Substitution(
    estimator=BaseRecursiveSelector._estimator_docstring,
    scoring=_scoring_docstring,
    threshold=_threshold_docstring,
    cv=_cv_docstring,
    variables=_variables_numerical_docstring,
    confirm_variables=BaseRecursiveSelector._confirm_variables_docstring,
    initial_model_performance_=_initial_model_performance_docstring,
    feature_importances_=BaseRecursiveSelector._feature_importances_docstring,
    performance_drifts_=BaseRecursiveSelector._performance_drifts_docstring,
    features_to_drop_=_features_to_drop_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_docstring,
    transform=_transform_docstring,
    fit_transform=_fit_transform_docstring,
    get_support=_get_support_docstring,
)
class RecursiveFeatureElimination(BaseRecursiveSelector):
    """
    RecursiveFeatureElimination() selects features following a recursive elimination
    process.

    The process is as follows:

    1. Train an estimator using all the features.

    2. Rank the features according to their importance derived from the estimator.

    3. Remove the least important feature and fit a new estimator.

    4. Calculate the performance of the new estimator.

    5. Calculate the performance difference between the new and original estimator.

    6. If the performance drop is below the threshold the feature is removed.

    7. Repeat steps 3-6 until all features have been evaluated.

    Model training and performance evaluation are done with cross-validation.

    More details in the :ref:`User Guide <recursive_elimination>`.

    Parameters
    ----------
    {estimator}

    {variables}

    {scoring}

    {threshold}

    {cv}

    {confirm_variables}

    Attributes
    ----------
    {initial_model_performance_}

    {feature_importances_}

    {performance_drifts_}

    {features_to_drop_}

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

    {get_support}

    {transform}

    """

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Find the important features. Note that the selector trains various models at
        each round of selection, so it might take a while.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
           The input dataframe
        y: array-like of shape (n_samples)
           Target variable. Required to train the estimator.
        """

        X, y = super().fit(X, y)

        # Sort the feature importance values increasingly
        self.feature_importances_.sort_values(ascending=True, inplace=True)

        # to collect selected features
        _selected_features = []

        # temporary copy where we will remove features recursively
        X_tmp = X[self.variables_].copy()

        # we need to update the performance as we remove features
        baseline_model_performance = self.initial_model_performance_

        # dict to collect features and their performance_drift after shuffling
        self.performance_drifts_ = {}

        # evaluate every feature, starting from the least important
        # remember that feature_importances_ is ordered already
        for feature in list(self.feature_importances_.index):

            # remove feature and train new model
            model_tmp = cross_validate(
                self.estimator,
                X_tmp.drop(columns=feature),
                y,
                cv=self.cv,
                scoring=self.scoring,
                return_estimator=False,
            )

            # assign new model performance
            model_tmp_performance = model_tmp["test_score"].mean()

            # Calculate performance drift
            performance_drift = baseline_model_performance - model_tmp_performance

            # Save feature and performance drift
            self.performance_drifts_[feature] = performance_drift

            if performance_drift > self.threshold:

                _selected_features.append(feature)

            else:
                # remove feature and adjust initial performance
                X_tmp = X_tmp.drop(columns=feature)

                if X_tmp.empty is True:
                    raise ValueError(
                        "All features have been removed. Try reducing the threshold."
                    )

                baseline_model = cross_validate(
                    self.estimator,
                    X_tmp,
                    y,
                    cv=self.cv,
                    return_estimator=False,
                    scoring=self.scoring,
                )

                # store initial model performance
                baseline_model_performance = baseline_model["test_score"].mean()

        self.features_to_drop_ = [
            f for f in self.variables_ if f not in _selected_features
        ]

        return self
