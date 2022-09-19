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
class RecursiveFeatureAddition(BaseRecursiveSelector):
    """
    RecursiveFeatureAddition() selects features following a recursive addition process.

    The process is as follows:

    1. Train an estimator using all the features.

    2. Rank the features according to their importance derived from the estimator.

    3. Train an estimator with the most important feature and determine performance.

    4. Add the second most important feature and train a new estimator.

    5. Calculate the difference in performance between estimators.

    6. If the performance increases beyond the threshold, the feature is kept.

    7. Repeat steps 4-6 until all features have been evaluated.

    Model training and performance calculation are done with cross-validation.

    More details in the :ref:`User Guide <recursive_addition>`.

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

        # Sort the feature importance values decreasingly
        self.feature_importances_.sort_values(ascending=False, inplace=True)

        # Extract most important feature from the ordered list of features
        first_most_important_feature = list(self.feature_importances_.index)[0]

        # Run baseline model using only the most important feature
        baseline_model = cross_validate(
            self.estimator,
            X[first_most_important_feature].to_frame(),
            y,
            cv=self.cv,
            scoring=self.scoring,
            return_estimator=True,
        )

        # Save baseline model performance
        baseline_model_performance = baseline_model["test_score"].mean()

        # list to collect selected features
        # It is initialized with the most important feature
        _selected_features = [first_most_important_feature]

        # dict to collect features and their performance_drift
        # It is initialized with the performance drift of
        # the most important feature
        self.performance_drifts_ = {first_most_important_feature: 0}

        # loop over the ordered list of features by feature importance starting
        # from the second element in the list.
        for feature in list(self.feature_importances_.index)[1:]:

            # Add feature and train new model
            model_tmp = cross_validate(
                self.estimator,
                X[_selected_features + [feature]],
                y,
                cv=self.cv,
                scoring=self.scoring,
                return_estimator=True,
            )

            # assign new model performance
            model_tmp_performance = model_tmp["test_score"].mean()

            # Calculate performance drift
            performance_drift = model_tmp_performance - baseline_model_performance

            # Save feature and performance drift
            self.performance_drifts_[feature] = performance_drift

            # If new performance model is
            if performance_drift > self.threshold:
                # add feature to the list of selected features
                _selected_features.append(feature)

                # Update new baseline model performance
                baseline_model_performance = model_tmp_performance

        self.features_to_drop_ = [
            f for f in self.variables_ if f not in _selected_features
        ]

        return self
