import narwhals as nw
import narwhals.dependencies as nwd
import numpy as np
from narwhals.typing import IntoDataFrame, IntoSeries
from sklearn.model_selection import cross_validate

from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _performance_drifts_docstring,
    _performance_drifts_std_docstring,
)
from feature_engine._docstrings.init_parameters.selection import (
    _confirm_variables_docstring,
)
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.selection._docstring import (
    _cv_docstring,
    _features_to_drop_docstring,
    _fit_docstring,
    _get_support_docstring,
    _groups_docstring,
    _initial_model_performance_docstring,
    _scoring_docstring,
    _threshold_docstring,
    _transform_docstring,
    _variables_attribute_docstring,
    _variables_numerical_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.selection.base_recursive_selector import BaseRecursiveSelector
from feature_engine.selection.base_selection_functions import _importance_series


@Substitution(
    scoring=_scoring_docstring,
    threshold=_threshold_docstring,
    cv=_cv_docstring,
    groups=_groups_docstring,
    variables=_variables_numerical_docstring,
    confirm_variables=_confirm_variables_docstring,
    initial_model_performance_=_initial_model_performance_docstring,
    performance_drifts_=_performance_drifts_docstring,
    performance_drifts_std_=_performance_drifts_std_docstring,
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
    estimator: object
        A scikit-learn estimator for regression or classification.

    {variables}

    {scoring}

    {threshold}

    {cv}

    {groups}

    {confirm_variables}

    Attributes
    ----------
    {initial_model_performance_}

    feature_importances_:
        The feature importance (comes from step 2), sorted from the least to the
        most important feature. A pandas Series with the features as index when X
        is a pandas dataframe, and a dictionary with the features as keys otherwise.

    feature_importances_std_:
        The standard deviation of the feature importance, as a pandas Series or a
        dictionary, like `feature_importances_`.

    {performance_drifts_}

    {performance_drifts_std_}

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

    Examples
    --------

    >>> import pandas as pd
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from feature_engine.selection import RecursiveFeatureElimination
    >>> X = pd.DataFrame(dict(x1 = [1000,2000,1000,1000,2000,3000],
    >>>                     x2 = [2,4,3,1,2,2],
    >>>                     x3 = [1,1,1,0,0,0],
    >>>                     x4 = [1,2,1,1,0,1],
    >>>                     x5 = [1,1,1,1,1,1]))
    >>> y = pd.Series([1,0,0,1,1,0])
    >>> rfe = RecursiveFeatureElimination(RandomForestClassifier(random_state=2), cv=2)
    >>> rfe.fit_transform(X, y)
       x2
    0   2
    1   4
    2   3
    3   1
    4   2
    5   2

    The same with polars:

    >>> import polars as pl
    >>> rfe.fit_transform(pl.DataFrame(X.to_dict(orient="list")), y.to_list())
    shape: (6, 1)
    ┌─────┐
    │ x2  │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 2   │
    │ 4   │
    │ 3   │
    │ 1   │
    │ 2   │
    │ 2   │
    └─────┘
    """

    def fit(self, X: IntoDataFrame, y: IntoSeries):
        """
        Find the important features. Note that the selector trains various models at
        each round of selection, so it might take a while.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
           The input dataframe
        y: array-like of shape (n_samples)
           Target variable. Required to train the estimator.
        """

        nw_X, y = super().fit(X, y)

        if nwd.is_pandas_dataframe(X) is True:
            # pandas is faster than narwhals.
            self.feature_importances_ = self.feature_importances_.sort_values()
        else:
            # numpy's default quicksort is the one of pandas' sort_values, so tied
            # features are evaluated in the same order with every backend.
            features = list(self.feature_importances_.keys())
            values = np.array(list(self.feature_importances_.values()))
            order = values.argsort()
            self.feature_importances_ = _importance_series(
                X, [features[i] for i in order], values[order]
            )

        # to collect selected features
        _selected_features = []

        # features left in the model as we remove them recursively
        remaining_features = list(self.variables_)

        # we need to update the performance as we remove features
        baseline_model_performance = self.initial_model_performance_

        # dict to collect features and their performance_drift after shuffling
        self.performance_drifts_ = {}
        self.performance_drifts_std_ = {}

        # evaluate every feature, starting from the least important
        for feature in self.feature_importances_.keys():

            # if there is only 1 feature left
            if len(remaining_features) == 1:
                self.performance_drifts_[feature] = 0
                _selected_features.append(feature)
                break

            # remove feature and train new model
            features_tmp = [f for f in remaining_features if f != feature]
            if nwd.is_pandas_dataframe(X) is True:
                # pandas is faster than narwhals.
                X_tmp = X[features_tmp]
            else:
                X_tmp = nw_X.select(nw.col(*features_tmp)).to_native()

            model_tmp = cross_validate(
                estimator=self.estimator,
                X=X_tmp,
                y=y,
                cv=self._cv,
                groups=self.groups,
                scoring=self.scoring,
                return_estimator=False,
            )

            # assign new model performance
            model_tmp_performance = model_tmp["test_score"].mean()

            # Calculate performance drift
            performance_drift = baseline_model_performance - model_tmp_performance

            # Save feature and performance drift
            self.performance_drifts_[feature] = performance_drift
            self.performance_drifts_std_[feature] = model_tmp["test_score"].std()

            if performance_drift > self.threshold:

                _selected_features.append(feature)

            else:
                # remove feature and adjust initial performance
                remaining_features = features_tmp

                baseline_model = cross_validate(
                    estimator=self.estimator,
                    X=X_tmp,
                    y=y,
                    cv=self._cv,
                    groups=self.groups,
                    return_estimator=False,
                    scoring=self.scoring,
                )

                # store initial model performance
                baseline_model_performance = baseline_model["test_score"].mean()

        self.features_to_drop_ = [
            f for f in self.variables_ if f not in _selected_features
        ]

        return self
