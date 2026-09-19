from typing import Any, Dict

import narwhals as nw
import narwhals.dependencies as nwd
import numpy as np
from narwhals.typing import IntoDataFrame, IntoSeries
from sklearn.model_selection import cross_validate

from feature_engine._docstrings.fit_attributes import (
    _feature_importances_docstring,
    _feature_importances_std_docstring,
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
    feature_importances_=_feature_importances_docstring,
    feature_importances_std_=_feature_importances_std_docstring,
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

    {feature_importances_}

    {feature_importances_std_}

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
    >>> from feature_engine.selection import RecursiveFeatureAddition
    >>> X = pd.DataFrame(dict(x1 = [1000,2000,1000,1000,2000,3000],
    >>>                     x2 = [2,4,3,1,2,2],
    >>>                     x3 = [1,1,1,0,0,0],
    >>>                     x4 = [1,2,1,1,0,1],
    >>>                     x5 = [1,1,1,1,1,1]))
    >>> y = pd.Series([1,0,0,1,1,0])
    >>> rfa = RecursiveFeatureAddition(RandomForestClassifier(random_state=42), cv=2)
    >>> rfa.fit_transform(X, y)
       x2  x4
    0   2   1
    1   4   2
    2   3   1
    3   1   1
    4   2   0
    5   2   1

    With polars:

    >>> import polars as pl
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from feature_engine.selection import RecursiveFeatureAddition
    >>> X = pl.DataFrame(dict(x1 = [1000,2000,1000,1000,2000,3000],
    >>>                     x2 = [2,4,3,1,2,2],
    >>>                     x3 = [1,1,1,0,0,0],
    >>>                     x4 = [1,2,1,1,0,1],
    >>>                     x5 = [1,1,1,1,1,1]))
    >>> y = pl.Series([1,0,0,1,1,0])
    >>> rfa = RecursiveFeatureAddition(RandomForestClassifier(random_state=42), cv=2)
    >>> rfa.fit_transform(X, y)
    shape: (6, 2)
    ┌─────┬─────┐
    │ x2  ┆ x4  │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 2   ┆ 1   │
    │ 4   ┆ 2   │
    │ 3   ┆ 1   │
    │ 1   ┆ 1   │
    │ 2   ┆ 0   │
    │ 2   ┆ 1   │
    └─────┴─────┘
    """

    def fit(self, X: IntoDataFrame, y: IntoSeries):
        """
        Find the important features. Note that the selector trains various models at
        each round of selection, so it might take a while.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
           The input dataframe.

        y: array-like of shape (n_samples)
           Target variable. Required to train the estimator.
        """

        nw_X, y = super().fit(X, y)

        importances = dict(self.feature_importances_)
        features = list(importances)
        values = np.array(list(importances.values()))
        # Same order as pandas' sort_values(ascending=False), which sorts the reversed
        # values, so features with the same importance are ranked as before.
        order = (len(values) - 1 - values[::-1].argsort(kind="quicksort"))[::-1]
        ranked_features = [features[i] for i in order]
        self.feature_importances_ = _importance_series(
            X, ranked_features, values[order]
        )

        first_most_important_feature = ranked_features[0]
        baseline_scores = self._cross_validate(
            X, nw_X, y, [first_most_important_feature]
        )
        baseline_model_performance = baseline_scores.mean()

        _selected_features = [first_most_important_feature]
        self.performance_drifts_: Dict[Any, float] = {first_most_important_feature: 0}
        self.performance_drifts_std_: Dict[Any, float] = {
            first_most_important_feature: 0
        }

        for feature in ranked_features[1:]:
            scores = self._cross_validate(X, nw_X, y, _selected_features + [feature])
            model_tmp_performance = scores.mean()
            performance_drift = model_tmp_performance - baseline_model_performance

            self.performance_drifts_[feature] = performance_drift
            self.performance_drifts_std_[feature] = scores.std()

            if performance_drift > self.threshold:
                _selected_features.append(feature)
                baseline_model_performance = model_tmp_performance

        self.features_to_drop_ = [
            f for f in self.variables_ if f not in _selected_features
        ]

        return self

    def _cross_validate(
        self, X: IntoDataFrame, nw_X: nw.DataFrame, y: IntoSeries, features: list
    ) -> np.ndarray:
        """Return the test scores of the estimator trained on the features."""
        if nwd.is_pandas_dataframe(X) is True:
            # pandas is faster than narwhals.
            X_model = X[features]
        else:
            X_model = nw_X.select(nw.col(*features)).to_native()

        return cross_validate(
            estimator=self.estimator,
            X=X_model,
            y=y,
            cv=self._cv,
            groups=self.groups,
            scoring=self.scoring,
        )["test_score"]
