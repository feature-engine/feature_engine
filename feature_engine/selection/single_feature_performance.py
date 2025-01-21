import warnings
from typing import List, Union

import pandas as pd

from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.init_parameters.selection import (
    _confirm_variables_docstring,
)
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.selection._docstring import (
    _cv_docstring,
    _groups_docstring,
    _estimator_docstring,
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
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import check_X_y
from feature_engine.selection.base_selector import BaseSelector
from feature_engine.tags import _return_tags

from .base_selection_functions import (
    _select_numerical_variables,
    single_feature_performance,
)

Variables = Union[None, int, str, List[Union[str, int]]]


@Substitution(
    estimator=_estimator_docstring,
    scoring=_scoring_docstring,
    threshold=_threshold_docstring,
    cv=_cv_docstring,
    groups=_groups_docstring,
    variables=_variables_numerical_docstring,
    confirm_variables=_confirm_variables_docstring,
    initial_model_performance_=_initial_model_performance_docstring,
    features_to_drop_=_features_to_drop_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_docstring,
    transform=_transform_docstring,
    fit_transform=_fit_transform_docstring,
    get_support=_get_support_docstring,
)
class SelectBySingleFeaturePerformance(BaseSelector):
    """
    SelectBySingleFeaturePerformance() selects features based on the performance
    of a machine learning model trained utilising a single feature. In other
    words, it trains a machine learning model for every single feature, then determines
    each model's performance. If the performance of the model is greater than a user
    specified threshold, then the feature is retained, otherwise removed.

    The models are trained on each individual features using cross-validation.
    The performance metric to evaluate and the machine learning model to train are
    specified by the user.

    More details in the :ref:`User Guide <single_feat_performance>`.

    Parameters
    ----------
    {estimator}

    {variables}

    {scoring}

    {threshold}

    {cv}

    {groups}

    {confirm_variables}

    Attributes
    ----------
    {features_to_drop_}

    feature_performance_:
        Dictionary with the single feature model performance per feature.

    feature_performance_std_:
        Dictionary with the standard deviation of the single feature model performance.

    {variables_}

    {feature_names_in_}

    {n_features_in_}


    Methods
    -------
    {fit}

    {fit_transform}

    {get_support}

    {transform}

    References
    ----------
    Selection based on single feature performance was used in Credit Risk modelling as
    discussed in the following talk at PyData London 2017:

    .. [1] Galli S. "Machine Learning in Financial Risk Assessment".
        https://www.youtube.com/watch?v=KHGGlozsRtA

    Examples
    --------

    >>> import pandas as pd
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from feature_engine.selection import SelectBySingleFeaturePerformance
    >>> X = pd.DataFrame(dict(x1 = [1000,2000,1000,1000,2000,3000],
    >>>                     x2 = [2,4,3,1,2,2],
    >>>                     x3 = [1,1,1,0,0,0],
    >>>                     x4 = [1,2,1,1,0,1],
    >>>                     x5 = [1,1,1,1,1,1]))
    >>> y = pd.Series([1,0,0,1,1,0])
    >>> sfp = SelectBySingleFeaturePerformance(
    >>>                     RandomForestClassifier(random_state=42),
    >>>                     cv=2)
    >>> sfp.fit_transform(X, y)
        x2  x3
    0   2   1
    1   4   1
    2   3   1
    3   1   0
    4   2   0
    5   2   0
    """

    def __init__(
        self,
        estimator,
        scoring: str = "roc_auc",
        cv=3,
        groups=None,
        threshold: Union[int, float, None] = None,
        variables: Variables = None,
        confirm_variables: bool = False,
    ):

        if threshold:
            if not isinstance(threshold, (int, float)):
                raise ValueError(
                    "`threshold` can only be integer, float or None. "
                    f"Got {threshold} instead."
                )

            if scoring == "roc_auc" and (threshold < 0.5 or threshold > 1):
                raise ValueError(
                    "`threshold` for roc-auc score should be between 0.5 and 1. "
                    f"Got {threshold} instead."
                )

            if scoring == "r2" and (threshold < 0 or threshold > 1):
                raise ValueError(
                    "`threshold` for r2 score should be between 0 and 1. "
                    f"Got {threshold} instead."
                )

        super().__init__(confirm_variables)
        self.variables = _check_variables_input_value(variables)
        self.estimator = estimator
        self.scoring = scoring
        self.threshold = threshold
        self.cv = cv
        self.groups = groups

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Determines model performance based on single features. Selects features whose
        performance is above the threshold.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
           The input dataframe

        y: array-like of shape (n_samples)
           Target variable. Required to train the estimator.
        """

        # check input dataframe
        X, y = check_X_y(X, y)

        self.variables_ = _select_numerical_variables(
            X, self.variables, self.confirm_variables
        )

        if len(self.variables_) == 1 and self.threshold is None:
            raise ValueError(
                "When evaluating a single feature you need to manually set a value "
                "for the threshold. "
                f"The transformer is evaluating the performance of {self.variables_} "
                f"and the threshold was left to {self.threshold} when initializing "
                f"the transformer."
            )

        self.feature_performance_, self.feature_performance_std_ = (
            single_feature_performance(
                X=X,
                y=y,
                variables=self.variables_,
                estimator=self.estimator,
                cv=self.cv,
                groups=self.groups,
                scoring=self.scoring,
            )
        )

        # select features
        if not self.threshold:
            threshold = pd.Series(self.feature_performance_).mean()
        else:
            threshold = self.threshold

        self.features_to_drop_ = [
            f
            for f in self.feature_performance_.keys()
            if self.feature_performance_[f] < threshold
        ]

        # check we are not dropping all the columns in the df
        if len(self.features_to_drop_) == len(X.columns):
            warnings.warn("All features will be dropped, try changing the threshold.")

        # save input features
        self._get_feature_names_in(X)

        return self

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        tags_dict["requires_y"] = True
        # add additional test that fails
        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has 1 mandatory parameter"
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"

        msg = "transformers need more than 1 feature to work"
        tags_dict["_xfail_checks"]["check_fit2d_1feature"] = msg

        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
