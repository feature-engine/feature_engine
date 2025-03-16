from types import GeneratorType
from typing import List, Union

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_validate

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
    _features_to_drop_docstring,
    _fit_docstring,
    _get_support_docstring,
    _groups_docstring,
    _scoring_docstring,
    _threshold_docstring,
    _transform_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine._prediction.target_mean_classifier import TargetMeanClassifier
from feature_engine._prediction.target_mean_regressor import TargetMeanRegressor
from feature_engine.dataframe_checks import check_X_y
from feature_engine.selection._selection_constants import (
    _CLASSIFICATION_METRICS,
    _REGRESSION_METRICS,
)
from feature_engine.selection.base_selector import BaseSelector
from feature_engine.tags import _return_tags

from .base_selection_functions import _select_all_variables

Variables = Union[None, int, str, List[Union[str, int]]]


@Substitution(
    scoring=_scoring_docstring,
    threshold=_threshold_docstring,
    cv=_cv_docstring,
    groups=_groups_docstring,
    confirm_variables=_confirm_variables_docstring,
    features_to_drop_=_features_to_drop_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_docstring,
    transform=_transform_docstring,
    fit_transform=_fit_transform_docstring,
    get_support=_get_support_docstring,
)
class SelectByTargetMeanPerformance(BaseSelector):
    """
    SelectByTargetMeanPerformance() uses the mean value of the target per category or
    per interval(if the variable is numerical), as proxy for target estimation. With
    this proxy, the selector determines the performance of each feature based on a
    metric of choice, and then selects the features based on this performance value.

    SelectByTargetMeanPerformance() can evaluate numerical and categorical variables,
    without much prior manipulation. In other words, you don't need to encode the
    categorical variables or transform the numerical variables to assess their
    importance if you use this transformer.

    SelectByTargetMeanPerformance() requires that the dataset is complete, without
    missing data.

    SelectByTargetMeanPerformance() determines the performance of each variable with
    cross-validation. More specifically:

    For each categorical variable:

    1. Determines the mean target value per category in the training folds.

    2. Replaces the categories by the target mean values in the test folds.

    3. Determines the performance of the transformed variables in the test folds.


    For each numerical variable:

    1. Discretises the variable into intervals of equal width or equal frequency.

    2. Determines the mean value of the target per interval in the training folds.

    3. Replaces the intervals by the target mean values in the test fold.

    4. Determines the performance of the transformed variable in the test fold.


    Finally, it selects the features which performance is bigger than the indicated
    threshold. If the threshold if left to None, it selects features which performance
    is bigger than the mean performance of all features.

    All the steps are performed with cross-validation. That means, that intervals and
    target mean values per interval or category are determined in a certain portion of
    the data, and evaluated in a left-out sample. The performance metric per variable
    is the average across the cross-validation folds.

    More details in the :ref:`User Guide <target_mean_selection>`.

    Parameters
    ----------
    variables: list, default=None
        The list of variables to evaluate. If None, the transformer will evaluate all
        variables in the dataset (except datetime).

    bins: int, default = 5
        If the dataset contains numerical variables, the number of bins into which
        the values will be sorted.

    strategy: str, default = 'equal_width'
        Whether the bins should be of equal width ('equal_width') or equal frequency
        ('equal_frequency').

    {scoring}

    {threshold}

    {cv}

    {groups}

    regression: boolean, default=False
        Indicates whether the target is one for regression or a classification.

    {confirm_variables}

    Attributes
    ----------
    {variables_}

    feature_performance_:
        Dictionary with the performance of each feature.

    feature_performance_std_:
        Dictionary with the standard deviation of each feature's performance.

    {features_to_drop_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

    {get_support}

    {transform}

    Notes
    -----
    Replacing the categories or intervals by the target mean is the equivalent to
    target mean encoding.

    See Also
    --------
    feature_engine.encoding.MeanEncoder
    feature_engine.discretisation.EqualWidthDiscretiser
    feature_engine.discretisation.EqualFrequencyDiscretiser

    References
    ----------
    .. [1] Miller, et al. "Predicting customer behaviour: The University of Melbourne’s
        KDD Cup report". JMLR Workshop and Conference Proceeding. KDD 2009
        http://proceedings.mlr.press/v7/miller09/miller09.pdf

    Examples
    --------

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from feature_engine.selection import SelectByTargetMeanPerformance
    >>> X = pd.DataFrame(dict(x1 = [1000,2000,1000,1000,2000,3000],
    >>>                     x2 = [1,1,1,0,0,0],
    >>>                     x3 = [1,2,1,1,0,1],
    >>>                     x4 = [1,1,1,1,1,1]))
    >>> y = pd.Series([1,0,0,1,1,0])
    >>> tmp = SelectByTargetMeanPerformance(bins = 3, cv=2,scoring='accuracy')
    >>> tmp.fit_transform(X, y)
        x2  x3  x4
    0   1   1   1
    1   1   2   1
    2   1   1   1
    3   0   1   1
    4   0   0   1
    5   0   1   1

    This transformer also works with Categorical examples:

    >>> X = pd.DataFrame(dict(x1 = ["a","b","a","a","b","b"],
    >>>             x2 = ["a","a","a","b","b","b"]))
    >>> y = pd.Series([1,0,0,1,1,0])
    >>> tmp = SelectByTargetMeanPerformance(bins = 3, cv=2,scoring='accuracy')
    >>> tmp.fit_transform(X, y)
      x2
    0  a
    1  a
    2  a
    3  b
    4  b
    5  b
    """

    def __init__(
        self,
        variables: Variables = None,
        bins: int = 5,
        strategy: str = "equal_width",
        scoring: str = "roc_auc",
        cv=3,
        groups=None,
        threshold: Union[int, float, None] = None,
        regression: bool = False,
        confirm_variables: bool = False,
    ):

        if not isinstance(bins, int):
            raise ValueError(f"bins must be an integer. Got {bins} instead.")

        if strategy not in ["equal_width", "equal_frequency"]:
            raise ValueError(
                "strategy takes only values 'equal_width' or 'equal_frequency'. "
                f"Got {strategy} instead."
            )

        if threshold is not None and not isinstance(threshold, (int, float)):
            raise ValueError(
                "threshold can only take integer or float. " f"Got {threshold} instead."
            )

        if regression is True and scoring not in _REGRESSION_METRICS:
            raise ValueError(
                f"The metric {scoring} is not suitable for regression. Set the "
                "parameter regression to False or choose a different performance "
                "metric."
            )

        if regression is False and scoring not in _CLASSIFICATION_METRICS:
            raise ValueError(
                f"The metric {scoring} is not suitable for classification. Set the"
                "parameter regression to True or choose a different performance "
                "metric."
            )

        super().__init__(confirm_variables)
        self.variables = _check_variables_input_value(variables)
        self.bins = bins
        self.strategy = strategy
        self.scoring = scoring
        self.cv = cv
        self.groups = groups
        self.threshold = threshold
        self.regression = regression

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Find the important features.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
           The input dataframe.

        y: array-like of shape (n_samples)
           Target variable. Required to train the estimator.
        """
        # check input dataframe
        X, y = check_X_y(X, y)

        self.variables_ = _select_all_variables(
            X, self.variables, self.confirm_variables, exclude_datetime=True
        )

        if len(self.variables_) == 1 and self.threshold is None:
            raise ValueError(
                "When evaluating a single feature you need to manually set a value "
                "for the threshold. "
                f"The transformer is evaluating the performance of {self.variables_} "
                f"and the threshold was left to {self.threshold} when initializing "
                f"the transformer."
            )

        # save input features
        self._get_feature_names_in(X)

        # set up the correct estimator
        if self.regression is True:
            est = TargetMeanRegressor(
                bins=self.bins,
                strategy=self.strategy,
            )
        else:
            est = TargetMeanClassifier(
                bins=self.bins,
                strategy=self.strategy,
            )

        self.feature_performance_ = {}
        self.feature_performance_std_ = {}

        cv = list(self.cv) if isinstance(self.cv, GeneratorType) else self.cv

        for variable in self.variables_:
            # clone estimator
            estimator = clone(est)

            # set the estimator to evaluate the required variable
            estimator.set_params(variables=variable)

            model = cross_validate(
                estimator=estimator,
                X=X,
                y=y,
                cv=cv,
                groups=self.groups,
                scoring=self.scoring,
            )

            self.feature_performance_[variable] = model["test_score"].mean()
            self.feature_performance_std_[variable] = model["test_score"].std()

        # select features
        if not self.threshold:
            threshold = pd.Series(self.feature_performance_).mean()
        else:
            threshold = self.threshold

        self.features_to_drop_ = [
            f for f in self.variables_ if self.feature_performance_[f] < threshold
        ]

        return self

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "all"
        tags_dict["requires_y"] = True
        tags_dict["binary_only"] = True
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"
        msg = "transformers need more than 1 feature to work"
        tags_dict["_xfail_checks"]["check_fit2d_1feature"] = msg
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
