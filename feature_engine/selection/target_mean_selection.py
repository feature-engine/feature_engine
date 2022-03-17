from typing import List, Union

import pandas as pd
from sklearn.model_selection import cross_validate
from feature_engine.dataframe_checks import _is_dataframe
from feature_engine.selection.base_selector import BaseSelector
from feature_engine.validation import _return_tags
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_all_variables,
)
from feature_engine._prediction.target_mean_regressor import TargetMeanRegressor
from feature_engine._prediction.target_mean_classifier import TargetMeanClassifier

from feature_engine.docstrings import (
    Substitution,
    _feature_names_in_docstring,
    _fit_transform_docstring,
    _n_features_in_docstring,
)
from feature_engine.selection._docstring import (
    _cv_docstring,
    _features_to_drop_docstring,
    _fit_docstring,
    _scoring_docstring,
    _threshold_docstring,
    _transform_docstring,
    _variables_attribute_docstring,
)

Variables = Union[None, int, str, List[Union[str, int]]]

@Substitution(
    scoring=_scoring_docstring,
    threshold=_threshold_docstring,
    cv=_cv_docstring,
    confirm_variables=BaseSelector._confirm_variables_docstring,
    features_to_drop_=_features_to_drop_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_docstring,
    transform=_transform_docstring,
    fit_transform=_fit_transform_docstring,
)
class SelectByTargetMeanPerformance(BaseSelector):
    """
    SelectByTargetMeanPerformance() uses the mean value of the target per category, or
    interval if the variable is numerical, as proxy for target estimation. With this
    proxy and the real target, the selector determines a performance metric for each
    feature, and then selects them based on this performance metric.

    SelectByTargetMeanPerformance() works with numerical and categorical variables.
    First, it eparates the training set into train and test sets. Then it works as
    follows:

    For each categorical variable:

    1. Determines the mean target value per category using the train set.

    2. Replaces the categories in the test set by the target mean values.

    3. Using the encoded variables and the real target calculates the roc-auc or r2.

    4. Selects the features which roc-auc or r2 is bigger than the threshold.

    For each numerical variable:

    1. Discretises the variable into intervals of equal width or equal frequency.

    2. Determines the mean value of the target per interval using the train set.

    3. Replaces the intervals in the test set, by the target mean values.

    4. Using the transformed variable and the real target calculates the roc-auc or r2.

    5. Selects the features which roc-auc or r2 is bigger than the threshold.

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
        Whether to create the bins for discretization of numerical variables of
        equal width ('equal_width') or equal frequency ('equal_frequency').

    {estimator}

    {variables}

    {scoring}

    {threshold}

    {cv}

    regression: boolean, default=True
        Indicates whether the discretiser should train a regression or a classification
        decision tree.

    {confirm_variables}

    Attributes
    ----------
    {variables_}

    feature_performance_:
        Dictionary with the performance proxy per feature.

    {features_to_drop_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

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
    """

    def __init__(
        self,
        variables: Variables = None,
        bins: int = 5,
        strategy: str = "equal_width",
        scoring: str = "roc_auc_score",
        cv=3,
        threshold: Union[int, float] = None,
        regression: bool = True,
        confirm_variables: bool = False,
    ):

        if not isinstance(bins, int):
            raise ValueError(f"bins must be an integer. Got {bins} instead.")

        if strategy not in ["equal_width", "equal_frequency"]:
            raise ValueError(
                "strategy takes only values 'equal_width' or 'equal_frequency'. "
                f"Got {strategy} instead."
            )

        if threshold and not isinstance(threshold, (int, float)):
            raise ValueError(
                "threshold can only take integer or float. "
                f"Got {threshold} instead."
            )

        super().__init__(confirm_variables)
        self.variables = _check_input_parameter_variables(variables)
        self.bins = bins
        self.strategy = strategy
        self.scoring = scoring
        self.cv = cv
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
        X = _is_dataframe(X)

        # check variables
        self.variables_ = _find_all_variables(X, self.variables)

        # If required exclude variables that are not in the input dataframe
        self._confirm_variables(X)

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

        for variable in self.variables_:
            model = cross_validate(
                est,
                X[variable].to_frame(),
                y,
                cv=self.cv,
                scoring=self.scoring,
            )

            self.feature_performance_[variable] = model["test_score"].mean()

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

        return tags_dict
