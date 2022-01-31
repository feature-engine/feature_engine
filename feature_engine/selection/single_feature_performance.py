import warnings
from typing import List, Union

import pandas as pd
from sklearn.model_selection import cross_validate

from feature_engine.dataframe_checks import _is_dataframe
from feature_engine.selection.base_selector import BaseSelector
from feature_engine.validation import _return_tags
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_numerical_variables,
)

Variables = Union[None, int, str, List[Union[str, int]]]


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
    estimator : object
        A Scikit-learn estimator for regression or classification.

    variables: str or list, default=None
        The list of variable(s) to be evaluated.
        If None, the transformer will evaluate all numerical variables in the dataset.

    scoring: str, default='roc_auc'
        Desired metric to optimise the performance for the estimator. Comes from
        sklearn.metrics. See the model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html

    threshold: float, int, default = None
        The value that defines if a feature will be kept or removed.

        If the metric is r2, the threshold needs to be set-up within 0 and 1. If the
        metrics is the roc-auc, the threshold needs to be set-up within 0.5 and 1. For
        metrics like the mean_square_error and the root_mean_square_error the
        threshold can be a bigger number.

        The threshold can be specified by the user. If None, it will be automatically
        set to the mean performance value of all features.

    cv: int, cross-validation generator or an iterable, default=3
        Determines the cross-validation splitting strategy. Possible inputs for cv are:

            - None, to use cross_validate's default 5-fold cross validation

            - int, to specify the number of folds in a (Stratified)KFold,

            - CV splitter
                - (https://scikit-learn.org/stable/glossary.html#term-CV-splitter)

            - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and y is either binary or
        multiclass, StratifiedKFold is used. In all other cases, Fold is used. These
        splitters are instantiated with `shuffle=False` so the splits will be the same
        across calls. For more details check Scikit-learn's `cross_validate`'s
        documentation,

    Attributes
    ----------
    features_to_drop_:
        List with the features to remove from the dataset.

    feature_performance_:
        Dictionary with the single feature model performance per feature.

    variables_:
        The variables that will be considered for the feature selection.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Find the important features.
    transform:
        Reduce X to the selected features.
    fit_transform:
        Fit to data, then transform it.

    References
    ----------
    Selection based on single feature performance was used in Credit Risk modelling as
    discussed in the following talk at PyData London 2017:

    .. [1] Galli S. "Machine Learning in Financial Risk Assessment".
        https://www.youtube.com/watch?v=KHGGlozsRtA
    """

    def __init__(
        self,
        estimator,
        scoring: str = "roc_auc",
        cv=3,
        threshold: Union[int, float] = None,
        variables: Variables = None,
    ):

        if threshold:
            if not isinstance(threshold, (int, float)):
                raise ValueError("threshold can only be integer, float or None")

            if scoring == "roc_auc" and (threshold < 0.5 or threshold > 1):
                raise ValueError(
                    "roc-auc score should vary between 0.5 and 1. Pick a "
                    "threshold within this interval."
                )

            if scoring == "r2" and (threshold < 0 or threshold > 1):
                raise ValueError(
                    "r2 takes values between -1 and 1. To select features the "
                    "transformer considers the absolute value. Pick a threshold within "
                    "0 and 1."
                )

        self.variables = _check_input_parameter_variables(variables)
        self.estimator = estimator
        self.scoring = scoring
        self.threshold = threshold
        self.cv = cv

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Select features.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
           The input dataframe

        y: array-like of shape (n_samples)
           Target variable. Required to train the estimator.
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find numerical variables or check variables entered by user
        self.variables_ = _find_or_check_numerical_variables(X, self.variables)

        self.feature_performance_ = {}

        # train a model for every feature and store the performance
        for feature in self.variables_:
            model = cross_validate(
                self.estimator,
                X[feature].to_frame(),
                y,
                cv=self.cv,
                return_estimator=False,
                scoring=self.scoring,
            )

            self.feature_performance_[feature] = model["test_score"].mean()

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

        self.n_features_in_ = X.shape[1]

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)

        return X

    transform.__doc__ = BaseSelector.transform.__doc__

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["allow_nan"] = True
        tags_dict["variables"] = "numerical"
        tags_dict["requires_y"] = True
        # add additional test that fails
        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has 1 mandatory parameter"
        return tags_dict
