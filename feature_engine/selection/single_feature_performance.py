import warnings
from typing import List, Union

import pandas as pd
from sklearn.model_selection import cross_validate

from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.init_parameters.selection import (
    _confirm_variables_docstring,
)
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import check_X_y
from feature_engine._docstrings.selection._docstring import (
    _cv_docstring,
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
from feature_engine.selection.base_selector import BaseSelector
from feature_engine.tags import _return_tags
from feature_engine.variable_handling._init_parameter_checks import (
    _check_init_parameter_variables,
)
from feature_engine.variable_handling.variable_type_selection import (
    find_or_check_numerical_variables,
)

Variables = Union[None, int, str, List[Union[str, int]]]


@Substitution(
    estimator=_estimator_docstring,
    scoring=_scoring_docstring,
    threshold=_threshold_docstring,
    cv=_cv_docstring,
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

    {confirm_variables}

    Attributes
    ----------
    {features_to_drop_}

    feature_performance_:
        Dictionary with the single feature model performance per feature.

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
        threshold: Union[int, float, None] = None,
        variables: Variables = None,
        confirm_variables: bool = False,
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

        super().__init__(confirm_variables)
        self.variables = _check_init_parameter_variables(variables)
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
        X, y = check_X_y(X, y)

        # If required exclude variables that are not in the input dataframe
        self._confirm_variables(X)

        # find numerical variables or check variables entered by user
        self.variables_ = find_or_check_numerical_variables(X, self.variables_)

        if len(self.variables_) == 1 and self.threshold is None:
            raise ValueError(
                "When evaluating a single feature you need to manually set a value "
                "for the threshold. "
                f"The transformer is evaluating the performance of {self.variables_} "
                f"and the threshold was left to {self.threshold} when initializing "
                f"the transformer."
            )

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
