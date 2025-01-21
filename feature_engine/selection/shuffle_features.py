from types import GeneratorType
from typing import List, MutableSequence, Union

import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import check_cv, cross_validate
from sklearn.utils.validation import _check_sample_weight, check_random_state

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

from .base_selection_functions import _select_numerical_variables

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
class SelectByShuffling(BaseSelector):
    """
    SelectByShuffling() selects features by determining the drop in machine learning
    model performance when each feature's values are randomly shuffled.

    If the variables are important, a random permutation of their values will
    decrease dramatically the machine learning model performance. Contrarily, the
    permutation of the values should have little to no effect on the model performance
    metric we are assessing if the feature is not predictive.

    The SelectByShuffling() first trains a machine learning model utilising all
    features. Next, it shuffles the values of 1 feature, obtains a prediction with the
    pre-trained model, and determines the performance drop (if any). If the drop in
    performance is bigger than a threshold then the feature is retained, otherwise
    removed. It continues until all features have been shuffled and examined.

    The user can determine the model for which performance drop after feature shuffling
    should be assessed. The user also determines the threshold in performance under
    which a feature will be removed, and the performance metric to evaluate.

    Model training and performance calculation are done with cross-validation.

    More details in the :ref:`User Guide <feature_shuffling>`.

    Parameters
    ----------
    {estimator}

    {variables}

    {scoring}

    {threshold}

    {cv}

    random_state: int, default=None
        Controls the randomness when shuffling features.

    {confirm_variables}

    Attributes
    ----------
    {initial_model_performance_}

    performance_drifts_:
        Dictionary with the performance drift per shuffled feature.

    performance_drifts_std_:
        Dictionary with the standard deviation of performance drift per shuffled
        feature.

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

    Notes
    -----
    This transformer is a similar concept to the `permutation_importance` from
    Scikit-learn. The function in Scikit-learn is used to evaluate feature importance
    instead of to select features.

    See Also
    --------
    sklearn.inspection.permutation_importance

    Examples
    --------

    >>> import pandas as pd
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from feature_engine.selection import SelectByShuffling
    >>> X = pd.DataFrame(dict(x1 = [1000,2000,1000,1000,2000,3000],
    >>>                     x2 = [2,4,3,1,2,2],
    >>>                     x3 = [1,1,1,0,0,0],
    >>>                     x4 = [1,2,1,1,0,1],
    >>>                     x5 = [1,1,1,1,1,1]))
    >>> y = pd.Series([1,0,0,1,1,0])
    >>> sbs = SelectByShuffling(
    >>>         RandomForestClassifier(random_state=42),
    >>>         cv=2,
    >>>         random_state=42,
    >>>       )
    >>> sbs.fit_transform(X, y)
       x2  x4  x5
    0   2   1   1
    1   4   2   1
    2   3   1   1
    3   1   1   1
    4   2   0   1
    5   2   1   1
    """

    def __init__(
        self,
        estimator,
        scoring: str = "roc_auc",
        cv=3,
        threshold: Union[float, int, None] = None,
        variables: Variables = None,
        random_state: Union[int, None] = None,
        confirm_variables: bool = False,
    ):

        if threshold and not isinstance(threshold, (int, float)):
            raise ValueError("threshold can only be integer or float or None")

        super().__init__(confirm_variables)

        self.variables = _check_variables_input_value(variables)
        self.estimator = estimator
        self.scoring = scoring
        self.threshold = threshold
        self.cv = cv
        self.random_state = random_state

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Union[MutableSequence, None] = None,
    ):
        """
        Find the important features.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
           The input dataframe.

        y: array-like of shape (n_samples)
           Target variable. Required to train the estimator.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
        """

        X, y = check_X_y(X, y)

        # reset the index
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        self.variables_ = _select_numerical_variables(
            X, self.variables, self.confirm_variables
        )

        # check that there are more than 1 variable to select from
        self._check_variable_number()

        cv = list(self.cv) if isinstance(self.cv, GeneratorType) else self.cv

        # train model with all features and cross-validation
        model = cross_validate(
            estimator=self.estimator,
            X=X[self.variables_],
            y=y,
            cv=cv,
            return_estimator=True,
            scoring=self.scoring,
            params={"sample_weight": sample_weight},
        )

        # store initial model performance
        self.initial_model_performance_ = model["test_score"].mean()

        # extract the validation folds
        cv_ = check_cv(cv, y=y, classifier=is_classifier(self.estimator))
        validation_indices = [val_index for _, val_index in cv_.split(X, y)]

        # get performance metric
        scorer = get_scorer(self.scoring)

        # seed
        random_state = check_random_state(self.random_state)

        # dict to collect features and their performance_drift after shuffling
        self.performance_drifts_ = {}
        self.performance_drifts_std_ = {}

        # shuffle features and save feature performance drift into a dict
        for feature in self.variables_:

            X_shuffled = X[self.variables_].copy()

            # shuffle individual feature
            X_shuffled[feature] = (
                X_shuffled[feature]
                .sample(frac=1, random_state=random_state)
                .reset_index(drop=True)
            )

            # determine the performance with the shuffled feature
            performance = [
                scorer(m, X_shuffled.iloc[idx], y.iloc[idx])
                for m, idx in zip(model["estimator"], validation_indices)
            ]

            performance_std = np.std(performance)
            performance = np.mean(performance)

            # determine drift in performance
            # Note, sklearn negates the log and error scores, so no need to manually
            # do the inversion
            # https://scikit-learn.org/stable/modules/model_evaluation.html
            # (https://scikit-learn.org/stable/modules/model_evaluation.html
            # #the-scoring-parameter-defining-model-evaluation-rules)
            performance_drift = self.initial_model_performance_ - performance

            # Save feature and performance drift
            self.performance_drifts_[feature] = performance_drift
            self.performance_drifts_std_[feature] = performance_std

        # select features
        if not self.threshold:
            threshold = pd.Series(self.performance_drifts_).mean()
        else:
            threshold = self.threshold

        self.features_to_drop_ = [
            f
            for f in self.performance_drifts_.keys()
            if self.performance_drifts_[f] < threshold
        ]

        # save input features
        self._get_feature_names_in(X)

        return self

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        tags_dict["requires_y"] = True
        # add additional test that fails
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"
        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has 1 mandatory parameter"

        msg = "transformers need more than 1 feature to work"
        tags_dict["_xfail_checks"]["check_fit2d_1feature"] = msg

        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
