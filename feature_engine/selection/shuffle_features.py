from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_validate
from sklearn.utils.validation import check_random_state

from feature_engine.dataframe_checks import _is_dataframe
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_numerical_variables,
)
from feature_engine.selection.base_selector import BaseSelector

Variables = Union[None, int, str, List[Union[str, int]]]


class SelectByShuffling(BaseSelector):
    """
    SelectByShuffling() selects features by determining the drop in machine learning
    model performance when each feature's values are randomly shuffled.

    If the variables are important, a random permutation of their values will
    decrease dramatically the machine learning model performance. Contrarily, the
    permutation of the values should have little to no effect on the model performance
    metric we are assessing.

    The SelectByShuffling() first trains a machine learning model utilising all
    features. Next, it shuffles the values of 1 feature, obtains a prediction with the
    pre-trained model, and determines the performance drop (if any). If the drop in
    performance is bigger than a threshold then the feature is retained, otherwise
    removed. It continues until all features have been shuffled and the drop in
    performance evaluated.

    The user can determine the model for which performance drop after feature shuffling
    should be assessed. The user also determines the threshold in performance under
    which a feature will be removed, and the performance metric to evaluate.

    Model training and performance calculation are done with cross-validation.

    Parameters
    ----------
    variables : str or list, default=None
        The list of variable(s) to be shuffled from the dataframe.
        If None, the transformer will shuffle all numerical variables in the dataset.

    estimator : object, default = RandomForestClassifier()
        A Scikit-learn estimator for regression or classification.

    scoring : str, default='roc_auc'
        Desired metric to optimise the performance for the estimator. Comes from
        sklearn.metrics. See the model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html

    threshold : float, int, default = None
        The value that defines if a feature will be kept or removed. Note that for
        metrics like roc-auc, r2_score and accuracy, the thresholds will be floats
        between 0 and 1. For metrics like the mean_square_error and the
        root_mean_square_error the threshold will be a big number. The threshold can be
        defined by the user. If None, the selector will select features which
        performance drift is smaller than the mean performance drift across all
        features.

    cv : int, default=3
        Desired number of cross-validation fold to be used to fit the estimator.

    random_state: int, default=None
        Controls the randomness when shuffling features.

    Attributes
    ----------
    initial_model_performance_:
        Performance of the model trained using the original dataset.

    performance_drifts_:
        Dictionary with the performance drift per shuffled feature.

    features_to_drop_:
        List with the features to remove from the dataset.

    Methods
    -------
    fit:
        Find the important features.
    transform:
        Reduce X to the selected features.
    fit_transform:
        Fit to data, then transform it.
    """

    def __init__(
        self,
        estimator=RandomForestClassifier(),
        scoring: str = "roc_auc",
        cv: int = 3,
        threshold: Union[float, int] = None,
        variables: Variables = None,
        random_state: int = None,
    ):

        if not isinstance(cv, int) or cv < 1:
            raise ValueError("cv can only take positive integers bigger than 1")

        if threshold and not isinstance(threshold, (int, float)):
            raise ValueError("threshold can only be integer or float or None")

        self.variables = _check_input_parameter_variables(variables)
        self.estimator = estimator
        self.scoring = scoring
        self.threshold = threshold
        self.cv = cv
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Find the important features.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
           The input dataframe
        y : array-like of shape (n_samples)
           Target variable. Required to train the estimator.

        Returns
        -------
        self
        """

        # check input dataframe
        X = _is_dataframe(X)

        # reset the index
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        # find numerical variables or check variables entered by user
        self.variables = _find_or_check_numerical_variables(X, self.variables)

        # train model with all features and cross-validation
        model = cross_validate(
            self.estimator,
            X[self.variables],
            y,
            cv=self.cv,
            return_estimator=True,
            scoring=self.scoring,
        )

        # store initial model performance
        self.initial_model_performance_ = model["test_score"].mean()

        # get performance metric
        scorer = get_scorer(self.scoring)

        # seed
        random_state = check_random_state(self.random_state)

        # dict to collect features and their performance_drift after shuffling
        self.performance_drifts_ = {}

        # shuffle features and save feature performance drift into a dict
        for feature in self.variables:

            X_shuffled = X[self.variables].copy()

            # shuffle individual feature
            X_shuffled[feature] = (
                X_shuffled[feature]
                .sample(frac=1, random_state=random_state)
                .reset_index(drop=True)
            )

            # determine the performance with the shuffled feature
            performance = np.mean(
                [scorer(m, X_shuffled, y) for m in model["estimator"]]
            )

            # determine drift in performance
            # Note, sklearn negates the log and error scores, so no need to manually
            # do the inversion
            # https://scikit-learn.org/stable/modules/model_evaluation.html
            # (https://scikit-learn.org/stable/modules/model_evaluation.html
            # #the-scoring-parameter-defining-model-evaluation-rules)
            performance_drift = self.initial_model_performance_ - performance

            # Save feature and performance drift
            self.performance_drifts_[feature] = performance_drift

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

        self.input_shape_ = X.shape

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)

        return X

    transform.__doc__ = BaseSelector.transform.__doc__
