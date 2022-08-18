from typing import List, Union

import pandas as pd
from sklearn.model_selection import cross_validate

from feature_engine._variable_handling.init_parameter_checks import (
    _check_init_parameter_variables,
)
from feature_engine._variable_handling.variable_type_selection import (
    _find_or_check_numerical_variables,
)
from feature_engine.dataframe_checks import check_X_y
from feature_engine.selection.base_selector import BaseSelector, get_feature_importances
from feature_engine.tags import _return_tags

Variables = Union[None, int, str, List[Union[str, int]]]


class BaseRecursiveSelector(BaseSelector):
    """
    Shared functionality for recursive selectors.

    Parameters
    ----------
    estimator: object
        A Scikit-learn estimator for regression or classification.
        The estimator must have either a `feature_importances` or `coef_` attribute
        after fitting.

    variables: str or list, default=None
        The list of variable to be evaluated. If None, the transformer will evaluate
        all numerical features in the dataset.

    scoring: str, default='roc_auc'
        Desired metric to optimise the performance of the estimator. Comes from
        sklearn.metrics. See the model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html

    threshold: float, int, default = 0.01
        The value that defines if a feature will be kept or removed. Note that for
        metrics like roc-auc, r2_score and accuracy, the thresholds will be floats
        between 0 and 1. For metrics like the mean_square_error and the
        root_mean_square_error the threshold can be a big number.
        The threshold must be defined by the user. Bigger thresholds will select less
        features.

    cv: int, cross-validation generator or an iterable, default=3
        Determines the cross-validation splitting strategy. Possible inputs for cv are:

            - None, to use cross_validate's default 5-fold cross validation

            - int, to specify the number of folds in a (Stratified)KFold,

            - CV splitter
                - (https://scikit-learn.org/stable/glossary.html#term-CV-splitter)

            - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and y is either binary or
        multiclass, StratifiedKFold is used. In all other cases, KFold is used. These
        splitters are instantiated with `shuffle=False` so the splits will be the same
        across calls. For more details check Scikit-learn's `cross_validate`'s
        documentation.

    confirm_variables: bool, default=False
        If set to True, variables that are not present in the input dataframe will be
        removed from the list of variables. Only used when passing a variable list to
        the parameter `variables`. See parameter variables for more details.

    Attributes
    ----------
    initial_model_performance_:
        Performance of the model trained using the original dataset.

    feature_importances_:
        Pandas Series with the feature importance (comes from step 2)

    performance_drifts_:
        Dictionary with the performance drift per examined feature (comes from step 5).

    features_to_drop_:
        List with the features to remove from the dataset.

    variables_:
        The variables that will be considered for the feature selection.

    feature_names_in_:
        List with the names of features seen during `fit`.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Find the important features.
    """

    _estimator_docstring = """estimator: object
        A Scikit-learn estimator for regression or classification.
        The estimator must have either a `feature_importances` or a `coef_` attribute
        after fitting.
        """.rstrip()

    _feature_importances_docstring = """feature_importances_:
        Pandas Series with the feature importance (comes from step 2)
        """.rstrip()

    _performance_drifts_docstring = """performance_drifts_:
        Dictionary with the performance drift per examined feature (comes from step 5).
        """.rstrip()

    def __init__(
        self,
        estimator,
        scoring: str = "roc_auc",
        cv=3,
        threshold: Union[int, float] = 0.01,
        variables: Variables = None,
        confirm_variables: bool = False,
    ):

        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold can only be integer or float")

        super().__init__(confirm_variables)
        self.variables = _check_init_parameter_variables(variables)
        self.estimator = estimator
        self.scoring = scoring
        self.threshold = threshold
        self.cv = cv

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Find initial model performance. Sort features by importance.

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
        self.variables_ = _find_or_check_numerical_variables(X, self.variables_)

        # check that there are more than 1 variable to select from
        self._check_variable_number()

        # save input features
        self._get_feature_names_in(X)

        # train model with all features and cross-validation
        model = cross_validate(
            self.estimator,
            X[self.variables_],
            y,
            cv=self.cv,
            scoring=self.scoring,
            return_estimator=True,
        )

        # store initial model performance
        self.initial_model_performance_ = model["test_score"].mean()

        # Initialize a dataframe that will contain the list of the feature/coeff
        # importance for each cross validation fold
        feature_importances_cv = pd.DataFrame()

        # Populate the feature_importances_cv dataframe with columns containing
        # the feature importance values for each model returned by the cross
        # validation.
        # There are as many columns as folds.
        for i in range(len(model["estimator"])):
            m = model["estimator"][i]
            feature_importances_cv[i] = get_feature_importances(m)

        # Add the variables as index to feature_importances_cv
        feature_importances_cv.index = self.variables_

        # Aggregate the feature importance returned in each fold
        self.feature_importances_ = feature_importances_cv.mean(axis=1)

        return X, y

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
