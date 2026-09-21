from types import GeneratorType
from typing import List, Tuple, Union

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame, IntoSeries
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_validate

from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine.dataframe_checks import check_X_y
from feature_engine.selection.base_selection_functions import (
    _importance_series,
    _select_numerical_variables,
    get_feature_importances,
)
from feature_engine.selection.base_selector import BaseSelector
from feature_engine.tags import _return_tags

Variables = Union[None, int, str, List[Union[str, int]]]


class BaseRecursiveSelector(BaseSelector):
    """
    Shared functionality for recursive selectors.

    Parameters
    ----------
    estimator: object
        A Scikit-learn estimator for regression or classification.

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

    groups: Array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting
        the dataset into train/test set. Only used in conjunction with a
        “Group” cv instance (e.g., GroupKFold).

    confirm_variables: bool, default=False
        If set to True, variables that are not present in the input dataframe will be
        removed from the list of variables. Only used when passing a variable list to
        the parameter `variables`. See parameter variables for more details.

    Attributes
    ----------
    initial_model_performance_:
        Performance of the model trained using the original dataset.

    feature_importances_:
        The feature importance (comes from step 2). A pandas Series with the
        features as index when X is a pandas dataframe, and a dictionary with the
        features as keys otherwise.

    feature_importances_std_:
        The standard deviation of the feature importance, as a pandas Series or a
        dictionary, like `feature_importances_`.

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

    def __init__(
        self,
        estimator,
        scoring: str = "roc_auc",
        cv=3,
        groups=None,
        threshold: Union[int, float] = 0.01,
        variables: Variables = None,
        confirm_variables: bool = False,
    ):

        if not isinstance(threshold, (int, float)):
            raise ValueError(
                f"threshold must be an integer or a float. Got {threshold} instead."
            )

        super().__init__(confirm_variables)
        self.variables = _check_variables_input_value(variables)
        self.estimator = estimator
        self.scoring = scoring
        self.threshold = threshold
        self.cv = cv
        self.groups = groups

    def fit(self, X: IntoDataFrame, y: IntoSeries) -> Tuple[nw.DataFrame, IntoSeries]:
        """
        Find initial model performance. Sort features by importance.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
           The input dataframe

        y: array-like of shape (n_samples)
           Target variable. Required to train the estimator.

        Returns
        -------
        nw_X: narwhals dataframe
            The input dataframe, as a narwhals dataframe.

        y: Series or numpy array
            The target, checked.
        """
        nw_X, y = check_X_y(X, y)

        self.variables_ = _select_numerical_variables(
            X, self.variables, self.confirm_variables
        )

        self._cv = list(self.cv) if isinstance(self.cv, GeneratorType) else self.cv

        # check that there are more than 1 variable to select from
        self._check_variable_number()

        self._get_feature_names_in(X)

        X_model = nw_X.select(nw.col(*self.variables_)).to_native()

        # train model with all features and cross-validation
        model = cross_validate(
            estimator=self.estimator,
            X=X_model,
            y=y,
            cv=self._cv,
            groups=self.groups,
            scoring=self.scoring,
            return_estimator=True,
        )

        self.initial_model_performance_ = model["test_score"].mean()

        # one row of feature importance per cross-validation fold
        importances = []
        for m in model["estimator"]:
            if hasattr(m, "feature_importances_") or hasattr(m, "coef_"):
                importances.append(get_feature_importances(m))
            else:
                r = permutation_importance(
                    m,
                    X_model,
                    y,
                    n_repeats=1,
                    random_state=10,
                )
                importances.append(r.importances_mean)
        importances_arr = np.array(importances)

        self.feature_importances_ = _importance_series(
            X, self.variables_, importances_arr.mean(axis=0)
        )
        self.feature_importances_std_ = _importance_series(
            X, self.variables_, importances_arr.std(axis=0, ddof=1)
        )

        return nw_X, y

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
