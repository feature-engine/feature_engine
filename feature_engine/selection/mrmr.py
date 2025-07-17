import copy

from types import GeneratorType
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.model_selection import GridSearchCV

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
    _scoring_docstring,
    _transform_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import check_X_y
from feature_engine.selection._selection_constants import (
    _CLASSIFICATION_METRICS,
    _REGRESSION_METRICS,
)
from feature_engine.selection.base_selector import BaseSelector
from feature_engine.tags import _return_tags
from feature_engine.variable_handling import (
    check_numerical_variables,
    find_numerical_variables,
    retain_variables_if_in_df,
)

_cv_docstring = _cv_docstring + """ Only used when `method = 'RFCQ'`."""

_scoring_docstring = _scoring_docstring + """. Only used when `method = 'RFCQ'`."""


Variables = Union[None, int, str, List[Union[str, int]]]


@Substitution(
    scoring=_scoring_docstring,
    cv=_cv_docstring,
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
class MRMR(BaseSelector):
    """
    |

    `MRMR()` selects features using the Minimum Redundancy and Maximum Relevance (MRMR)
    framework. With MRMR, we select features that have a strong relationship with the
    target variable (relevance), but weak relationship with other predictor variables
    (redundance).

    Relevance is determined by calculating the mutual information or the F-statistic
    (from ANOVA or correlation) between each predictor and target. The relevance can
    also be determined as the random forest derived feature importance.

    Redundancy is calculated as the mean correlation or mean mutual information of each
    feature to other predictor variables.

    An importance score is then calculated as the difference or the ratio between
    relevance and redundance.

    MRMR is an iterative algorithm. It first determines the relevance of all features
    and selects the one whose value is the highest.

    In the second round, it determines the redundance of all features respect to the
    selected one, calculates the importance score, and selects the one with the highest
    value.

    After that, it repeats the procedure from the second step, this time taking the
    average redundance of the remaining features to those already selected.

    Relevance and Redundance values can be combined as follows:

    |

    .. csv-table::
        :header: Method, Relevance, Redundance, Scheme

        'MID', Mutual information, Mutual information, Difference,
        'MIQ', Mutual information, Mutual information, Ratio,
        'FCD', F-Statistic, Correlation, Difference,
        'FCQ', F-Statistic, Correlation, Ratio,
        'RFCQ', Random Forests, Correlation, Ratio,

    |

    More details in the :ref:`User Guide <mrmr>`.

    Parameters
    ----------
    variables: list, default=None
        The list of variables to evaluate. If `None`, the transformer will evaluate all
        numerical variables in the dataset.

    method: str, default = 'MIQ'
        How to estimate the relevance, redundance and relation between the two. Check
        table above for more details.

    max_features: int, default = None
        The number of features to select. If `None`, it defaults to 20% of the features
        seen during `fit()`.

    discrete_features: bool, str, array, default='auto'
        If bool, then determines whether to consider all features discrete or
        continuous. If array, then it should be a boolean mask with shape
        (n_features,). Ensure that the array matches the discrete features passed in
        `variables` if not None, or in X.columns otherwise. If ‘auto’, it is assigned
        to False for dense X and to True for sparse X. Only used when `method` is
        `'MIQ'` or `'MID'`.

    n_neighbors: int, default=3
        Number of neighbors to use for MI estimation for continuous variables. Higher
        values reduce variance of the estimation, but could introduce a bias. Only used
        when `method` is `'MIQ'` or `'MID'`.

    {scoring}

    {cv}

    param_grid: dictionary, default=None
        The hyperparameters to optimize for the random forest through a grid search.
        `param_grid` can contain any of the permitted hyperparameters for Scikit-learn's
        RandomForestRegressor() or RandomForestClassifier(). If None, then param_grid
        will optimize the 'max_depth' over `[1, 2, 3, 4]`. Only used when `method` is
        `'RFCQ'`.

    regression: boolean, default=True
        Indicates whether the target is one for regression or a classification.

    {confirm_variables}

    random_state: int, default=None
        Seed for reproducibility. Used when `method` is one of `'RFCQ'`, `'MIQ'`, or
        `'MID'` as seed for scikit-learn's `mutual_info_classif`,
        `mutual_info_regression` or random forest model.

    n_jobs: int, default=None
     The number of jobs to use for computing the mutual information. The
     parallelization is done on the columns of X. None means 1 unless in a
     joblib.parallel_backend context. -1 means using all processors. Used when `method`
     is one of `'RFCQ'`, `'MIQ'`, or `'MID'` for scikit-learn's `mutual_info_classif`,
     `mutual_info_regression` or random forest model.

    Attributes
    ----------
    {variables_}

    relevance_:
        Array with the mutual information, f-statistic or random forest derived
        importance for each feature respect to the target.

    {features_to_drop_}

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
    .. [1] Zhao, et al. "Maximum Relevance and Minimum Redundancy Feature Selection
        Methods for a Marketing Machine Learning Platform". 2019
        https://arxiv.org/abs/1908.05376

    Examples
    --------

    >>> from sklearn.datasets import fetch_california_housing
    >>> from feature_engine.selection import MRMR
    >>> X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    >>> X.drop(labels=["Latitude", "Longitude"], axis=1, inplace=True)
    >>> mrmr_sel = MRMR(method="MIQ", regression=True, random_state=3)
    >>> X_t = mrmr_sel.fit_transform(X, y)
    >>> print(X_t.head())
       MedInc  AveOccup
    0  8.3252  2.555556
    1  8.3014  2.109842
    2  7.2574  2.802260
    3  5.6431  2.547945
    4  3.8462  2.181467
    """

    def __init__(
        self,
        variables: Variables = None,
        method: str = "MIQ",
        max_features: Optional[int] = None,
        discrete_features="auto",
        n_neighbors=3,
        scoring: str = "roc_auc",
        cv=3,
        param_grid: Optional[dict] = None,
        regression: bool = False,
        confirm_variables: bool = False,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ):

        if not isinstance(method, str) or method not in [
            "MIQ",
            "MID",
            "FCQ",
            "FCD",
            "RFCQ",
        ]:
            raise ValueError(
                "method must be one of 'MIQ', 'MID', 'FCQ', 'FCD', 'RFCQ'. "
                f"Got {method} instead."
            )

        if max_features is not None and not (
            isinstance(max_features, int) and max_features > 0
        ):
            raise ValueError(
                "max_features must be an integer with the number of features to "
                f"select. Got {max_features} instead."
            )

        if (
            variables is not None
            and max_features is not None
            and not isinstance(variables, int)
        ):
            if max_features >= len(variables):
                raise ValueError(
                    f"The number of variables to examine is {len(variables)}, which is "
                    "less than or equal to the number of features to select indicated "
                    f"in `max_features`, which is {max_features}. Please check the "
                    "values entered in the parameters `variables` and `max_features`."
                )

        if (
            regression is True
            and method == "RFCQ"
            and scoring not in _REGRESSION_METRICS
        ):
            raise ValueError(
                f"The metric {scoring} is not suitable for regression. Set the "
                "parameter regression to False or choose a different performance "
                "metric."
            )

        if (
            regression is False
            and method == "RFCQ"
            and scoring not in _CLASSIFICATION_METRICS
        ):
            raise ValueError(
                f"The metric {scoring} is not suitable for classification. Set the"
                "parameter regression to True or choose a different performance "
                "metric."
            )

        super().__init__(confirm_variables)
        self.variables = _check_variables_input_value(variables)
        self.method = method
        self.max_features = max_features
        self.discrete_features = discrete_features
        self.n_neighbors = n_neighbors
        self.scoring = scoring
        self.cv = cv
        self.param_grid = param_grid
        self.regression = regression
        self.random_state = random_state
        self.n_jobs = n_jobs

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

        if self.variables is None:
            self.variables_ = find_numerical_variables(X)
        else:
            if self.confirm_variables is True:
                variables_ = retain_variables_if_in_df(X, self.variables)
                self.variables_ = check_numerical_variables(X, variables_)
            else:
                self.variables_ = check_numerical_variables(X, self.variables)

        # check that there are more than 1 variable to select from
        self._check_variable_number()

        # save input features
        self._get_feature_names_in(X)

        # start the search
        self.relevance_ = self._calculate_relevance(X[self.variables_], y)

        # select most relevant feature
        relevance = self.relevance_
        n = relevance.argmax()  # most relevant feature
        relevance = np.delete(relevance, n)

        remaining = copy.deepcopy(self.variables_)  # all variables to examine
        feature = remaining[n]  # most important feature so far
        selected = [feature]  # selected feature so far
        remaining.remove(feature)  # remaining features to examine

        # calculate redundance
        redundance = self._calculate_redundance(X[remaining], X[feature])

        # find feature with highest mrmr
        mrmr = self._calculate_mrmr(relevance, redundance)

        if self.max_features is None:
            iter = int(0.2 * len(self.variables_)) - 2
        else:
            iter = self.max_features - 2

        for i in range(iter):
            # find feature with maximum relevance and minimum redundancy
            n = mrmr.argmax()

            # adjust feature lists
            feature = remaining[n]
            selected.append(feature)
            remaining.remove(feature)

            relevance = np.delete(relevance, n)
            if i == 0:
                redundance = np.delete(redundance, n)
            else:
                redundance = np.delete(redundance, n, axis=1)

            new_redundance = self._calculate_redundance(X[remaining], X[feature])
            redundance = np.vstack([redundance, new_redundance])
            mean_redundance = redundance.mean(axis=0)

            mrmr = self._calculate_mrmr(relevance, mean_redundance)

        n = mrmr.argmax()
        selected.append(remaining[n])

        self.features_to_drop_ = [f for f in self.variables_ if f not in selected]

        return self

    def _calculate_relevance(self, X, y):

        if self.method in ["MIQ", "MID"]:
            if self.regression is True:
                relevance = mutual_info_regression(
                    X=X,
                    y=y,
                    discrete_features=self.discrete_features,
                    n_neighbors=self.n_neighbors,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                )
            else:
                relevance = mutual_info_classif(
                    X=X,
                    y=y,
                    discrete_features=self.discrete_features,
                    n_neighbors=self.n_neighbors,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                )

        elif self.method in ["FCQ", "FCD"]:
            if self.regression is True:
                relevance = f_regression(X, y)[0]
            else:
                relevance = f_classif(X, y)[0]

        else:
            if self.regression is True:
                model = RandomForestRegressor(
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                )
            else:
                model = RandomForestClassifier(
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                )

            if self.param_grid:
                param_grid = self.param_grid
            else:
                param_grid = {"max_depth": [1, 2, 3, 4]}

            cv = list(self.cv) if isinstance(self.cv, GeneratorType) else self.cv

            model = GridSearchCV(
                model, cv=cv, scoring=self.scoring, param_grid=param_grid
            )

            model.fit(X, y)

            relevance = model.best_estimator_.feature_importances_

        return relevance

    def _calculate_redundance(self, X, y):

        if self.method in ["FCD", "FCQ", "RFCQ"]:
            redundance = X.corrwith(y).values
            redundance = np.absolute(redundance)

        else:
            redundance = mutual_info_regression(
                X=X,
                y=y,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

        return redundance

    def _calculate_mrmr(self, relevance, redundance):
        if self.method in ["MID", "FCD"]:
            mrmr = relevance - redundance
        else:
            mrmr = relevance / redundance
        return mrmr

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        tags_dict["requires_y"] = True
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"

        msg = "transformers need more than 1 feature to work"
        tags_dict["_xfail_checks"]["check_fit2d_1feature"] = msg
        tags_dict["_xfail_checks"]["check_fit2d_1sample"] = msg

        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
