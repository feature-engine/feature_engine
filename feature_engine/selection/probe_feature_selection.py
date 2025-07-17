from typing import List, Union

import numpy as np
import pandas as pd

from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.init_parameters.selection import (
    _confirm_variables_docstring,
    _estimator_docstring,
)
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.selection._docstring import (
    _cv_docstring,
    _groups_docstring,
    _features_to_drop_docstring,
    _fit_docstring,
    _get_support_docstring,
    _scoring_docstring,
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
    find_feature_importance,
    single_feature_performance,
)

Variables = Union[None, int, str, List[Union[str, int]]]


@Substitution(
    estimator=_estimator_docstring,
    scoring=_scoring_docstring,
    cv=_cv_docstring,
    groups=_groups_docstring,
    confirm_variables=_confirm_variables_docstring,
    variables=_variables_numerical_docstring,
    feature_names_in_=_feature_names_in_docstring,
    features_to_drop_=_features_to_drop_docstring,
    variables_=_variables_attribute_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_docstring,
    transform=_transform_docstring,
    fit_transform=_fit_transform_docstring,
    get_support=_get_support_docstring,
)
class ProbeFeatureSelection(BaseSelector):
    """
    ProbeFeatureSelection() generates one or more probe (i.e., random) features based
    on a user-selected distribution. The distribution options are 'normal', 'binomial',
    'uniform', 'discrete_uniform', 'poisson', or 'all'. 'all' creates `n_probes` of
    each of the five aforementioned distributions.

    Using cross validation, ProbeFeatureSelection() fits a Scikit-learn estimator
    to the provided variables plus the probe features. Next, it derives the
    feature importance for each variable and probe feature from the fitted model.

    Alternatively, ProbeFeatureSelection() fits a Scikit-learn estimator per feature
    and probe feature (single feature models), and then determines the performance
    returned by that model using a metric of choice.

    Finally, ProbeFeatureSelection() selects the features whose importance is greater
    than those of the probes. In the case of there being more than one probe feature,
    ProbeFeatureSelection() can take the average, maximum, or mean plus 3 std feature
    importance of all the probe features as threshold for the feature selection.

    The variables whose feature importance is smaller than the feature importance of
    the probe feature(s) are dropped from the dataset.

    More details in the :ref:`User Guide <probe_features>`.

    Parameters
    ----------
    estimator: object
        A Scikit-learn estimator for regression or classification. If `collective=True`,
        the estimator must have either a `feature_importances_` or a `coef_` attribute
        after fitting.

    {variables}

    collective: bool, default=True
         Whether the feature importance should be derived from an estimator trained on
         the entire dataset (True), or trained using individual features (False).

    {scoring}

    n_probes: int, default=1
        Number of probe features to create per distribution.

    distribution: str, list, default='normal'
        The distribution used to create the probe features. The options are 'normal',
        'binomial', 'uniform', 'discrete_uniform', 'poisson' and 'all'. 'all' creates
        `n_probes` features per distribution type, i.e., normal, binomial,
        uniform, discrete_uniform and poisson. The remaining options create
        `n_probes` features per selected distributions.

    n_categories: int, default=10
        If `distribution` is 'discrete_uniform' then integers are sampled from 0
        to `n_categories`. If `distribution` is 'poisson', then samples are taken from
        `np.random.poisson(n_categories, n_obs)`.

    threshold: str, default='mean'
        Indicates how to combine the importance of the probe features as threshold for
        the feature selection. If 'mean', then features are selected if their
        importance is greater than the mean of the probes. If 'max', then features are
        selected if their importance is greater than the maximun importance of all
        probes. If 'mean_plus_std', then features are selected if their importance is
        greater than the mean plus three times the standard deviation of the probes.

    {cv}

    {groups}

    Attributes
    ----------
    probe_features_:
        A dataframe comprised of the pseudo-randomly generated features based
        on the selected distribution.

    feature_importances_:
        Pandas Series with the feature importance. If `collective=True`, the feature
        importance is given by the coefficients of linear models or the importance
        derived from tree-based models. If `collective=False`, the feature importance
        is given by a performance metric returned by a model trained using that
        individual feature.

    feature_importances_std_:
        Pandas Series with the standard deviation of the feature importance.

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

    References
    ----------
    .. [1] Stoppiglia, et al. "Ranking a Random Feature for Variable and Feature
        Selection". JMLR: 1399-1414, 2003
        https://dl.acm.org/doi/pdf/10.5555/944919.944980

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.linear_model import LogisticRegression
    >>> from feature_engine.selection import ProbeFeatureSelection
    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    >>> sel = ProbeFeatureSelection(
    >>>     estimator=LogisticRegression(max_iter=1000000),
    >>>     scoring="roc_auc",
    >>>     n_probes=3,
    >>>     distribution="normal",
    >>>     cv=3,
    >>>     random_state=150,
    >>> )
    >>> X_tr = sel.fit_transform(X, y)
    >>> print(X.shape, X_tr.shape)
    (569, 30) (569, 19)
    """

    def __init__(
        self,
        estimator,
        variables: Variables = None,
        collective: bool = True,
        scoring: str = "roc_auc",
        n_probes: int = 1,
        distribution: Union[str, list] = "normal",
        n_categories: int = 10,
        threshold: str = "mean",
        cv=5,
        groups=None,
        random_state: int = 0,
        confirm_variables: bool = False,
    ):
        if not isinstance(collective, bool):
            raise ValueError(
                f"collective takes values True or False. Got {collective} instead."
            )

        error_msg = (
            "distribution takes values 'normal', 'binary', 'uniform', "
            "'discrete_uniform', 'poisson', or 'all'. "
            f"Got {distribution} instead."
        )

        allowed_distributions = [
            "normal",
            "binary",
            "uniform",
            "discrete_uniform",
            "poisson",
            "all",
        ]

        if not isinstance(distribution, (str, list)):
            raise ValueError(error_msg)
        if isinstance(distribution, str) and distribution not in allowed_distributions:
            raise ValueError(error_msg)
        if isinstance(distribution, list) and not all(
            dist in allowed_distributions for dist in distribution
        ):
            raise ValueError(error_msg)

        if not isinstance(n_probes, int):
            raise ValueError(f"n_probes must be an integer. Got {n_probes} instead.")

        if not isinstance(n_categories, int) or n_categories < 1:
            raise ValueError(
                f"n_categories must be a positive integer. Got {n_categories} instead."
            )

        if not isinstance(threshold, str) or threshold not in [
            "mean",
            "max",
            "mean_plus_std",
        ]:
            raise ValueError(
                "threshold takes values 'mean', 'max' or 'mean_plus_std'. "
                f"Got {threshold} instead."
            )

        super().__init__(confirm_variables)
        self.estimator = estimator
        self.variables = variables
        self.collective = collective
        self.scoring = scoring
        self.distribution = distribution
        self.n_categories = n_categories
        self.cv = cv
        self.groups = groups
        self.n_probes = n_probes
        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Find the important features.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]

        y: array-like of shape (n_samples)
            Target variable. Required to train the estimator.
        """
        # check input dataframe
        X, y = check_X_y(X, y)

        self.variables_ = _select_numerical_variables(
            X, self.variables, self.confirm_variables
        )

        # save input features
        self._get_feature_names_in(X)

        # create probe feature distributions
        self.probe_features_ = self._generate_probe_features(X.shape[0])

        # required for a (train/test) split dataset
        X.reset_index(drop=True, inplace=True)

        X_new = pd.concat([X[self.variables_], self.probe_features_], axis=1)

        if self.collective is True:
            # train model using entire dataset and derive feature importance
            f_importance_mean, f_importance_std = find_feature_importance(
                X=X_new,
                y=y,
                estimator=self.estimator,
                cv=self.cv,
                groups=self.groups,
                scoring=self.scoring,
            )
            self.feature_importances_ = f_importance_mean
            self.feature_importances_std_ = f_importance_std

        else:
            # trains a model per feature (single feature models)
            f_importance_mean, f_importance_std = single_feature_performance(
                X=X_new,
                y=y,
                variables=X_new.columns,
                estimator=self.estimator,
                cv=self.cv,
                groups=self.groups,
                scoring=self.scoring,
            )
            self.feature_importances_ = pd.Series(f_importance_mean)
            self.feature_importances_std_ = pd.Series(f_importance_std)

        # get features with lower importance than the probe features
        self.features_to_drop_ = self._get_features_to_drop()

        return self

    def _generate_probe_features(self, n_obs: int) -> pd.DataFrame:
        """
        Returns a dataframe comprised of the probe features using the
        selected distribution.
        """
        # create dataframe
        df = pd.DataFrame()

        # set random state
        np.random.seed(self.random_state)

        if isinstance(self.distribution, str):
            distribution = set([self.distribution])
        else:
            distribution = set(self.distribution)

        if {"normal", "all"} & distribution:
            for i in range(self.n_probes):
                df[f"gaussian_probe_{i}"] = np.random.normal(0, 3, n_obs)

        if {"binary", "all"} & distribution:
            for i in range(self.n_probes):
                df[f"binary_probe_{i}"] = np.random.randint(0, 2, n_obs)

        if {"uniform", "all"} & distribution:
            for i in range(self.n_probes):
                df[f"uniform_probe_{i}"] = np.random.uniform(0, 1, n_obs)

        if {"discrete_uniform", "all"} & distribution:
            for i in range(self.n_probes):
                df[f"discrete_uniform_probe_{i}"] = np.random.randint(
                    0, self.n_categories, n_obs
                )

        if {"poisson", "all"} & distribution:
            for i in range(self.n_probes):
                df[f"poisson_probe_{i}"] = np.random.poisson(self.n_categories, n_obs)

        return df

    def _get_features_to_drop(self):
        """
        Identify the variables that have a lower feature importance than the average
        feature importance of all the probe features.
        """

        # if more than 1 probe feature, calculate threshold based on
        # probe feature importance.
        if self.probe_features_.shape[1] > 1:
            if self.threshold == "mean":
                threshold = self.feature_importances_[
                    self.probe_features_.columns
                ].values.mean()
            elif self.threshold == "max":
                threshold = self.feature_importances_[
                    self.probe_features_.columns
                ].values.max()
            else:
                threshold = (
                    self.feature_importances_[
                        self.probe_features_.columns
                    ].values.mean()
                    + 3
                    * self.feature_importances_[
                        self.probe_features_.columns
                    ].values.std()
                )

        else:
            threshold = self.feature_importances_[self.probe_features_.columns].values

        features_to_drop = []

        for var in self.variables_:
            if self.feature_importances_[var] < threshold:
                features_to_drop.append(var)

        return features_to_drop

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        tags_dict["requires_y"] = True
        # add additional test that fails
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"
        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has 1 mandatory parameter"

        # msg = "transformers need more than 1 feature to work"
        # tags_dict["_xfail_checks"]["check_fit2d_1feature"] = msg

        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
