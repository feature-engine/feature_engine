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
    'uniform', or 'all'. 'all' creates at least one feature for each of the
    three aforementioned distributions.

    Using cross validation, ProbeFeatureSelection() fits a Scikit-learn estimator
    to the provided variables plus the probe features. Next, it derives the
    feature importance for each variable and probe feature from the fitted model.

    Alternatively, ProbeFeatureSelection() fits a Scikit-learn estimator per feature
    and probe feature (single feature models), and then determines the performance
    returned by that model,, using a metric of choice.

    Finally, ProbeFeatureSelection() selects the features whose importance is greater
    than those of the probes. In the case of there being more than one probe feature,
    ProbeFeatureSelection() takes the average feature importance of all the probe
    features.

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
        Number of probe features to be created. If distribution is 'all', n_probes must
        be a multiple of 3.

    distribution: str, default='normal'
        The distribution used to create the probe features. The options are
        'normal', 'binomial', 'uniform', and 'all'. 'all' creates at least 1 or more
        probe features comprised of each distribution type, i.e., normal, binomial,
        and uniform. The remaining options create `n_probes` features of the selected
        distribution.

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
        https://jmlr.org/papers/volume3/stoppiglia03a/stoppiglia03a.pdf

    Examples
    --------

    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.linear_model import LogisticRegression
    >>> from feature_engine.selection import ProbeFeatureSelection
    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    >>> sel = ProbeFeatureSelection(
    >>>     estimator=LogisticRegression(),
    >>>     scoring="roc_auc",
    >>>     n_probes=3,
    >>>     distribution="normal",
    >>>     cv=3,
    >>>     random_state=150,
    >>> )
    >>> X_tr = sel.fit_transform(X, y)
    >>> print(X.shape, X_tr.shape)
    (569, 30) (569, 9)
    """

    def __init__(
        self,
        estimator,
        variables: Variables = None,
        collective: bool = True,
        scoring: str = "roc_auc",
        n_probes: int = 1,
        distribution: str = "normal",
        cv=5,
        groups=None,
        random_state: int = 0,
        confirm_variables: bool = False,
    ):
        if not isinstance(collective, bool):
            raise ValueError(
                f"collective takes values True or False. Got {collective} instead."
            )

        if distribution not in ["normal", "binary", "uniform", "all"]:
            raise ValueError(
                "distribution takes values 'normal', 'binary', 'uniform', or 'all'. "
                f"Got {distribution} instead."
            )

        if distribution == "all" and n_probes % 3 != 0:
            raise ValueError(
                "If distribution is 'all' the n_probes must be a multiple of 3. "
                f"Got {n_probes} instead."
            )

        if not isinstance(n_probes, int):
            raise ValueError(f"n_probes must be an integer. Got {n_probes} instead.")

        super().__init__(confirm_variables)
        self.estimator = estimator
        self.variables = variables
        self.collective = collective
        self.scoring = scoring
        self.distribution = distribution
        self.cv = cv
        self.groups = groups
        self.n_probes = n_probes
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
        Returns a dataframe comprised of the probe feature using the
        selected distribution.
        """
        # create dataframe
        df = pd.DataFrame()

        # set random state
        np.random.seed(self.random_state)
        if self.distribution == "all":
            generation_cnt = self.n_probes // 3
            for i in range(generation_cnt):
                df[f"gaussian_probe_{i}"] = np.random.normal(0, 3, n_obs)
                df[f"binary_probe_{i}"] = np.random.randint(0, 2, n_obs)
                df[f"uniform_probe_{i}"] = np.random.uniform(0, 1, n_obs)

        # when distribution is normal, binary, or uniform
        else:
            for i in range(self.n_probes):
                if self.distribution == "normal":
                    df[f"gaussian_probe_{i}"] = np.random.normal(0, 3, n_obs)

                elif self.distribution == "binary":
                    df[f"binary_probe_{i}"] = np.random.randint(0, 2, n_obs)

                else:
                    df[f"uniform_probe_{i}"] = np.random.uniform(0, 1, n_obs)

        return df

    def _get_features_to_drop(self):
        """
        Identify the variables that have a lower feature importance than the average
        feature importance of all the probe features.
        """

        # if more than 1 probe feature, calculate average feature importance
        if self.n_probes > 1:
            probe_features_avg_importance = self.feature_importances_[
                self.probe_features_.columns
            ].values.mean()

        else:
            probe_features_avg_importance = self.feature_importances_[
                self.probe_features_.columns
            ].values

        features_to_drop = []

        for var in self.variables_:
            if self.feature_importances_[var] < probe_features_avg_importance:
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
