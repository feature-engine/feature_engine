from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.init_parameters.selection import (
    _confirm_variables_docstring,
    _estimator_docstring,
)
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import check_X_y
from feature_engine._docstrings.selection._docstring import (
    _cv_docstring,
    _features_to_drop_docstring,
    _fit_docstring,
    _get_support_docstring,
    _scoring_docstring,
    _transform_docstring,
    _variables_attribute_docstring,
    _variables_numerical_docstring,
)
from feature_engine.selection.base_selector import BaseSelector, get_feature_importances
from feature_engine.tags import _return_tags
from feature_engine.variable_handling import find_or_check_numerical_variables

Variables = Union[None, int, str, List[Union[str, int]]]


@Substitution(
    estimator=_estimator_docstring,
    scoring=_scoring_docstring,
    cv=_cv_docstring,
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
    ProbeFeatureSelection() generates one or more probe features based on the
    user-selected distribution. The distribution options are 'normal', 'binomial',
    'uniform', or 'all'. 'all' creates at least one distribution for each of the
    three aforementioned distributions.

    Using cross validation, the class fits a Scikit-learn estimator to the
    provided dataset's variables and the probe features.

    The class derives the feature importance for each variable and probe feature.
    In the case of there being more than one probe feature, ProbeFeatureSelection()
    calculates the average feature importance of all the probe features.

    The variables that have a feature importance less than the feature importance or
    average feature importance of the probe feature(s) are dropped from the dataset.

    More details in the :ref:`User Guide <probe_features>`.

    Parameters
    ----------
    {estimator}

    {variables}

    {scoring}

    n_probes: int, default=1
        Number of probe features to be created. If distribution is 'all',
        n_probes must be a multiple of 3.

    distribution: str, default='normal'
        The distribution used to create the probe features. The options are
        'normal', 'binomial', 'uniform', and 'all'. 'all' creates at least 1 or more
        probe features comprised of each distribution type, i.e., normal, binomial,
        and uniform. The remaining options create `n_probes` features of the selected
        distribution.

    {cv}

    Attributes
    ----------
    probe_features_:
        A dataframe comprised of the pseudo-randomly generated features based
        on the selected distribution.

    feature_importances_:
        Pandas Series with the feature importance.

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
    print(X.shape, X_tr.shape)
    """

    def __init__(
        self,
        estimator,
        variables: Variables = None,
        scoring: str = "roc_auc",
        n_probes: int = 1,
        distribution: str = "normal",
        cv: int = 5,
        random_state: int = 0,
        confirm_variables: bool = False,
    ):
        if distribution not in ["normal", "binary", "uniform", "all"]:
            raise ValueError(
                "distribution takes on 'normal', 'binary', 'uniform', or 'all' as "
                f"values. Got {distribution} instead."
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
        self.scoring = scoring
        self.distribution = distribution
        self.cv = cv
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

        # if required exclude variables that are not in the input dataframe
        # instantiates the 'variables_' attribute
        self._confirm_variables(X)

        # find numerical variables
        self.variables_ = find_or_check_numerical_variables(X, self.variables_)

        # save input features
        self._get_feature_names_in(X)

        # create probe feature distributions
        self.probe_features_ = self._generate_probe_features(X.shape[0])

        # required for a (train/test) split dataset
        X.reset_index(drop=True, inplace=True)

        X_new = pd.concat([X[self.variables_], self.probe_features_], axis=1)

        # train model with all variables including the probe features
        model = cross_validate(
            self.estimator,
            X_new,
            y,
            cv=self.cv,
            scoring=self.scoring,
            return_estimator=True,
        )

        # Initialize a dataframe that will contain the list of the feature/coeff
        # importance for each cross-validation fold
        feature_importances_cv = pd.DataFrame()

        # Populate the feature_importances_cv dataframe with columns containing
        # the feature importance values for each model returned by the cross
        # validation.
        # There are as many columns as folds.
        for i in range(len(model["estimator"])):
            m = model["estimator"][i]
            feature_importances_cv[i] = get_feature_importances(m)

        # add the variables as the index to feature_importances_cv
        feature_importances_cv.index = X_new.columns

        # aggregate the feature importance returned in each fold
        self.feature_importances_ = feature_importances_cv.mean(axis=1)

        # get features that have an importance less than the probe features'
        # avg importance attribute used in transform() which is inherited
        # from parent class
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

                elif self.distribution == "uniform":
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
