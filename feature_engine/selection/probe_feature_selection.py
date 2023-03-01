import numpy as np
import pandas as pd

from feature_engine.variable_handling import find_or_check_numerical_variables
from feature_engine.dataframe_checks import check_X_y
from feature_engine._docstrings.fit_attributes import (
    _feature_importances_docstring,
    _feature_names_in_docstring,
    _n_features_in_docstring,
)

from feature_engine._docstrings.init_parameters.selection import (
    _confirm_variables_docstring,
    _estimator_docstring,
)

from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.substitute import Substitution
from feature_engine.selection._docstring import (
    _cv_docstring,
    _features_to_drop_docstring,
    _fit_docstring,
    _get_support_docstring,
    _scoring_docstring,
    _transform_docstring,
    _variables_numerical_docstring,
)

@Substitution(
    estimator=_estimator_docstring,
    scoring=_scoring_docstring,
    cv=_cv_docstring,
    confirm_variables=_confirm_variables_docstring,
    variables_=_variables_numerical_docstring,
    feature_importances_=_feature_importances_docstring,
    feature_names_in_=_feature_names_in_docstring,
    features_to_drop_=_features_to_drop_docstring,
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

    The transformer ranks the variables based on their feature importances. The
    variables that have a feature importance less than the feature importance or
    average feature importance of the probe feature(s) are dropped from the dataset.

    Parameters
    ----------
    {estimator}

    {scoring}

    n_probes: int, default=1
        Number of probe features to be created. If distribution is 'all',
        n_probes must be a multiple of 3.

    distribution: str, default='normal'
        The distribution used to create the probe features. The options are
        'normal', 'binomial', 'uniform', and 'all'. 'all' creates at least 1 or more
        probe features comprised of each distribution type, i.e., normal, binomial,
        and uniform. The remaining options create `n_probes` features of the selected
        distribution

    {cv}

    Attributes
    ----------
    probe_features_:
        A dataframe comprised of the pseudo-randomly generated features based
        on the selected distribution.

    {feature_importances_}

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
    """

