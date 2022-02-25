# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_input_matches_training_df,
    _is_dataframe,
)
from feature_engine.docstrings import (
    Substitution,
    _drop_original_docstring,
    _feature_names_in_docstring,
    _fit_not_learn_docstring,
    _fit_transform_docstring,
    _missing_values_docstring,
    _n_features_in_docstring,
    _variables_numerical_docstring,
)
from feature_engine.validation import _return_tags
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_numerical_variables,
)


@Substitution(
    variables=_variables_numerical_docstring,
    missing_values=_missing_values_docstring,
    drop_original=_drop_original_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    fit_transform=_fit_transform_docstring,
)
class WindowFeatures(BaseEstimator, TransformerMixin):
    """
        LagFeatures adds lag features to the dataframe. A lag feature is a feature with
        information about a prior time step.

        LagFeatures has the same functionality as pandas `shift()` with the exception that
        only one of `periods` or `freq` can be indicated at a time. LagFeatures builds on
        top of pandas `shift()` in that multiple lags can be created at the same time and
        the features with names will be concatenated to the original dataframe.

        To be compatible with LagFeatures, the dataframe's index must have unique values
        and no NaN.

        LagFeatures works only with numerical variables. You can pass a list of variables
        to lag. Alternatively, LagFeatures will automatically select and lag all numerical
        variables found in the training set.

        More details in the :ref:`User Guide <lag_features>`.

        Parameters
        ----------
        {variables}

        periods: int, list of ints, default=1
            Number of periods to shift. Can be a positive integer or list of positive
            integers. If list, features will be created for each one of the periods in the
            list. If the parameter `freq` is specified, `periods` will be ignored.

        freq: str, list of str, default=None
            Offset to use from the tseries module or time rule. See parameter `freq` in
            pandas `shift()`. It is the same functionality. If freq is a list, lag features
            will be created for each one of the frequency values in the list. If freq is not
            None, then this parameter overrides the parameter `periods`.

        sort_index: bool, default=True
            Whether to order the index of the dataframe before creating the lag features.

        {missing_values}

        {drop_original}

        Attributes
        ----------
        variables_:
            The group of variables that will be lagged.

        {feature_names_in_}

        {n_features_in_}

        Methods
        -------
        {fit}

        transform:
            Add lag features.

        {fit_transform}

        See Also
        --------
        pandas.shift
        """

    def __init__(
            self,
            variables: List[str] = None,
            window: Union[str, int] = None,
            periods: int = 1,
            freq: str = None,
            sort_index: bool = True,
            missing_values: str = "raise",
            drop_original: bool = False,
    ) -> None:

        if not isinstance(window, [str, int]):
            raise ValueError(
                f"window must be a string or integer. Got {window} instead."
            )

        if not isinstance(periods, int):
            raise ValueError(
                f"periods must be an integer. Got {periods} instead."
            )

        if not isinstance(freq, str):
            raise ValueError(
                f"freq must be a string. Got {freq} instead."
            )

        if missing_values not in ["raise", "ignore"]:
            raise ValueError(
                "missing_values takes only values 'raise' or 'ignore'. "
                f"Got {missing_values} instead."
            )

        if not isinstance(drop_original, bool):
            raise ValueError(
                "drop_original takes only boolean values True and False. "
                f"Got {drop_original} instead."
            )

        self.variables = _check_input_parameter_variables(variables)
        self.window = window
        self.periods = periods
        self.freq = freq
        self.sort_index = sort_index
        self.missing_values = missing_values
        self.drop_original = drop_original