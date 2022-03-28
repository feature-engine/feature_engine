# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union, Callable

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

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
from feature_engine.timeseries.forecasting import BaseForecast
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
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
class WindowFeatures(BaseForecast):
    """
    WindowFeatures performs windowing operations on the specified variables.
    A windowing operation perform an aggregation - e.g. mean or sum - over
    a sliding partition of values.

    WindowFeatures combines the function of pandas 'rolling()' and 'shift()'.

    To be compatible with WindowFeatures, the dataframe's index must have unique values
    and no NaN.

    WindowFeatures works only with numerical variables. You can pass a list of variables
    to lag. Alternatively, WindowFeatures will automatically select and lag all numerical
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
        pandas `shift()`. If freq is not None, then this parameter overrides the
        parameter `periods`.

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
            window: Union[str, int] = 1,
            function: Callable = np.mean,
            periods: int = 1,
            freq: str = None,
            sort_index: bool = True,
            missing_values: str = "raise",
            drop_original: bool = False,
    ) -> None:

        if isinstance(periods, int) and periods > 0:
            self.periods = periods
        else:
            raise ValueError(
                f"periods must be a positive integer. Got {periods} instead."
            )

        if isinstance(window, int) and window > 0:
            self.window = window
        elif isinstance(window, str):
            self.window = window
        else:
            raise ValueError(
                f"window must be a positive integer or string. Got {window} instead."
            )

        if function not in (np.mean, np.std, np.median, np.min, np.max, np.sum):
            raise ValueError(
                f"function must be np.mean, np.std, np.median, np.min, np.max, "
                f"or np.sum. Got {function} instead."
            )
        if periods is None and not isinstance(freq, str):
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
        self.function = function
        self.freq = freq
        self.sort_index = sort_index
        self.missing_values = missing_values
        self.drop_original = drop_original

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Adds window features.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: Pandas dataframe, shape = [n_samples, n_features + lag_features]
            The dataframe with the original plus the new variables.
        """
        # Performs various checks
        X = super().transform(X)

        # if freq is not None, it overrides periods.
        if self.freq is not None:
            tmp = (X[self.variables_]
                   .rolling(window=self.window).apply(self.function)
                   .shift(freq=self.freq)
                   )

        else:
            tmp = (X[self.variables_]
                   .rolling(window=self.window).apply(self.function)
                   .shift(periods=self.periods)
                   )

        tmp.columns = self.get_feature_names_out(self.variables_)

        X = X.merge(tmp, left_index=True, right_index=True, how="left")

        if self.drop_original:
            X = X.drop(self.variables_, axis=1)

        return X

    def get_feature_names_out(self, input_features: Optional[List] = None) -> List:
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features: list, default=None
            Input features. If `input_features` is `None`, then the names of all the
            variables in the transformed dataset (original + new variables) is returned.
            Alternatively, only the names for the lag features derived from
            input_features will be returned.

        Returns
        -------
        feature_names_out: list
            The feature names.
        """
        check_is_fitted(self)

        # create names for all window features or just the indicated ones.
        if input_features is None:
            input_features_ = self.variables_
        else:
            if not isinstance(input_features, list):
                raise ValueError(
                    f"input_features must be a list. Got {input_features} instead."
                )
            if any([f for f in input_features if f not in self.variables_]):
                raise ValueError(
                    "Some features in input_features were not transformed. This method only "
                    "provides the names of the transform features with this method."
                )
            # create just indicated window features
            input_features_ = input_features

        if self.freq is not None:
            feature_names = [
                str(feature) + f"_window_{self.window}_freq_{self.freq}"
                for feature in input_features_
            ]
        else:
            feature_names = [
                str(feature) + f"_window_{self.window}_periods_{self.periods}"
                for feature in input_features_
            ]

        # return names of all variables if input_features is None
        if input_features is None:
            if self.drop_original is True:
                # removes names of variables to drop
                feature_names = [
                    f for f in self.feature_names_in_ if f not in self.variables_
                ]
            else:
                feature_names = self.feature_names_in_ + feature_names

        return feature_names
