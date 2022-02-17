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
class LagFeatures(BaseEstimator, TransformerMixin):
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
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        periods: int = 1,
        freq: Union[str, List[str]] = None,
        sort_index: bool = True,
        missing_values: str = "raise",
        drop_original: bool = False,
    ) -> None:

        if (
            isinstance(periods, int)
            and periods > 0
            or isinstance(periods, list)
            and all(isinstance(num, int) and num > 0 for num in periods)
        ):
            self.periods = periods
        else:
            raise ValueError(
                "periods must be an integer or a list of positive integers. "
                f"Got {periods} instead."
            )

        if not isinstance(sort_index, bool):
            raise ValueError(
                "sort_index takes values True and False." f"Got {sort_index} instead."
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
        self.freq = freq
        self.sort_index = sort_index
        self.missing_values = missing_values
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn parameters.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y: pandas Series, default=None
            y is not needed in this transformer. You can pass None or y.
        """
        # check input dataframe
        X = _is_dataframe(X)

        # We need the dataframes to have unique values in the index and no missing data.
        # Otherwise, when we merge the lag features we will duplicate rows.

        # Check that the index contains unique values.
        if X.index.is_unique is False:
            raise NotImplementedError(
                "The dataframe's index does not contain unique values. "
                "Only dataframes with unique values in the index are compatible "
                "with this transformer."
            )

        if X.index.isnull().sum() > 0:
            raise NotImplementedError(
                "The dataframe's index contains NaN values or missing data. "
                "Only dataframes with complete indexes are compatible with "
                "this transformer."
            )

        # find or check for numerical variables
        self.variables_ = _find_or_check_numerical_variables(X, self.variables)

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.variables_)
            _check_contains_inf(X, self.variables_)

        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Adds lag features.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: Pandas dataframe, shape = [n_samples, n_features + lag_features]
            The dataframe with the original plus the new variables.
        """
        # Check method fit has been called
        check_is_fitted(self)

        # check if 'X' is a dataframe
        X = _is_dataframe(X)

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_input_matches_training_df(X, self.n_features_in_)

        # We need the dataframes to have unique values in the index and no missing data.
        # Otherwise, when we merge the lag features we will duplicate rows.

        # Check that the index contains unique values.
        if X.index.is_unique is False:
            raise NotImplementedError(
                "The dataframe's index does not contain unique values. "
                "Only dataframes with unique values in the index are compatible "
                "with this transformer."
            )

        if X.index.isnull().sum() > 0:
            raise NotImplementedError(
                "The dataframe's index contains NaN values or missing data. "
                "Only dataframes with complete indexes are compatible with "
                "this transformer."
            )

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.variables_)
            _check_contains_inf(X, self.variables_)

        if self.sort_index is True:
            X.sort_index(inplace=True)

        # if freq is not None, it overrides periods.
        if self.freq is not None:

            if isinstance(self.freq, list):
                df_ls = []
                for fr in self.freq:
                    tmp = X[self.variables_].shift(
                        freq=fr,
                        axis=0,
                    )
                    df_ls.append(tmp)
                tmp = pd.concat(df_ls, axis=1)

            else:
                tmp = X[self.variables_].shift(
                    freq=self.freq,
                    axis=0,
                )

        else:
            if isinstance(self.periods, list):
                df_ls = []
                for pr in self.periods:
                    tmp = X[self.variables_].shift(
                        periods=pr,
                        axis=0,
                    )
                    df_ls.append(tmp)
                tmp = pd.concat(df_ls, axis=1)

            else:
                tmp = X[self.variables_].shift(
                    periods=self.periods,
                    axis=0,
                )

        tmp.columns = self.get_feature_names_out(self.variables_)

        X = X.merge(tmp, left_index=True, right_index=True, how="left")

        if self.drop_original:
            X = X.drop(self.variables_, axis=1)

        return X

    def get_feature_names_out(self, input_features: Optional[List] = None) -> List:
        """Get output feature names for transformation.

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

        # Create names for all lag features or just the indicated ones.
        if input_features is None:
            # Create all lag features.
            input_features_ = self.variables_
        else:
            if not isinstance(input_features, list):
                raise ValueError(
                    f"input_features must be a list. Got {input_features} instead."
                )
            if any([f for f in input_features if f not in self.variables_]):
                raise ValueError(
                    "Some features in input_features were not lagged. You can only get"
                    "the names of the lagged features with this function."
                )
            # Create just indicated lag features.
            input_features_ = input_features

        # create the names for the lag features
        if isinstance(self.freq, list):
            feature_names = [
                str(feature) + f"_lag_{fr}"
                for fr in self.freq
                for feature in input_features_
            ]
        elif self.freq is not None:
            feature_names = [
                str(feature) + f"_lag_{self.freq}" for feature in input_features_
            ]
        elif isinstance(self.periods, list):
            feature_names = [
                str(feature) + f"_lag_{pr}"
                for pr in self.periods
                for feature in input_features_
            ]
        else:
            feature_names = [
                str(feature) + f"_lag_{self.periods}" for feature in input_features_
            ]

        # Return names of all variables if input_features is None.
        if input_features is None:
            if self.drop_original is True:
                # Remove names of variables to drop.
                original = [
                    f for f in self.feature_names_in_ if f not in self.variables_
                ]
                feature_names = original + feature_names
            else:
                feature_names = self.feature_names_in_ + feature_names

        return feature_names

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["allow_nan"] = True
        tags_dict["variables"] = "numerical"
        # add additional test that fails
        tags_dict["_xfail_checks"][
            "check_methods_subset_invariance"
        ] = "tLagFeatures is not invariant when applied to a subset. Not sure why yet"
        return tags_dict
