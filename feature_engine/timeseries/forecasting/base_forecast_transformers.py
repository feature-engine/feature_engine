from typing import List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine._base_transformers.mixins import (
    GetFeatureNamesOutMixin,
    TransformXyMixin,
)
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.init_parameters.all_trasnformers import (
    _drop_original_docstring,
    _missing_values_docstring,
)
from feature_engine._docstrings.methods import _fit_not_learn_docstring
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_X_matches_training_df,
    check_X,
)
from feature_engine.tags import _return_tags
from feature_engine.variable_handling import (
    check_numerical_variables,
    find_numerical_variables,
)


@Substitution(
    missing_values=_missing_values_docstring,
    drop_original=_drop_original_docstring,
    feature_names_in_=_feature_names_in_docstring,
    fit=_fit_not_learn_docstring,
    n_features_in_=_n_features_in_docstring,
)
class BaseForecastTransformer(
    TransformerMixin, BaseEstimator, GetFeatureNamesOutMixin, TransformXyMixin
):
    """
    Shared methods across time-series forecasting transformers.

    Parameters
    ----------
    variables: str, int, or list of strings or integers, default=None.
        The variables to use to create the new features.

    {missing_values}

    {drop_original}

    drop_na: bool, default=False.
        Whether the NAN introduced in the created features should be removed.

    Attributes
    ----------
    {feature_names_in_}

    {n_features_in_}

    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        missing_values: str = "raise",
        drop_original: bool = False,
        drop_na: bool = False,
    ) -> None:

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

        if not isinstance(drop_na, bool):
            raise ValueError(
                "drop_na takes only boolean values True and False. "
                f"Got {drop_na} instead."
            )

        self.variables = _check_variables_input_value(variables)
        self.missing_values = missing_values
        self.drop_original = drop_original
        self.drop_na = drop_na

    def _check_index(self, X: pd.DataFrame):
        """
        Check that the index does not have missing data and its values are unique.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The dataset.
        """
        if X.index.isnull().any():
            raise NotImplementedError(
                "The dataframe's index contains NaN values or missing data. "
                "Only dataframes with complete indexes are compatible with "
                "this transformer."
            )

        if X.index.is_unique is False:
            raise NotImplementedError(
                "The dataframe's index does not contain unique values. "
                "Only dataframes with unique values in the index are "
                "compatible with this transformer."
            )

        return self

    def _check_na_and_inf(self, X: pd.DataFrame):
        """
        Checks that the dataframe does not contain NaN or Infinite values.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The dataset for training or transformation.
        """
        _check_contains_na(X, self.variables_)
        _check_contains_inf(X, self.variables_)

        return self

    def _get_feature_names_in(self, X: pd.DataFrame):
        """
        Finds the number and name of the features in the training set.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The dataset for training or transformation.
        """

        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]

        return self

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
        X = check_X(X)

        # We need the dataframes to have unique values in the index and no missing data.
        # Otherwise, when we merge the new features we will duplicate rows.
        self._check_index(X)

        # find or check for numerical variables
        if self.variables is None:
            self.variables_ = find_numerical_variables(X)
        else:
            self.variables_ = check_numerical_variables(X, self.variables)

        # check if dataset contains na
        if self.missing_values == "raise":
            self._check_na_and_inf(X)

        self._get_feature_names_in(X)

        return self

    def _check_transform_input_and_state(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Common checks performed before the feature transformation.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.
        """
        # check method fit has been called
        check_is_fitted(self)

        # check if 'X' is a dataframe
        X = check_X(X)

        # check if input data contains the same number of columns as the fitted
        # dataframe.
        _check_X_matches_training_df(X, self.n_features_in_)

        # Dataframes must have unique values in the index and no missing data.
        # Otherwise, when we merge the created features we will duplicate rows.
        self._check_index(X)

        # check if dataset contains na
        if self.missing_values == "raise":
            self._check_na_and_inf(X)

        # reorder variables to match train set
        X = X[self.feature_names_in_]

        if self.sort_index is True:
            X.sort_index(inplace=True)

        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["allow_nan"] = True
        tags_dict["variables"] = "numerical"
        # add additional test that fails
        tags_dict["_xfail_checks"][
            "check_methods_subset_invariance"
        ] = "LagFeatures is not invariant when applied to a subset. Not sure why yet"
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
