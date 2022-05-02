import warnings
from typing import List, Optional, Union, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_contains_na,
    _check_X_matches_training_df,
    check_X,
    check_X_y,
)
from feature_engine.get_feature_names_out import _get_feature_names_out
from feature_engine._docstrings.substitute import Substitution
from feature_engine._docstrings.methods import _get_feature_names_out_docstring
from feature_engine.encoding._docstrings import (
    _errors_docstring,
    _ignore_format_docstring,
    _variables_docstring,
)
from feature_engine.tags import _return_tags
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_all_variables,
    _find_or_check_categorical_variables,
)


@Substitution(
    ignore_format=_ignore_format_docstring,
    variables=_variables_docstring,
)
class BaseCategoricalTransformer(BaseEstimator, TransformerMixin):
    """shared set-up checks and methods across categorical transformers

    Parameters
    ----------
    {variables}.

    {ignore_format}
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        ignore_format: bool = False,
    ) -> None:

        if not isinstance(ignore_format, bool):
            raise ValueError(
                "ignore_format takes only booleans True and False. "
                f"Got {ignore_format} instead."
            )

        self.variables = _check_input_parameter_variables(variables)
        self.ignore_format = ignore_format

    def _check_X(self, X: pd.DataFrame) -> pd.DataFrame:
        return check_X(X)

    def _check_X_y(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        return check_X_y(X, y)

    def _check_or_select_variables(self, X: pd.DataFrame):
        """
        Finds categorical variables, or alternatively checks that the variables
        entered by the user are of type object (categorical).
        Checks absence of NA.

        Parameters
        ----------
        X: Pandas DataFrame

        Raises
        ------
        TypeError
            If any user provided variable is not categorical
        ValueError
            If there are no categorical variables in the df or the df is empty
            If the variable(s) contain null values
        """
        if not self.ignore_format:
            # find categorical variables or check variables entered by user are object
            self.variables_: List[
                Union[str, int]
            ] = _find_or_check_categorical_variables(X, self.variables)
        else:
            # select all variables or check variables entered by the user
            self.variables_ = _find_all_variables(X, self.variables)

        # check if dataset contains na
        _check_contains_na(X, self.variables_)

    def _get_feature_names_in(self, X: pd.DataFrame):

        # save input features
        self.feature_names_in_ = X.columns.tolist()

        # save train set shape
        self.n_features_in_ = X.shape[1]

    def _check_transform_input_and_state(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Checks that the input is a dataframe and of the same size than the one used
        in the fit method. Checks absence of NA.

        Parameters
        ----------
        X: Pandas DataFrame

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            - If the variable(s) contain null values.
            - If the df has different number of features than the df used in fit()

        Returns
        -------
        X: Pandas DataFrame
            The same dataframe entered by the user.
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        # Check input data contains same number of columns as df used to fit
        _check_X_matches_training_df(X, self.n_features_in_)

        # check if dataset contains na
        _check_contains_na(X, self.variables_)

        # reorder df to match train set
        X = X[self.feature_names_in_]

        return X

    def _check_encoding_dictionary(self):
        """After fit(), the encoders should return a dictionary with the original values
        to numerical mappings as key, values. This function checks that the dictionary
        was created and is not empty.
        """

        # check that dictionary is not empty
        if len(self.encoder_dict_) == 0:
            raise ValueError(
                "Encoder could not be fitted. Check the parameters and the variables "
                "in your dataframe."
            )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace categories with the learned parameters.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features].
            The dataset to transform.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features].
            The dataframe containing the categories replaced by numbers.
        """

        X = self._check_transform_input_and_state(X)

        # replace categories by the learned parameters
        for feature in self.encoder_dict_.keys():
            X[feature] = X[feature].map(self.encoder_dict_[feature])

            # if original variables are cast as categorical, they will remain
            # categorical after the encoding, and this is probably not desired
            if pd.api.types.is_categorical_dtype(X[feature]):
                if all(isinstance(x, int) for x in X[feature]):
                    X[feature] = X[feature].astype("int")
                else:
                    X[feature] = X[feature].astype("float")

        # check if NaN values were introduced by the encoding
        if X[self.encoder_dict_.keys()].isnull().sum().sum() > 0:

            # obtain the name(s) of the columns have null values
            nan_columns = (
                X[self.encoder_dict_.keys()]
                .columns[X[self.encoder_dict_.keys()].isnull().any()]
                .tolist()
            )

            if len(nan_columns) > 1:
                nan_columns_str = ", ".join(nan_columns)
            else:
                nan_columns_str = nan_columns[0]

            if self.errors == "ignore":
                warnings.warn(
                    "During the encoding, NaN values were introduced in the feature(s) "
                    f"{nan_columns_str}."
                )
            elif self.errors == "raise":
                raise ValueError(
                    "During the encoding, NaN values were introduced in the feature(s) "
                    f"{nan_columns_str}."
                )

        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert the encoded variable back to the original values.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features].
            The transformed dataframe.

        Returns
        -------
        X_tr: pandas dataframe of shape = [n_samples, n_features].
            The un-transformed dataframe, with the categorical variables containing the
            original values.
        """

        X = self._check_transform_input_and_state(X)

        # replace encoded categories by the original values
        for feature in self.encoder_dict_.keys():
            inv_map = {v: k for k, v in self.encoder_dict_[feature].items()}
            X[feature] = X[feature].map(inv_map)

        return X

    @Substitution(get_feature_names_out=_get_feature_names_out_docstring)
    def get_feature_names_out(
        self, input_features: Optional[List[Union[str, int]]] = None
    ) -> List[Union[str, int]]:
        """{get_feature_names_out}"""

        check_is_fitted(self)

        feature_names = _get_feature_names_out(
            features_in=self.feature_names_in_,
            transformed_features=self.variables_,
            input_features=input_features,
        )

        return feature_names

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "categorical"
        # the below test will fail because sklearn requires to check for inf, but
        # you can't check inf of categorical data, numpy returns and error.
        # so we need to leave without this test
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"
        return tags_dict


@Substitution(
    ignore_format=_ignore_format_docstring,
    variables=_variables_docstring,
    errors=_errors_docstring,
)
class BaseCategorical(BaseCategoricalTransformer):
    """
    BaseCategorical() is the parent class to some of the encoders.
    It shares set-up checks of init parameters.

    Parameters
    ----------
    {variables}

    {ignore_format}

    {errors}
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        ignore_format: bool = False,
        errors: str = "ignore",
    ) -> None:
        if errors not in ["raise", "ignore"]:
            raise ValueError(
                "errors takes only values 'raise' and 'ignore ."
                f"Got {errors} instead."
            )

        super().__init__(variables, ignore_format)
        self.errors = errors
