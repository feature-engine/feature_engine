import warnings
from typing import List, Union

import narwhals as nw
import narwhals.dependencies as nwd
from narwhals.typing import IntoDataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
from feature_engine._check_init_parameters.check_init_input_params import (
    _check_return_empty_is_bool,
)
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.init_parameters.all_transformers import (
    _missing_values_docstring,
    _return_empty_docstring,
    _variables_categorical_docstring,
)
from feature_engine._docstrings.init_parameters.encoders import _ignore_format_docstring
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import (
    _check_contains_na,
    _check_X_matches_training_df,
    check_X,
)
from feature_engine.tags import _return_tags
from feature_engine.variable_handling import (
    check_all_variables,
    check_categorical_variables,
    find_all_variables,
    find_categorical_variables,
)


@Substitution(
    ignore_format=_ignore_format_docstring,
    variables=_variables_categorical_docstring,
    return_empty=_return_empty_docstring,
)
class CategoricalInitMixin:
    """Shared initialization parameters across transformers. Sets and checks init
    parameters.

    Parameters
    ----------
    {variables}

    {return_empty}

    {ignore_format}
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
        ignore_format: bool = False,
    ) -> None:

        if not isinstance(ignore_format, bool):
            raise ValueError(
                "ignore_format takes only booleans True and False. "
                f"Got {ignore_format} instead."
            )

        _check_return_empty_is_bool(return_empty)

        self.variables = _check_variables_input_value(variables)
        self.return_empty = return_empty
        self.ignore_format = ignore_format


@Substitution(
    missing_values=_missing_values_docstring,
    ignore_format=_ignore_format_docstring,
    variables=_variables_categorical_docstring,
)
class CategoricalInitMixinNA:
    """Shared initialization parameters across transformers. Sets and checks init
    parameters.

    Parameters
    ----------
    {variables}.

    {missing_values}

    {ignore_format}
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        missing_values: str = "raise",
        ignore_format: bool = False,
    ) -> None:

        if missing_values not in ["raise", "ignore"]:
            raise ValueError(
                "missing_values takes only values 'raise' or 'ignore'. "
                f"Got {missing_values} instead."
            )

        if not isinstance(ignore_format, bool):
            raise ValueError(
                "ignore_format takes only booleans True and False. "
                f"Got {ignore_format} instead."
            )

        self.variables = _check_variables_input_value(variables)
        self.ignore_format = ignore_format
        self.missing_values = missing_values


class CategoricalMethodsMixin(TransformerMixin, BaseEstimator, GetFeatureNamesOutMixin):
    """Shared methods across categorical transformers.

    - BaseEstimator brings methods get_params() and set_params().
    - TransformerMixin brings method fit_transform()
    - GetFeatureNamesOutMixin brings method get_feature_names_out().
    """

    def _check_na(self, X: IntoDataFrame, variables):
        if self.missing_values == "raise":
            _check_contains_na(X, variables, error_msg="optional")

    def _check_or_select_variables(self, X: IntoDataFrame):
        """
        Finds categorical variables, or alternatively checks that the variables
        entered by the user are of type object (categorical).
        Checks absence of NA.

        Parameters
        ----------
        X: dataframe

        Raises
        ------
        TypeError
            If any user provided variable is not categorical
        ValueError
            If there are no categorical variables in the df or the df is empty
            If the variable(s) contain null values
        """
        # select variables to encode
        if self.ignore_format is True:
            if self.variables is None:
                variables_ = find_all_variables(X)
            else:
                variables_ = check_all_variables(X, self.variables)
        else:
            if self.variables is None:
                variables_ = find_categorical_variables(
                    X, return_empty=self.return_empty
                )
            else:
                variables_ = check_categorical_variables(X, self.variables)

        return variables_

    def _get_feature_names_in(self, X: IntoDataFrame):
        """
        Returns attributes `featrure_names_in_` and `n_feature_names_in_`, which are
        standard for all transformers in the library.
        """
        # save input features
        is_pandas = nwd.is_pandas_dataframe(X)
        if is_pandas is True:
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = nw.from_native(X, eager_only=True).columns

        # save train set shape
        self.n_features_in_ = X.shape[1]

    def _check_transform_input_and_state(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Checks that the input is a dataframe and of the same size than the one used
        in the fit method. Checks absence of NA.

        Parameters
        ----------
        X: dataframe

        Raises
        ------
        TypeError
            If the input is not a dataframe
        ValueError
            - If the variable(s) contain null values.
            - If the df has different number of features than the df used in fit()

        Returns
        -------
        X: dataframe
            The same dataframe entered by the user.
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        # Check input data contains same number of columns as df used to fit
        _check_X_matches_training_df(X, self.n_features_in_)

        # reorder df to match train set
        is_pandas = nwd.is_pandas_dataframe(X)
        if is_pandas is True:
            X = X[self.feature_names_in_]
        else:
            X = (
                nw.from_native(X, eager_only=True)
                .select(self.feature_names_in_)
                .to_native()
            )

        return X

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """Replace categories with the learned parameters.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features].
            The dataset to transform.

        Returns
        -------
        X_new: dataframe of shape = [n_samples, n_features].
            The dataframe containing the categories replaced by numbers.
        """

        X = self._check_transform_input_and_state(X)

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.variables_, error_msg="optional")

        X = self._encode(X)

        return X

    def _encode(self, X: IntoDataFrame) -> IntoDataFrame:
        # replace categories by the learned parameters.
        # narwhals' replace_strict() lets one expression both map known
        # categories and fill unseen/missing ones via `default`, so the
        # pandas-only category-dtype fixup this used to need (map() leaves
        # category dtype behind) is no longer necessary: replace_strict
        # already resolves to a plain numeric dtype on both backends.
        # get_column()/Series.replace_strict() (rather than nw.col(), which
        # only accepts string names) is what lets this handle pandas
        # integer column names too, same as DecisionTreeFeatures.
        default = self._unseen if self.unseen == "encode" else None
        nw_X = nw.from_native(X, eager_only=True)
        new_series = [
            nw_X.get_column(feature).replace_strict(mapping, default=default)
            for feature, mapping in self.encoder_dict_.items()
        ]
        X = nw_X.with_columns(*new_series).to_native()

        if self.unseen != "encode":
            # check if nan values were introduced by the transformation
            self._check_nan_values_after_transformation(X)

        return X

    def _check_nan_values_after_transformation(self, X):

        # check if NaN values were introduced by the encoding
        nw_X = nw.from_native(X, eager_only=True)
        nan_columns = [
            feature
            for feature in self.encoder_dict_.keys()
            if nw_X.get_column(feature).null_count() > 0
        ]

        if len(nan_columns) > 0:

            if len(nan_columns) > 1:
                nan_columns_str = ", ".join(str(col) for col in nan_columns)
            else:
                nan_columns_str = str(nan_columns[0])

            if self.unseen == "ignore":
                warnings.warn(
                    "During the encoding, NaN values were introduced in the feature(s) "
                    f"{nan_columns_str}."
                )
            elif self.unseen == "raise":
                raise ValueError(
                    "During the encoding, NaN values were introduced in the feature(s) "
                    f"{nan_columns_str}."
                )

    def inverse_transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """Convert the encoded variable back to the original values.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features].
            The transformed dataframe.

        Returns
        -------
        X_tr: dataframe of shape = [n_samples, n_features].
            The un-transformed dataframe, with the categorical variables containing the
            original values.
        """

        X = self._check_transform_input_and_state(X)

        # replace encoded categories by the original values. get_column()
        # rather than nw.col() again, to support pandas integer column names.
        nw_X = nw.from_native(X, eager_only=True)
        new_series = [
            nw_X.get_column(feature).replace_strict(
                {v: k for k, v in mapping.items()}, default=None
            )
            for feature, mapping in self.encoder_dict_.items()
        ]
        X = nw_X.with_columns(*new_series).to_native()

        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "categorical"
        # the below test will fail because sklearn requires to check for inf, but
        # you can't check inf of categorical data, numpy returns and error.
        # so we need to leave without this test
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
