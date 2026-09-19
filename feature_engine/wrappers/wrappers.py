import warnings
from typing import List, Optional, Union

import narwhals as nw
import narwhals.dependencies as nwd
import numpy as np
from narwhals.typing import IntoDataFrame, IntoSeries
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from feature_engine._check_init_parameters.check_init_input_params import (
    _check_return_empty_is_bool,
)
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine.dataframe_checks import _check_X_matches_training_df, check_X
from feature_engine.tags import _return_tags
from feature_engine.variable_handling import (
    check_numerical_variables,
    find_numerical_variables,
)
from feature_engine.variable_handling.check_variables import check_all_variables
from feature_engine.variable_handling.find_variables import find_all_variables

_SELECTORS = [
    "GenericUnivariateSelect",
    "RFE",
    "RFECV",
    "SelectFdr",
    "SelectFpr",
    "SelectFromModel",
    "SelectFwe",
    "SelectKBest",
    "SelectPercentile",
    "SequentialFeatureSelector",
    "VarianceThreshold",
]

_CREATORS = [
    # 'FeatureHasher',
    "OneHotEncoder",
    "PolynomialFeatures",
]

_TRANSFORMERS = [
    # transformers
    "Binarizer",
    "FunctionTransformer",
    "KBinsDiscretizer",
    "PowerTransformer",
    "QuantileTransformer",
    # imputers
    "SimpleImputer",
    "IterativeImputer",
    "KNNImputer",
    # encoders
    "OrdinalEncoder",
    # scalers
    "MaxAbsScaler",
    "MinMaxScaler",
    "StandardScaler",
    "RobustScaler",
    "Normalizer",
]

_ALL_TRANSFORMERS = _SELECTORS + _CREATORS + _TRANSFORMERS

_INVERSE_TRANSFORM = [
    "PowerTransformer",
    "QuantileTransformer",
    "OrdinalEncoder",
    "MaxAbsScaler",
    "MinMaxScaler",
    "StandardScaler",
    "RobustScaler",
]


class SklearnWrapper(TransformerMixin, BaseEstimator):
    """
    Wrapper to apply scikit-learn transformers to a selected group of variables. It
    supports the following transformers:

    - Binarizer and KBinsDiscretizer (only when encoding=Ordinal)
    - FunctionTransformer, PowerTransformer and QuantileTransformer
    - SimpleImputer, IterativeImputer and KNNImputer (only when add_indicators=False)
    - OrdinalEncoder and OneHotEncoder (only when sparse is False)
    - MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler, Normalizer
    - All selection transformers including VarianceThreshold
    - PolynomialFeatures

    More details in the :ref:`User Guide <sklearn_wrapper>`.

    Parameters
    ----------
    transformer: sklearn transformer
        The desired scikit-learn transformer.

    variables: list, default=None
        The list of variables to be transformed. If None, the wrapper will select all
        variables of type numeric for all transformers, except the SimpleImputer,
        OrdinalEncoder and OneHotEncoder, in which case, it will select all variables
        in the dataset.

    return_empty: bool, default=False
        Whether to return an empty list when no variables of the required type are
        found. If False, the transformer raises an error. This parameter is only
        used when `variables` is `None`.

    Attributes
    ----------
    transformer_:
        The fitted Scikit-learn transformer.

    variables_:
        The group of variables that will be transformed.

    features_to_drop_:
        The variables that will be dropped. Only present when using selection
        transformers

    feature_names_in_:
        List with the names of features seen during `fit`.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Fit scikit-learn transformer.

    fit_transform:
        Fit to data, then transform it.

    get_feature_names_out:
        Get output feature names for transformation.

    get_params:
        Get parameters for this estimator.

    set_params:
        Set the parameters of this estimator.

    inverse_transform:
        Convert the data back to the original representation.

    transform:
        Transform data with the scikit-learn transformer.

    Notes
    -----
    This transformer offers similar functionality to the ColumnTransformer from
    scikit-learn, but it allows entering the transformations directly into a
    Pipeline and returns a dataframe of the same library as the input, for
    example, pandas or polars.

    See Also
    --------
    sklearn.compose.ColumnTransformer

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.wrappers import SklearnWrapper
    >>> from sklearn.preprocessing import StandardScaler
    >>> X = pd.DataFrame(dict(x1 = ["a","b","c"], x2 = [1,2,3], x3 = [4,5,6]))
    >>> skw = SklearnWrapper(StandardScaler())
    >>> skw.fit(X)
    >>> skw.transform(X)
      x1        x2        x3
    0  a -1.224745 -1.224745
    1  b  0.000000  0.000000
    2  c  1.224745  1.224745

    >>> import pandas as pd
    >>> from feature_engine.wrappers import SklearnWrapper
    >>> from sklearn.preprocessing import OneHotEncoder
    >>> X = pd.DataFrame(dict(x1 = ["a","b","c"], x2 = [1,2,3], x3 = [4,5,6]))
    >>> skw = SklearnWrapper(
    >>>     OneHotEncoder(sparse_output = False), variables = "x1")
    >>> skw.fit(X)
    >>> skw.transform(X)
       x2  x3  x1_a  x1_b  x1_c
    0   1   4   1.0   0.0   0.0
    1   2   5   0.0   1.0   0.0
    2   3   6   0.0   0.0   1.0

    >>> import pandas as pd
    >>> from feature_engine.wrappers import SklearnWrapper
    >>> from sklearn.preprocessing import PolynomialFeatures
    >>> X = pd.DataFrame(dict(x1 = ["a","b","c"], x2 = [1,2,3], x3 = [4,5,6]))
    >>> skw = SklearnWrapper(PolynomialFeatures(include_bias = False))
    >>> skw.fit(X)
    >>> skw.transform(X)
      x1   x2   x3  x2^2  x2 x3  x3^2
    0  a  1.0  4.0   1.0    4.0  16.0
    1  b  2.0  5.0   4.0   10.0  25.0
    2  c  3.0  6.0   9.0   18.0  36.0

    With polars:

    >>> import polars as pl
    >>> from feature_engine.wrappers import SklearnWrapper
    >>> from sklearn.preprocessing import OneHotEncoder
    >>> X = pl.DataFrame(dict(x1 = ["a","b","c"], x2 = [1,2,3], x3 = [4,5,6]))
    >>> skw = SklearnWrapper(
    >>>     OneHotEncoder(sparse_output = False), variables = "x1")
    >>> skw.fit(X)
    >>> skw.transform(X)
    shape: (3, 5)
    ┌─────┬─────┬──────┬──────┬──────┐
    │ x2  ┆ x3  ┆ x1_a ┆ x1_b ┆ x1_c │
    │ --- ┆ --- ┆ ---  ┆ ---  ┆ ---  │
    │ i64 ┆ i64 ┆ f64  ┆ f64  ┆ f64  │
    ╞═════╪═════╪══════╪══════╪══════╡
    │ 1   ┆ 4   ┆ 1.0  ┆ 0.0  ┆ 0.0  │
    │ 2   ┆ 5   ┆ 0.0  ┆ 1.0  ┆ 0.0  │
    │ 3   ┆ 6   ┆ 0.0  ┆ 0.0  ┆ 1.0  │
    └─────┴─────┴──────┴──────┴──────┘
    """

    def __init__(
        self,
        transformer,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
    ) -> None:

        if not isinstance(transformer, TransformerMixin):
            raise TypeError(
                "transformer expected a Scikit-learn transformer. "
                f"Got {transformer} instead."
            )

        if transformer.__class__.__name__ not in _ALL_TRANSFORMERS:
            raise NotImplementedError(
                "This transformer is not compatible with the wrapper. "
                "Supported transformers are {}.".format(", ".join(_ALL_TRANSFORMERS))
            )

        if (
            transformer.__class__.__name__
            in ["SimpleImputer", "KNNImputer", "IterativeImputer"]
            and transformer.add_indicator is True
        ):
            raise NotImplementedError(
                "The imputer is only compatible with the wrapper when the "
                "parameter `add_indicator` is False. "
            )

        if (
            transformer.__class__.__name__ == "KBinsDiscretizer"
            and transformer.encode != "ordinal"
        ):
            raise NotImplementedError(
                "The KBinsDiscretizer is only compatible with the wrapper when the "
                "parameter `encode` is `ordinal`. "
            )

        if transformer.__class__.__name__ == "OneHotEncoder":
            msg = (
                "SklearnWrapper can only wrap OneHotEncoder if the "
                "sparse is set to False."
            )
            if getattr(transformer, "sparse", False) or getattr(
                transformer, "sparse_output", False
            ):
                raise NotImplementedError(msg)

        _check_return_empty_is_bool(return_empty)

        self.transformer = transformer
        self.variables = _check_variables_input_value(variables)
        self.return_empty = return_empty

    def fit(self, X: IntoDataFrame, y: Optional[IntoSeries] = None):
        """
        Fits the scikit-learn transformer to the selected variables.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The dataset to fit the transformer. Can be a pandas, polars, or any other
            dataframe supported by narwhals.

        y: Series, default=None
            The target variable. Only needed by the transformers that use it, like
            the feature selectors.
        """

        nw_X = check_X(X)

        self.transformer_ = clone(self.transformer)

        if self.transformer_.__class__.__name__ in [
            "OneHotEncoder",
            "OrdinalEncoder",
            "SimpleImputer",
            "FunctionTransformer",
        ]:
            if self.variables is None:
                self.variables_ = find_all_variables(
                    X, return_empty=self.return_empty
                )
            else:
                self.variables_ = check_all_variables(X, self.variables)

        else:
            if self.variables is None:
                self.variables_ = find_numerical_variables(
                    X, return_empty=self.return_empty
                )
            else:
                self.variables_ = check_numerical_variables(X, self.variables)

        if nwd.is_pandas_dataframe(X) is True:
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = nw_X.columns
        self.n_features_in_ = nw_X.shape[1]

        if len(self.variables_) == 0:
            return self

        # set explicitly, so a global sklearn output config can't change the container
        # transform() expects. FunctionTransformer warns if its function returns arrays.
        if (
            nwd.is_pandas_dataframe(X) is True
            and self.transformer_.__class__.__name__ != "FunctionTransformer"
        ):
            self.transformer_.set_output(transform="pandas")
        else:
            self.transformer_.set_output(transform="default")

        self.transformer_.fit(self._to_sklearn_input(X, nw_X), y)

        if self.transformer_.__class__.__name__ in _SELECTORS:
            selected = [
                self.variables_[i] for i in self.transformer_.get_support(indices=True)
            ]
            self.features_to_drop_ = [f for f in self.variables_ if f not in selected]

        return self

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Apply the transformation to the dataframe. Only the selected variables will be
        modified.

        If the scikit-learn transformer is the OneHotEncoder or the PolynomialFeatures,
        the new features will be concatenated to the input dataset.

        If the scikit-learn transformer is for feature selection, the non-selected
        features will be dropped from the dataframe.

        For all other transformers, the original variables will be replaced by the
        transformed ones.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: dataframe
            The transformed dataset.
        """
        check_is_fitted(self)
        nw_X = check_X(X)
        _check_X_matches_training_df(X, self.n_features_in_)

        if len(self.variables_) == 0:
            return self._select_columns(X, nw_X, self.feature_names_in_)

        # Feature selection: transformers that remove features
        if self.transformer_.__class__.__name__ in _SELECTORS:
            return self._select_columns(
                X,
                nw_X,
                [f for f in self.feature_names_in_ if f not in self.features_to_drop_],
            )

        # Transformers that add features: creators
        if self.transformer_.__class__.__name__ in _CREATORS:
            X_remaining = self._select_columns(
                X,
                nw_X,
                [f for f in self.feature_names_in_ if f not in self.variables_],
            )
            X_new = self.transformer_.transform(self._to_sklearn_input(X, nw_X))
            # pandas input: set_output already returned a dataframe with X's index.
            if nwd.is_pandas_dataframe(X) is True:
                nw_new = nw.from_native(X_new, eager_only=True)
            else:
                nw_new = self._to_frame(
                    X_new,
                    list(self.transformer_.get_feature_names_out(self.variables_)),
                    nw_X,
                )
            return nw.concat(
                [nw.from_native(X_remaining, eager_only=True), nw_new],
                how="horizontal",
            ).to_native()

        # Transformers that modify existing features
        # pandas is faster than narwhals.
        if nwd.is_pandas_dataframe(X) is True:
            X = X[self.feature_names_in_]
            X[self.variables_] = self.transformer_.transform(X[self.variables_])
            return X
        else:
            X_new = self.transformer_.transform(self._to_sklearn_input(X, nw_X))
            nw_new = self._to_frame(X_new, self.variables_, nw_X)
            return (
                nw_X.select(self.feature_names_in_)
                .with_columns(*nw_new.iter_columns())
                .to_native()
            )

    def inverse_transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """Convert the transformed variables back to the original values. Only
        implemented for the following scikit-learn transformers:

        PowerTransformer, QuantileTransformer, OrdinalEncoder,
        MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler.

        If you would like this method implemented for additional transformers,
        please check if they have the inverse_transform method in scikit-learn and then
        raise an issue in our repo.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features].
            The transformed dataframe.

        Returns
        -------
        X_tr: dataframe of shape = [n_samples, n_features].
            The dataframe with the original values.
        """
        check_is_fitted(self)
        nw_X = check_X(X)

        if self.transformer_.__class__.__name__ not in _INVERSE_TRANSFORM:
            raise NotImplementedError(
                "The method `inverse_transform` is not implemented for this "
                "transformer. Supported transformers are {}.".format(
                    ", ".join(_INVERSE_TRANSFORM)
                )
            )

        X_inv = self.transformer_.inverse_transform(self._to_sklearn_input(X, nw_X))

        # pandas is faster than narwhals.
        if nwd.is_pandas_dataframe(X) is True:
            # replacing whole columns leaves the user's dataframe untouched, so a
            # shallow copy is enough.
            X = X.copy(deep=False)
            X[self.variables_] = X_inv
            return X
        else:
            nw_inv = self._to_frame(X_inv, self.variables_, nw_X)
            return nw_X.with_columns(*nw_inv.iter_columns()).to_native()

    def _to_sklearn_input(self, X: IntoDataFrame, nw_X: nw.DataFrame):
        """Return the variables to transform in the format passed to the
        scikit-learn transformer."""
        if nwd.is_pandas_dataframe(X) is True:
            return X[self.variables_]

        # the function in a FunctionTransformer expects the user's dataframe.
        if self.transformer_.__class__.__name__ == "FunctionTransformer":
            return nw_X.select(self.variables_).to_native()

        nw_vars = nw_X.select(self.variables_)
        X_np = nw_vars.to_numpy()
        # scikit-learn treats NaN, not None, as missing in text columns, which is
        # what it gets from pandas.
        if X_np.dtype == object and nw_vars.null_count().to_numpy().sum() > 0:
            X_np[np.equal(X_np, None)] = np.nan
        return X_np

    def _to_frame(
        self, X_new, columns: List[Union[str, int]], nw_X: nw.DataFrame
    ) -> nw.DataFrame:
        """Return the output of the scikit-learn transformer as a narwhals
        dataframe with the given column names, in the backend of nw_X."""
        # the function in a FunctionTransformer may return a dataframe.
        if nwd.is_into_dataframe(X_new) is True:
            nw_new = nw.from_native(X_new, eager_only=True)
            return nw_new.rename(dict(zip(nw_new.columns, columns)))

        # scikit-learn marks missing values with NaN, polars and others with null.
        if X_new.dtype == object:
            X_new[np.not_equal(X_new, X_new)] = None
        elif X_new.dtype.kind == "f" and bool(np.isnan(X_new).any()) is True:
            return nw.from_numpy(
                X_new, schema=columns, backend=nw_X.implementation
            ).with_columns(nw.all().fill_nan(None))

        return nw.from_numpy(X_new, schema=columns, backend=nw_X.implementation)

    def _select_columns(
        self, X: IntoDataFrame, nw_X: nw.DataFrame, columns: List[Union[str, int]]
    ) -> IntoDataFrame:
        # pandas is faster than narwhals.
        if nwd.is_pandas_dataframe(X) is True:
            return X[columns]
        else:
            return nw_X.select(columns).to_native()

    def get_feature_names_out(
        self, input_features: Optional[List[Union[str, int]]] = None
    ) -> List:
        """Get output feature names for transformation.

        input_features: list, default=None
            If `None`, then the names of all the variables in the transformed dataset
            is returned. For those transformers that create and add new features to the
            dataset, like the OneHotEncoder or the PolynomialFeatures, you have the
            option to pass a list with the input features to obtain the newly created
            variables. For all other transformers, this parameter will be ignored.

        Returns
        -------
        feature_names_out: list
            The feature names.
        """
        # Check method fit has been called
        check_is_fitted(self)

        if self.transformer_.__class__.__name__ in _TRANSFORMERS:
            feature_names = self.feature_names_in_

        if self.transformer_.__class__.__name__ in _CREATORS:
            if input_features is None:
                added_features = self.transformer_.get_feature_names_out(
                    self.variables_
                )
                original_features = [
                    feature
                    for feature in self.feature_names_in_
                    if feature not in self.variables_
                ]
                feature_names = original_features + list(added_features)
            else:
                feature_names = list(
                    self.transformer_.get_feature_names_out(input_features)
                )

        if self.transformer_.__class__.__name__ in _SELECTORS:
            feature_names = [
                f for f in self.feature_names_in_ if f not in self.features_to_drop_
            ]

        return feature_names

    def _more_tags(self):
        tags_dict = _return_tags()
        # add additional test that fails
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"
        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has 1 mandatory parameter"
        return tags_dict

    def __sklearn_tags__(self):
        return super().__sklearn_tags__()


# TODO: remove in version 2.1.0
class SklearnTransformerWrapper(SklearnWrapper):
    def __init__(
        self,
        transformer,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
    ) -> None:
        warnings.warn(
            "SklearnTransformerWrapper was deprecated in favour of SklearnWrapper in "
            "version 2.0.0 and will be removed in version 2.1.0. To silence this "
            "warning, use SklearnWrapper instead.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(
            transformer=transformer,
            variables=variables,
            return_empty=return_empty,
        )
