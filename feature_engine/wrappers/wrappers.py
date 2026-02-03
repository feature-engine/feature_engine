from typing import List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

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


class SklearnTransformerWrapper(TransformerMixin, BaseEstimator):
    """
    Wrapper to apply Scikit-learn transformers to a selected group of variables. It
    supports the following transformers:

    - Binarizer and KBinsDiscretizer (only when encoding=Ordinal)
    - FunctionTransformer, PowerTransformer and QuantileTransformer
    - SimpleImputer, IterativeImputer and KNNImputer (only when add_indicators=False)
    - OrdinalEncoder and OneHotEncoder (only when sparse is False)
    - MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler, Normalizer
    - All selection transformers including VarianceThreshold
    - PolynomialFeautures

    More details in the :ref:`User Guide <sklearn_wrapper>`.

    Parameters
    ----------
    transformer: sklearn transformer
        The desired Scikit-learn transformer.

    variables: list, default=None
        The list of variables to be transformed. If None, the wrapper will select all
        variables of type numeric for all transformers, except the SimpleImputer,
        OrdinalEncoder and OneHotEncoder, in which case, it will select all variables
        in the dataset.

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
        Fit Scikit-learn transformer.

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
        Transform data with the Scikit-learn transformer.

    Notes
    -----
    This transformer offers similar functionality to the ColumnTransformer from
    Scikit-learn, but it allows entering the transformations directly into a
    Pipeline and returns pandas dataframes.

    See Also
    --------
    sklearn.compose.ColumnTransformer

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.wrappers import SklearnTransformerWrapper
    >>> from sklearn.preprocessing import StandardScaler
    >>> X = pd.DataFrame(dict(x1 = ["a","b","c"], x2 = [1,2,3], x3 = [4,5,6]))
    >>> skw = SklearnTransformerWrapper(StandardScaler())
    >>> skw.fit(X)
    >>> skw.transform(X)
      x1        x2        x3
    0  a -1.224745 -1.224745
    1  b  0.000000  0.000000
    2  c  1.224745  1.224745

    >>> import pandas as pd
    >>> from feature_engine.wrappers import SklearnTransformerWrapper
    >>> from sklearn.preprocessing import OneHotEncoder
    >>> X = pd.DataFrame(dict(x1 = ["a","b","c"], x2 = [1,2,3], x3 = [4,5,6]))
    >>> skw = SklearnTransformerWrapper(
    >>>     OneHotEncoder(sparse_output = False), variables = "x1")
    >>> skw.fit(X)
    >>> skw.transform(X)
       x2  x3  x1_a  x1_b  x1_c
    0   1   4   1.0   0.0   0.0
    1   2   5   0.0   1.0   0.0
    2   3   6   0.0   0.0   1.0

    >>> import pandas as pd
    >>> from feature_engine.wrappers import SklearnTransformerWrapper
    >>> from sklearn.preprocessing import PolynomialFeatures
    >>> X = pd.DataFrame(dict(x1 = ["a","b","c"], x2 = [1,2,3], x3 = [4,5,6]))
    >>> skw = SklearnTransformerWrapper(PolynomialFeatures(include_bias = False))
    >>> skw.fit(X)
    >>> skw.transform(X)
      x1   x2   x3  x2^2  x2 x3  x3^2
    0  a  1.0  4.0   1.0    4.0  16.0
    1  b  2.0  5.0   4.0   10.0  25.0
    2  c  3.0  6.0   9.0   18.0  36.0
    """

    def __init__(
        self,
        transformer,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
    ) -> None:

        if not issubclass(transformer.__class__, TransformerMixin):
            raise TypeError(
                "transformer expected a Scikit-learn transformer. "
                f"got {transformer} instead. "
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
                "The SklearnTransformerWrapper can only wrap the OneHotEncoder if the "
                "sparse is set to False."
            )
            if getattr(transformer, "sparse", False) or getattr(
                transformer, "sparse_output", False
            ):
                raise NotImplementedError(msg)

        self.transformer = transformer
        self.variables = _check_variables_input_value(variables)

    def fit(self, X: pd.DataFrame, y: Optional[str] = None):
        """
        Fits the Scikit-learn transformer to the selected variables.

        Parameters
        ----------
        X: Pandas DataFrame
            The dataset to fit the transformer.

        y: pandas Series, default=None
            The target variable.
        """

        # check input dataframe
        X = check_X(X)

        self.transformer_ = clone(self.transformer)

        if self.transformer_.__class__.__name__ in [
            "OneHotEncoder",
            "OrdinalEncoder",
            "SimpleImputer",
            "FunctionTransformer",
        ]:
            if self.variables is None:
                self.variables_ = find_all_variables(X)
            else:
                self.variables_ = check_all_variables(X, self.variables)

        else:
            if self.variables is None:
                self.variables_ = find_numerical_variables(X)
            else:
                self.variables_ = check_numerical_variables(X, self.variables)

        self.transformer_.fit(X[self.variables_], y)

        if self.transformer_.__class__.__name__ in _SELECTORS:
            # Find features to drop.
            selected = X[self.variables_].columns[self.transformer_.get_support()]
            self.features_to_drop_ = [f for f in self.variables_ if f not in selected]

        # save input features
        self.feature_names_in_ = X.columns.tolist()

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the transformation to the dataframe. Only the selected variables will be
        modified.

        If the Scikit-learn transformer is the OneHotEncoder or the  PolynomialFeatures,
        the new features will be concatenated to the input dataset.

        If the Scikit-learn transformer is for feature selection, the non-selected
        features will be dropped from the dataframe.

        For all other transformers, the original variables will be replaced by the
        transformed ones.

        Parameters
        ----------
        X: Pandas DataFrame
            The data to transform.

        Returns
        -------
        X_new: Pandas DataFrame
            The transformed dataset.
        """
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        # Check that input data contains same number of columns than
        # the dataframe used to fit the imputer.

        _check_X_matches_training_df(X, self.n_features_in_)

        # reorder df to match train set
        X = X[self.feature_names_in_]

        # Transformers that add features: creators
        if self.transformer_.__class__.__name__ in [
            "OneHotEncoder",
            "PolynomialFeatures",
        ]:
            new_features_df = pd.DataFrame(
                data=self.transformer_.transform(X[self.variables_]),
                columns=self.transformer_.get_feature_names_out(self.variables_),
                index=X.index,
            )
            X = pd.concat([X.drop(columns=self.variables_), new_features_df], axis=1)

        # Feature selection: transformers that remove features
        elif self.transformer_.__class__.__name__ in _SELECTORS:

            # return the dataframe with the selected features
            X.drop(columns=self.features_to_drop_, inplace=True)

        # Transformers that modify existing features
        else:
            X[self.variables_] = self.transformer_.transform(X[self.variables_])

        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert the transformed variables back to the original values. Only
        implemented for the following Scikit-learn transformers:

        PowerTransformer, QuantileTransformer, OrdinalEncoder,
        MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler.

        If you would like this method implemented for additional transformers,
        please check if they have the inverse_transform method in Scikit-learn and then
        raise an issue in our repo.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features].
            The transformed dataframe.

        Returns
        -------
        X_tr: pandas dataframe of shape = [n_samples, n_features].
            The dataframe with the original values.
        """
        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        if self.transformer_.__class__.__name__ not in _INVERSE_TRANSFORM:
            raise NotImplementedError(
                "The method `inverse_transform` is not implemented for this "
                "transformer. Supported transformers are {}.".format(
                    ", ".join(_INVERSE_TRANSFORM)
                )
            )
        # For safety, we check that the transformer has the method implemented.
        if hasattr(self.transformer_, "inverse_transform") and callable(
            self.transformer_.inverse_transform
        ):
            X[self.variables_] = self.transformer_.inverse_transform(X[self.variables_])
        else:
            raise NotImplementedError(
                "This Scikit-learn transformer does not have the method "
                "`inverse_transform` implemented."
            )
        return X

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
