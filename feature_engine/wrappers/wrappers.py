from typing import List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_input_matches_training_df,
    _is_dataframe,
)
from feature_engine.validation import _return_tags
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_all_variables,
    _find_or_check_numerical_variables,
)

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
    "MissingIndicator",
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


class SklearnTransformerWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper to apply Scikit-learn transformers to a selected group of variables. It
    works with transformers like the SimpleImputer, OrdinalEncoder, OneHotEncoder, all
    the scalers and also the transformers for feature selection.

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

    feature_names_in_:
        List with the names of features seen during `fit`.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Fit Scikit-learn transformer

    transform:
        Transform data with the Scikit-learn transformer

    fit_transform:
        Fit to data, then transform it.

    Notes
    -----
    This transformer offers similar functionality to the ColumnTransformer from
    Scikit-learn, but it allows entering the transformations directly into a
    Pipeline.

    See Also
    --------
    sklearn.compose.ColumnTransformer
    """

    def __init__(
        self,
        transformer,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
    ) -> None:

        if not issubclass(transformer.__class__, TransformerMixin):
            raise TypeError(
                "transformer expected a Scikit-learn transformer. "
                f"got {transformer} instead."
            )

        if transformer.__class__.__name__ not in _ALL_TRANSFORMERS:
            raise NotImplementedError(
                "This transformer is not compatible with the wrapper. "
                "Supported transformers are {}.".format(", ".join(_ALL_TRANSFORMERS))
            )

        self.transformer = transformer
        self.variables = _check_input_parameter_variables(variables)

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
        X = _is_dataframe(X)

        self.transformer_ = clone(self.transformer)

        if (
            self.transformer_.__class__.__name__ == "OneHotEncoder"
            and self.transformer_.sparse
        ):
            raise AttributeError(
                "The SklearnTransformerWrapper can only wrap the OneHotEncoder if you "
                "set its sparse attribute to False."
            )

        if self.transformer_.__class__.__name__ in [
            "OneHotEncoder",
            "OrdinalEncoder",
            "SimpleImputer",
            "MissingIndicator",
        ]:
            self.variables_ = _find_all_variables(X, self.variables)

        else:
            self.variables_ = _find_or_check_numerical_variables(X, self.variables)

        self.transformer_.fit(X[self.variables_], y)

        # save input features
        self.feature_names_in_ = X.columns.tolist()

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the transformation to the dataframe. Only the selected variables will be
        modified.

        **Note**

        If the Scikit-learn transformer is the OneHotEncoder, the dummy features will
        be concatenated to the input dataset. Note that the original categorical
        variables will not be removed from the dataset after encoding. If this is the
        desired effect, please use Feature-engine's OneHotEncoder instead.

        Parameters
        ----------
        X: Pandas DataFrame
            The data to transform

        Returns
        -------
        X_new: Pandas DataFrame
            The transformed dataset.
        """
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # Check that input data contains same number of columns than
        # the dataframe used to fit the imputer.

        _check_input_matches_training_df(X, self.n_features_in_)

        # reorder df to match train set
        X = X[self.feature_names_in_]

        # Transformers that add features
        if self.transformer_.__class__.__name__ in _CREATORS:
            new_features_df = pd.DataFrame(
                data=self.transformer_.transform(X[self.variables_]),
                columns=self.transformer_.get_feature_names_out(self.variables_),
                index=X.index,
            )
            X = pd.concat([X, new_features_df], axis=1)

        # Feature selection
        elif self.transformer_.__class__.__name__ in _SELECTORS:

            # the variables that will be dropped
            features_to_drop = [
                f
                for f in self.variables_
                if f not in self.transformer_.get_feature_names_out()
            ]

            # return the dataframe with the selected features
            X.drop(columns=features_to_drop)

        else:
            X[self.variables_] = self.transformer_.transform(X[self.variables_])

        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert the encoded variable back to the original values. Only
        works if the Scikit-learn transformer has the method implemented.

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
        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        if hasattr(self.transformer_, "inverse_transform") and callable(
            self.transformer_.inverse_transform
        ):
            X = self.transformer_.inverse_transform(X)
        else:
            raise NotImplementedError(
                "This Scikit-learn transformer does not have the method "
                "`inverse_transform` implemented."
            )
        return X

    def get_feature_names_out(self, input_features: Optional[List] = None) -> List:
        """Get output feature names for transformation. Only works if the Scikit-learn
        transformer has the method implemented.

        Returns
        -------
        feature_names_out: list
            The feature names.
        """
        # Check method fit has been called
        check_is_fitted(self)

        if hasattr(self.transformer_, "get_feature_names_out") and callable(
            self.transformer_.get_feature_names_out
        ):
            return self.transformer_.get_feature_names_out(input_features)
        else:
            raise NotImplementedError(
                "This Scikit-learn transformer does not have the method "
                "`get_feature_names_out` implemented."
            )

    def _more_tags(self):
        tags_dict = _return_tags()
        # add additional test that fails
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"
        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has 1 mandatory parameter"
        return tags_dict
