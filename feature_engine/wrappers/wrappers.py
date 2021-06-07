from typing import List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone

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


class SklearnTransformerWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper to apply Scikit-learn transformers to a selected group of variables. It
    works with transformers like the SimpleImputer, OrdinalEncoder, OneHotEncoder, all
    the scalers and also the transformers for feature selection.

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

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Fit Scikit-learn transformer
    transform:
        Transform data with Scikit-learn transformer
    fit_transform:
        Fit to data, then transform it.
    """

    def __init__(
        self,
        transformer,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
    ) -> None:

        if not issubclass(transformer.__class__, BaseEstimator):
            raise TypeError(
                "transformer expected a Scikit-learn transformer, "
                f"got {transformer} instead."
            )

        self.transformer = transformer
        self.variables = _check_input_parameter_variables(variables)

    def fit(self, X: pd.DataFrame, y: Optional[str] = None):
        """
        Fits the Scikit-learn transformer to the selected variables.

        If you enter None in the variables parameter, all variables will be
        automatically transformed by the OneHotEncoder, OrdinalEncoder or
        SimpleImputer. For the rest of the transformers, only the numerical variables
        will be selected and transformed.

        If you enter a list in the variables attribute, the SklearnTransformerWrapper
        will check that those variables exist in the dataframe and are of type
        numeric for all transformers except the OneHotEncoder, OrdinalEncoder or
        SimpleImputer, which also accept categorical variables.

        Parameters
        ----------
        X: Pandas DataFrame
            The dataset to fit the transformer
        y: pandas Series, default=None
            The target variable.

        Raises
        ------
         TypeError
            If the input is not a Pandas DataFrame

        Returns
        -------
        self
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
                "set its sparse attribute to False"
            )

        if self.transformer_.__class__.__name__ in [
            "OneHotEncoder",
            "OrdinalEncoder",
            "SimpleImputer",
        ]:
            self.variables_ = _find_all_variables(X, self.variables)

        else:
            self.variables_ = _find_or_check_numerical_variables(X, self.variables)

        self.transformer_.fit(X[self.variables_], y)

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the transformation to the dataframe. Only the selected variables will be
        modified.

        If transformer is the OneHotEncoder, the dummy features will be concatenated
        to the input dataset. Note that the original categorical variables will not be
        removed from the dataset after encoding. If this is the desired effect, please
        use Feature-engine's OneHotEncoder instead.

        Parameters
        ----------
        X: Pandas DataFrame
            The data to transform

        Raises
        ------
         TypeError
            If the input is not a Pandas DataFrame

        Returns
        -------
        X: Pandas DataFrame
            The transformed dataset.
        """

        # check that input is a dataframe
        X = _is_dataframe(X)

        # Check that input data contains same number of columns than
        # the dataframe used to fit the imputer.

        _check_input_matches_training_df(X, self.n_features_in_)

        if self.transformer_.__class__.__name__ == "OneHotEncoder":
            ohe_results_as_df = pd.DataFrame(
                data=self.transformer_.transform(X[self.variables_]),
                columns=self.transformer_.get_feature_names(self.variables_),
            )
            X = pd.concat([X, ohe_results_as_df], axis=1)

        elif self.transformer_.__class__.__name__ in [
            "SelectKBest",
            "SelectPercentile",
            "SelectFromModel",
        ]:

            # the variables selected by the transformer
            selected_variables = X.columns[self.transformer_.get_support(indices=True)]

            # the variables that were not examined, in case there are any
            remaining_variables = [
                var for var in X.columns if var not in self.variables_
            ]

            X = X[list(selected_variables) + list(remaining_variables)]

        else:
            X[self.variables_] = self.transformer_.transform(X[self.variables_])

        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        # add additional test that fails
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"
        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has 1 mandatory parameter"
        return tags_dict
