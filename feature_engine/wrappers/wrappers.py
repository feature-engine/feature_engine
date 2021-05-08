from typing import List, Optional, Union

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    SelectFromModel,
)

from feature_engine.dataframe_checks import (
    _check_input_matches_training_df,
    _is_dataframe,
)

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
    transformer : sklearn transformer, default=None
        The desired Scikit-learn transformer. If None, it defaults to SimpleImputer().

    variables : list, default=None
        The list of variables to be transformed. If None, the wrapper will select all
        variables of type numeric for all transformers except the SimpleImputer,
        OrdinalEncoder and OneHotEncoder, in which case it will select all variables in
        the dataset.

    Attributes
    ----------
    transformer_:
        The fitted Scikit-learn transformer.

    variables_:
        The group of variables that will be transformed.

    Methods
    -------
    fit:
        Fit Scikit-learn transformers
    transform:
        Transforms with Scikit-learn transformers
    fit_transform:
        Fit to data, then transform it.
    """

    def __init__(
            self,
            transformer=None,
            variables: Union[None, int, str, List[Union[str, int]]] = None,
    ) -> None:
        self.variables = _check_input_parameter_variables(variables)
        self.transformer = transformer

    def fit(self, X: pd.DataFrame, y: Optional[str] = None):
        """
        Fits the Scikit-learn transformers to the selected variables.

        If the user entered None in the variables parameter, all variables will be
        automatically transformed by the OneHotEncoder, OrdinalEncoder or
        SimpleImputer. For the rest of the transformers, only the numerical variables
        will be selected and transformed.

        If the user entered a list in the variables attribute, the SklearnWrapper will
        check that those variables exist in the dataframe and are of type numerical,
        for all transformers except OneHotEncoder, OrdinalEncoder or SimpleImputer.

        Parameters
        ----------
        X : Pandas DataFrame
            The dataset to fit the transformer
        y : pandas Series, default=None
            This parameter exists only for compatibility with Pipeline.

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

        if self.transformer is None:
            self.transformer_ = SimpleImputer()
        else:
            self.transformer_ = clone(self.transformer)

        if isinstance(self.transformer_, OneHotEncoder) and self.transformer_.sparse:
            raise AttributeError(
                "The SklearnTransformerWrapper can only wrap the OneHotEncoder if you "
                "set its sparse attribute to False"
            )

        if isinstance(self.transformer_,
                      (OneHotEncoder, OrdinalEncoder, SimpleImputer)):
            self.variables_ = _find_all_variables(X, self.variables)

        else:
            self.variables_ = _find_or_check_numerical_variables(X, self.variables)

        self.transformer_.fit(X[self.variables_], y)

        self.input_shape_ = X.shape

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the transformation to the dataframe. Only the selected varriables will be
        modified.

        If transformer is the OneHotEncoder, the dummy features will be concatenated
        to the input dataset. Note that the original categorical variables will not be
        removed from the dataset after encoding. If this is the desired effect, please
        use Feature-engine's OneHotEncoder instead.

        Parameters
        ----------
        X : Pandas DataFrame
            The data to transform

        Raises
        ------
         TypeError
            If the input is not a Pandas DataFrame

        Returns
        -------
        X : Pandas DataFrame
            The transformed dataset.
        """

        # check that input is a dataframe
        X = _is_dataframe(X)

        # Check that input data contains same number of columns than
        # the dataframe used to fit the imputer.

        _check_input_matches_training_df(X, self.input_shape_[1])

        if isinstance(self.transformer_, OneHotEncoder):
            ohe_results_as_df = pd.DataFrame(
                data=self.transformer_.transform(X[self.variables_]),
                columns=self.transformer_.get_feature_names(self.variables_),
            )
            X = pd.concat([X, ohe_results_as_df], axis=1)

        elif isinstance(self.transformer_,
                        (SelectKBest, SelectPercentile, SelectFromModel)):

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
