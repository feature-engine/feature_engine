from typing import Dict, List, Union

import pandas as pd
from numpy import ndarray
from numpy.typing import ArrayLike
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    check_X,
    check_X_y,
)
from feature_engine.variable_handling import check_numerical_variables


class TransformXyMixin:
    def transform_x_y(self, X: pd.DataFrame, y: pd.Series):
        """
        Transform, align and adjust both X and y based on the transformations applied
        to X, ensuring that they correspond to the same set of rows if any were
        removed from X.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The dataframe to transform.

        y: pandas Series or Dataframe of length = n_samples
            The target variable to transform. Can be multi-output.

        Returns
        -------
        X_new: pandas dataframe
            The transformed dataframe of shape [n_samples - n_rows, n_features]. It may
            contain less rows than the original dataset.

        y_new: pandas Series or DataFrame
            The transformed target variable of length [n_samples - n_rows]. It contains
            as many rows as those left in X_new.
        """
        X, y = check_X_y(X, y)
        X = self.transform(X)
        y = y.loc[X.index]
        return X, y


class FitFromDictMixin:
    def _fit_from_dict(self, X: pd.DataFrame, user_dict_: Dict) -> pd.DataFrame:
        """
        Checks that input is a dataframe, checks that variables in the dictionary
        entered by the user are of type numerical.

        Parameters
        ----------
        X : Pandas DataFrame

        user_dict_ : Dictionary. Default = None
            Any dictionary allowed by the transformer and entered by user.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame or a numpy array
            If any of the variables in the dictionary are not numerical
        ValueError
            If there are no numerical variables in the df or the df is empty
            If the variable(s) contain null values

        Returns
        -------
        X : Pandas DataFrame
            The same dataframe entered as parameter
        """
        # check input dataframe
        X = check_X(X)

        # find or check for numerical variables
        variables = list(user_dict_.keys())
        self.variables_ = check_numerical_variables(X, variables)

        # check if dataset contains na or inf
        _check_contains_na(X, self.variables_)
        _check_contains_inf(X, self.variables_)

        # save input features
        self.feature_names_in_ = X.columns.tolist()

        # save train set shape
        self.n_features_in_ = X.shape[1]

        return X


class GetFeatureNamesOutMixin:
    def get_feature_names_out(
        self,
        input_features: Union[List[Union[str, int]], ArrayLike] = None,
    ) -> List[Union[str, int]]:
        """
        Get output feature names for transformation. In other words, returns the
        variable names of transformed dataframe.

        Parameters
        ----------
        input_features : array or list, default=None
            This parameter exits only for compatibility with the Scikit-learn pipeline.

            - If `None`, then `feature_names_in_` is used as feature names in.
            - If an array or list, then `input_features` must match `feature_names_in_`.

        Returns
        -------
        feature_names_out: list
            Transformed feature names.
        """
        check_is_fitted(self)

        if input_features is not None:
            # If input to fit is an array, then the variable names in
            # feature_names_in_ are "x0", "x1","x2" ..."xn".
            if self.feature_names_in_ == [f"x{i}" for i in range(self.n_features_in_)]:

                # If the input was an array, we let the user enter the variable names.
                if len(input_features) == self.n_features_in_:
                    if isinstance(input_features, list):
                        feature_names = input_features
                    else:
                        feature_names = list(input_features)

                    # For transformers that add features to the data.
                    feature_names = self._add_new_feature_names(feature_names)

                    # For transformers that remove features from data, i..e, selectors.
                    feature_names = self._remove_feature_names(
                        feature_names, indices=True
                    )

                    return feature_names

                else:
                    raise ValueError(
                        "The number of input_features does not match the number of "
                        "features seen in the dataframe used in fit."
                    )
            else:
                msg = "input_features is not equal to feature_names_in_"
                if isinstance(input_features, list):
                    if input_features != self.feature_names_in_:
                        raise ValueError(msg)
                elif isinstance(input_features, ndarray) or isinstance(
                    input_features, pd.core.indexes.base.Index
                ):
                    if list(input_features) != self.feature_names_in_:
                        raise ValueError(msg)
                else:
                    raise ValueError(
                        "input_features must be a list or an array. "
                        "Got {input_features} instead."
                    )

        feature_names = self.feature_names_in_

        # For transformers that add features to the dataframe:
        feature_names = self._add_new_feature_names(feature_names)

        # For transformers that remove features from data, i..e, selectors.
        feature_names = self._remove_feature_names(feature_names, indices=False)

        return feature_names

    def _add_new_feature_names(self, feature_names):
        # For transformers that add features to the dataframe:
        if hasattr(self, "_get_new_features_name") and callable(
            self._get_new_features_name
        ):
            feature_names = feature_names + self._get_new_features_name()

            if self.drop_original is True and self.variables_ is not None:
                # Remove names of variables to drop.
                feature_names = [f for f in feature_names if f not in self.variables_]

        return feature_names

    def _remove_feature_names(self, feature_names, indices=False) -> List:
        # For transformers that remove features from data, i..e, selectors.
        if hasattr(self, "features_to_drop_"):
            if indices is True:
                mask = self.get_support(indices=True)
                feature_names = [feature_names[i] for i in mask]
            else:
                feature_names = [
                    f for f in feature_names if f not in self.features_to_drop_
                ]
        return feature_names
