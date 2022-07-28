# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine._docstrings.class_inputs import (
    _variables_categorical_docstring
)
from feature_engine._docstrings.fit_attributes import (
    _variables_attribute_docstring,
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_transform_docstring,
)

from feature_engine.dataframe_checks import (
    _check_X_matches_training_df,
    check_X,
    check_X_y
)
from feature_engine.encoding import WoEEncoder
from feature_engine.encoding._docstrings import (
    _errors_docstring,
    _ignore_format_docstring,
)
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_categorical_variables,
)

@Substitution(
    variables=_variables_categorical_docstring,
    errors=_errors_docstring,
    ignore_format=_ignore_format_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in=_feature_names_in_docstring,
    n_features_in=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
)
class InformationValue(BaseEstimator, TransformerMixin):
    """
    InformationValue() calculates the information value (IV) for each variable.
    The transformer is only compatible with categorical variables (type 'object'
    or 'categorical') and binary classification.

    You can pass a list of variables to score. Alternatively, the
    transformer will find and score all categorical variables (type 'object'
    or 'categorical').

    IV will allow you to assess each variable's independent contribution to
    the target variable and rank the variables in terms of their univariate
    predictive strength.

    Parameters
    ----------
    {variables}

    sort_values: bool, default=False
        Determines whether to sort the variables in ascending order after
        the information values are calculated.

    {errors}

    {ignore_format}

    Attributes
    ----------
    {variables_}

    {feature_names_in}

    {n_features_in}

    class_diff_encoder_dict_:
         The difference between the binomial distributions of positive and
         negative classes for each unique value of the selected categorical
         variables.

    woe_encoder_dict_:
        Contains the weights of evidence (WoE) for all the unique values of
        the selected categorical variables.

    information_values_:
        A dictionary comprised of the information values for the selected
        categorical variables.

    Methods
    -------
    fit:
        Calculates the information values for the selected categorical
        variables.

    {fit_transform}

    transform:
        Returns a dataframe with the variables and their corresponding
        information values.

    See Also
    --------
    feature_engine.encoding.WoEEncoder
    """

    def __init__(
            self,
            variables: Union[None, int, str, List[Union[str, int]]] = None,
            sort_values: bool = False,
            ignore_format: bool = False,
            errors: str = "ignore",
    ) -> None:

        if not isinstance(sort_values, bool):
            raise ValueError(
                "sort_values must be a boolean variable. Got {sort_values} instead."
            )

        self.variables = _check_input_parameter_variables(variables)
        self.sort_values = sort_values

        # parameters are checked when WoEEncoder is instatiated
        self.ignore_format = ignore_format
        self.errors = errors

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """
        Learn the information value.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the categorical variables.

        y: pandas series of shape = [n_samples, ]
            Target, must be binary.

        """
        # check input dataframe
        X, y = check_X_y(X, y)

        if y.nunique() != 2:
            raise ValueError(
                "This selector is designed for binary classification. The target "
                "used has more than 2 unique values."
            )

        # find categorical variables or check variables entered by user
        self.variables_ = _find_or_check_categorical_variables(X, self.variables)

        # derive the difference in the binomial distributions for each unique value
        # for each selected categorical variable
        self.class_diff_encoder_dict_ = self._calc_diff_between_class_distributions(
            X, y
        )

        # get WoE values for unique values of selected categorical variables
        self.woe_encoder_dict_ = self._calc_woe_encoder_dict(X, y)

        # get information values for unique values of selected categorical variables
        self.information_values_ = self._calc_information_values()

        self._get_feature_names_in(X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns information value for all the labels of each selected categorical
        feature.


        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features].
            The input dataframe.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_selected_features, 2]
            The selected features and corresponding information values.

        """

        # check if fit was performed prior to transform
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        # check if number of columns in test dataset matches to train dataset
        _check_X_matches_training_df(X, self.n_features_in_)

        X_new = pd.DataFrame.from_dict(
            data=self.information_values_,
            orient="index"
        ).reset_index()

        X_new.columns = ["variable", "information_value"]

        if self.sort_values:
            X_new.sort_values("information_value", ascending=False, inplace=True)

        return X_new

    def _calc_diff_between_class_distributions(
            self, X: pd.DataFrame, y: pd.Series
    ) -> Dict:
        """
        Returns a dictionary comprised of the categorical variables and the difference
        between the class distributions for each unique value.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the categorical variables.

        y: pandas series of shape = [n_samples, ]
            Target, must be binary.

        Returns
        -------
        encoder_dict: dict
            Difference between the class distributions for each unique value of
            the selected variables.

q       """
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ["target"]

        if any(label for label in y.unique() if label not in [0, 1]):
            temp["target"] = np.where(temp["target"] == y.unique()[0], 0, 1)

        # derive the difference in the binomial distributions for each unique value
        # for each selected categorical variable
        encoder_dict = {}

        total_pos = temp["target"].sum()
        total_neg = temp.shape[0] - total_pos
        temp["non_target"] = np.where(temp["target"] == 1, 0, 1)

        for var in self.variables_:
            pos = temp.groupby([var])["target"].sum() / total_pos
            neg = temp.groupby([var])["non_target"].sum() / total_neg

            temp_grouped = pd.concat([pos, neg], axis=1)
            temp_grouped["difference"] = (
                    temp_grouped["target"] - temp_grouped["non_target"]
            )

            encoder_dict[var] = temp_grouped["difference"].to_dict()

        return encoder_dict

    def _calc_woe_encoder_dict(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Learn and return the WoE encoder dictionary for the relevant variables.

        Parameters
        ----------
        None

        Returns
        -------
        iv_encoder_dict: dict
            The information values for each feature's unique values.
        """
        encoder = WoEEncoder(
            variables=self.variables_,
            ignore_format=self.ignore_format,
            errors=self.errors,
        )
        encoder.fit(X, y)

        return encoder.encoder_dict_

    def _calc_information_values(self) -> Dict:
        """
        Derive the information value for the selected categorical variables.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the categorical variables.

        y: pandas series of shape = [n_samples, ]
            Target, must be binary.

        Returns
        -------
        iv_encoder_dict: dict
            The information values for each feature's unique values.
        """

        info_val_dict = {var: {} for var in self.variables_}
        class_dist_diff_values = list(self.class_diff_encoder_dict_.values())
        woe_values = list(self.woe_encoder_dict_.values())

        # calcule information values for each variable's unique values
        for var, diff_dict, woe_dict in zip(
                self.variables_, class_dist_diff_values, woe_values
        ):
            for (key_diff, val_diff), (key_woe, val_woe) in zip(
                    diff_dict.items(), woe_dict.items()
            ):
                info_val_dict[var][key_diff] = val_diff * val_woe

        # sum the information values for each variable
        information_values = {var: 0 for var in self.variables_}

        for var2, iv_dict_sub in info_val_dict.items():
            for variable_label, value in iv_dict_sub.items():
                information_values[var2] += value

        return information_values

    def _get_feature_names_in(self, X):
        """Get the names and number of features in the training set. The dataframe
        used during fit."""

        self.feature_names_in_ = X.columns.to_list()
        self.n_features_in_ = X.shape[1]

        return self
