from typing import List, Union

import numpy as np
import pandas as pd

from feature_engine.dataframe_checks import check_X_y

from feature_engine.selection.base_selector import BaseSelector
from feature_engine.encoding import WoEEncoder
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_categorical_variables,
)


class InformationValue(BaseSelector):
    """


    See Also
    --------
    feature_engine.encoding.WoEEncoder
    """

    def __init__(
            self,
            variables: Union[None, int, str, List[Union[str, int]]] = None,
            confirm_variables: bool = False,
            ignore_format: bool = False,
            errors: str = "ignore",
    ) -> None:

        super().__init__(confirm_variables)
        self.variables = _check_input_parameter_variables(variables)
        # parameters are checked when WoEEncoder is instatiated
        self.ignore_format = ignore_format
        self.errors = errors

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Learn the information value.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the categorical variables.

        y: pandas series.
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

        # find categorical variables or check variables entered by user
        self.variables_ = _find_or_check_categorical_variables(X, self.variables)

        # get WoE values for values of selected categorical variables
        X_woe = self._calc_woe(X, y)

        # if target does not have values 0 and 1, remap values to compute the average
        # if any(label for label in y.unique() if label not in [0, 1]):

        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns information value for all the labels of each selected categorical feature.


        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features].
            The input dataframe.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_selected_features]
            Pandas dataframe with the selected features.

        """

        X = super().transform(X)

        pass

    def _calc_diff_between_class_distributions(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Returns a dictionary comprised of the categorical variables and the difference
        between the class distributions for each unique value.

        """
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

            temp_grouped = pd.([pos, neg], axis=1)
            temp_grouped["difference"] = temp_grouped["target"] - temp_grouped["non_target"]

            encoder_dict[var] = temp_grouped["difference"].to_dict()

        return encoder_dict

    def _calc_woe_encoder_dict(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Learn and return the WoE encoder dictionary for the relevant variables.

        """
        encoder = WoEEncoder(
            variables=self.variables_,
            ignore_format=self.ignore_format,
            errors=self.errors,
        )
        encoder.fit(X, y)

        return encoder.encoder_dict_
