# Authors: Kyle Gilde <kylegilde@gmail.com>

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from feature_engine.creation.base_creation import BaseCreation
from feature_engine._docstrings.methods import (
    _fit_not_learn_docstring,
    _fit_transform_docstring,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.class_inputs import (
    _drop_original_docstring,
    _missing_values_docstring,
)

from feature_engine._docstrings.substitute import Substitution
from feature_engine.variable_manipulation import _find_or_check_datetime_variables


@Substitution(
    missing_values=_missing_values_docstring,
    drop_original=_drop_original_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    transform=BaseCreation._transform_docstring,
    fit_transform=_fit_transform_docstring,
)
class RelativeFeatures(BaseCreation):
    """
    DatetimeSubtraction() applies datetime subtraction between a group
    of variables and one or more reference features. It adds one or more additional
    features to the dataframe with the result of the operations.

    In other words, DatetimeSubtraction() subtracts a group of features from a group of
    reference variables, and returns the result as new variables in the dataframe.

    The transformed dataframe will contain the additional features indicated in the
    new_variables_name list plus the original set of variables.

    More details in the :ref:`User Guide <datetime_subtraction>`.

    Parameters
    ----------
    variables: list
        The list of datetime variables that the reference variables will be subtracted
        from.

    reference: list
        The list of datetime reference variables that will be subtracted from the
        `variables`.

    output_unit: string, default='D'
        The string representation of the output unit of the datetime differences.
        The default is `D` for day. This parameter is passed to numpy.timedelta64.
        Other possible values are  `Y` for year, `M` for month,  `W` for week,
        `h` for hour, `m` for minute, `s` for second, `ms` for millisecond,
        `us` or `μs` for microsecond, `ns` for nanosecond, `ps` for picosecond,
        `fs` for femtosecond and `as` for attosecond.

    {missing_values}

    {drop_original}

    Attributes
    ----------
    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

    {transform}

    """

    def __init__(
        self,
        variables: List[Union[str, int]],
        reference: List[Union[str, int]],
        output_unit: str = 'D',
        missing_values: str = "ignore",
        drop_original: bool = False,
    ) -> None:

        if (
            not isinstance(variables, list)
            or not all(isinstance(var, (int, str)) for var in variables)
            or len(set(variables)) != len(variables)
        ):
            raise ValueError(
                "variables must be a list of strings or integers. "
                f"Got {variables} instead."
            )

        if (
            not isinstance(reference, list)
            or not all(isinstance(var, (int, str)) for var in reference)
            or len(set(reference)) != len(reference)
        ):
            raise ValueError(
                "reference must be a list of strings or integers. "
                f"Got {reference} instead."
            )

        valid_output_units = {'D', 'Y', 'M', 'W', 'h', 'm', 's', 'ms', 'us', 'μs', 'ns',
                              'ps', 'fs', 'as'}

        if output_unit not in valid_output_units:
            raise ValueError(f"output_unit accepts the following values: "
                             f"{valid_output_units}")

        super().__init__(missing_values, drop_original)
        self.variables = variables
        self.reference = reference
        self.output_unit = output_unit

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn any parameter.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: pandas Series, or np.array. Default=None.
            It is not needed in this transformer. You can pass y or None.
        """
        # Common checks and attributes
        X = super().fit(X, y)

        # check variables are datetime
        self.reference = _find_or_check_datetime_variables(X, self.reference)
        self.variables = _find_or_check_datetime_variables(X, self.variables)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add new features.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: Pandas dataframe
            The input dataframe plus the new variables.
        """

        X = super().transform(X)

        self._sub(X)

        if self.drop_original:
            X.drop(
                columns=set(self.variables + self.reference),
                inplace=True,
            )

        return X

    def _sub(self, X):

        for reference in self.reference:
            varname = [f"{var}_sub_{reference}" for var in self.variables]
            X[varname] = (
                X[self.variables].sub(X[reference], axis=0)
                .apply(lambda s: s / np.timedelta64(1, self.output_unit))
            )

        return X

    def get_feature_names_out(self, input_features: Optional[bool] = None) -> List:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features: bool, default=None
            If `input_features` is `None`, then the names of all the variables in the
            transformed dataset (original + new variables) is returned. Alternatively,
            if `input_features` is True, only the names for the new features will be
            returned.

        Returns
        -------
        feature_names_out: list
            The feature names.
        """
        check_is_fitted(self)

        if input_features is not None and not isinstance(input_features, bool):
            raise ValueError(
                "input_features takes None or a boolean, True or False. "
                f"Got {input_features} instead."
            )

        # Names of new features
        feature_names = []
        for reference in self.reference:
            varname = [f"{var}_sub_{reference}" for var in self.variables]
            feature_names.extend(varname)

        if input_features is None or input_features is False:
            if self.drop_original is True:
                # Remove names of variables to drop.
                original = [
                    f
                    for f in self.feature_names_in_
                    if f not in self.variables + self.reference
                ]
                feature_names = original + feature_names
            else:
                feature_names = self.feature_names_in_ + feature_names

        return feature_names