# Authors: Pradumna Suryawanshi <pradumnasuryawanshi@gmail.com>
# License: BSD 3 clause

from typing import Optional, List, Union

import pandas as pd

from feature_engine.dataframe_checks import _is_dataframe
from feature_engine.imputation.base_imputer import BaseImputer
from feature_engine.parameter_checks import _define_numerical_dict
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_numerical_variables,
)




class Na_transformer(BaseImputer):

    def __init__(
        self,
        missing_only: bool = True,
        variables: Union[None, int, str, List[Union[str, int]]] = None
    ) -> None:

        if not isinstance(missing_only, bool):
            raise ValueError("missing_only takes values True or False")

        self.variables = _check_input_parameter_variables(variables)
        self.missing_only = missing_only


    
    def return_na(self):
        return self.dropped_data_



    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        # check input dataframe
        X = _is_dataframe(X)

        # find variables for which indicator should be added
        if self.missing_only:
            if not self.variables:
                self.variables_ = [
                    var for var in X.columns if X[var].isnull().sum() > 0
                ]
            else:
                self.variables_ = [
                    var for var in self.variables if X[var].isnull().sum() > 0
                ]

        else:
            if not self.variables:
                self.variables_ = [var for var in X.columns]
            else:
                self.variables_ = self.variables

        self.input_shape_ = X.shape

        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        X = self._check_transform_input_and_state(X)


        ser=pd.isnull(X[self.variables_]).any(1)
        new=ser[ser==True]
        self.dropped_data_=X[X.index.isin(new.index)].copy()

        X.dropna(axis=0,how='any',subset=self.variables, inplace=True)

        return X


