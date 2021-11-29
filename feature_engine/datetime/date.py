# Authors: dodoarg <eargiolas96@gmail.com>

from typing import Dict, List, Optional, Union

import pandas as pd

from feature_engine.base_transformers import DateTimeBaseTransformer
from feature_engine.validation import _return_tags
from feature_engine.variable_manipulation import _check_input_parameter_variables
from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_input_matches_training_df,
    _is_dataframe,
)

class ExtractDateFeatures(DateTimeBaseTransformer):
    """
    placeholder
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None
    ) -> None:

        self.variables = _check_input_parameter_variables(variables)
        

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = super().fit(X,y) #should check if variables are datetime
        self.n_features_in_ = X.shape[1]
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        #should I call check_is_fitted?

        X = _is_dataframe(X)
        X = super().transform(X)
        
        for var in self.variables_:
            X[var+"_month"] = X[var].dt.month

        return X