# Authors: dodoarg <eargiolas96@gmail.com>

from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np

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

    Defaults to extracting the year only?
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        features_to_extract: Union[None, str, List[Union[str, int]]] = "year"
    ) -> None:

        self.supported = ["month", "quarter", "semester", "year", "week_of_the_year"] 
        self.variables  = _check_input_parameter_variables(variables)

        if features_to_extract:
            if isinstance(features_to_extract, str):
                features_to_extract = [features_to_extract]
            if any(feature not in self.supported
                   for feature in features_to_extract):
                raise ValueError(
                    "At least one of the requested feature is not supported. "
                    "Supported features are {}.".format(', '.join(self.supported))
                )
        self.features_to_extract = features_to_extract #run the above checks somewhere else


    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = super().fit(X,y) #should check if variables are datetime
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        #should I call check_is_fitted?

        X = _is_dataframe(X)
        X = super().transform(X)
        
        #maybe iterate the following with func handles or smth
        if "month" in self.features_to_extract:
            for var in self.variables_:
                X[var+"_month"]    = X[var].dt.month

        if "quarter" in self.features_to_extract:
            for var in self.variables_:
                X[var+"_quarter"]  = X[var].dt.quarter
        
        if "semester" in self.features_to_extract:
            for var in self.variables_:
                X[var+"_semester"] = np.where(X[var].dt.month <= 6, 1, 2).astype(np.int64)
            
        if "year" in self.features_to_extract:
            for var in self.variables_:
                X[var+"_year"]     = X[var].dt.year

        if "week_of_the_year" in self.features_to_extract:
            for var in self.variables_:
                X[var+"_woty"] = X[var].dt.isocalendar().week.astype(np.int64)

        return X