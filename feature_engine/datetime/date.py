# Authors: dodoarg <eargiolas96@gmail.com>

from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np

from feature_engine.base_transformers import DateTimeBaseTransformer
from feature_engine.variable_manipulation import _check_input_parameter_variables
from feature_engine.dataframe_checks import _is_dataframe

class ExtractDateFeatures(DateTimeBaseTransformer):
    """
    ExtractDateFeatures extract various date-related features from
    datetime features present in the dataset, adding new columns
    accordingly.

    **Notes**
    Day of the month extraction isn't supported by panda modules.
    This transformer implements a workaround that counts how many blocks
    of 7 days have passed in a given month. 
    A more serious approach based on the calendar should be implemented
    ***

    Parameters
    ----------
    variables: list, default=None
        The list of variables to impute. If None, the imputer will find and
        select all datetime variables, including those converted from 
        features of type object/category.

    features_to_extract: list, default = "year"
        The list of date features to extract. See supported attribute

    drop_datetime: bool, default = "True"
        Whether to drop datetime variables from the dataframe, including
        those that have been converted by the transformer.
        Note: if you pass a subset of features via the variables argument
        of the transformer and drop_datetime=True, there might be more 
        datetime features in the dataframe that will not be dropped
        upon calling the transform method

    Attributes
    ----------
    variables:
        variable(s) as passed as argument when instantiating the transformer

    supported:
        list of supported date features

    Available upon calling the fit method:

    variables_:
        List of variables from which to extract date features

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Find the variables from which to extract date features
    transform:
        Add the new date features specified in features_to_extract argument
    fit_transform:
        Fit to the data, then transform it.
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        features_to_extract: Union[None, str, List[Union[str, int]]] = "year",
        drop_datetime: bool = True
    ) -> None:

        #get the list of supported features from a const variable somewhere?
        self.supported = [ 
            "month", "quarter", "semester",
            "year", "week_of_the_year",
            "day_of_the_week", "day_of_the_month",
            "is_weekend", "week_of_the_month"
        ] 
        self.variables  = _check_input_parameter_variables(variables)
        self.drop_datetime = drop_datetime
        
        if features_to_extract == "all":
            self.features_to_extract = self.supported
            return

        if isinstance(features_to_extract, str):
            features_to_extract = [features_to_extract]

        if any(feature not in self.supported
               for feature in features_to_extract):
            raise ValueError(
                "At least one of the requested feature is not supported. "
                "Supported features are {}.".format(', '.join(self.supported))
            )

        self.features_to_extract = features_to_extract #run the above checks somewhere else?


    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = super().fit(X,y) 
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        X = _is_dataframe(X)
        X = super().transform(X)

        #maybe iterate the following with func handles or smth
        #maybe rearrange final columns so that features extracted from 
        #the same variable are grouped together

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

        if "day_of_the_week" in self.features_to_extract:
            for var in self.variables_:
                X[var+"_dotw"] = X[var].dt.isocalendar().day.astype(np.int64)

        if "day_of_the_month" in self.features_to_extract:
            for var in self.variables_:
                X[var+"_dotm"] = X[var].dt.day

        #maybe add option to choose if friday should be considered a w.e. day?
        if "is_weekend" in self.features_to_extract:
            for var in self.variables_:
                X[var+"_is_weekend"] = np.where(X[var].dt.isocalendar().day<=5, False, True)

        if "week_of_the_month" in self.features_to_extract:
            for var in self.variables_:
                X[var+"_wotm"] = X[var].dt.day.apply(lambda d: (d-1)//7 + 1)

        if self.drop_datetime:
            X.drop(self.variables_, axis=1, inplace=True)

        return X