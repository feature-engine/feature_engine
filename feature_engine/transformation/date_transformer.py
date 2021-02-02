from typing import List, Optional, Union

import numpy as np
import pandas as pd

from feature_engine.base_transformers import BaseNumericalTransformer
from feature_engine.variable_manipulation import _check_input_parameter_variables


class date_feature_extractor(BaseNumericalTransformer):
    def __init__(
        self,
        features: Union[None, int, str, List[Union[str, int]]] = None,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
    ) -> None:

        if not fearures:
            raise ValueError(
                "Feature list cannot be empty.Enter features to extract from dates"
            )

        self.variables = _check_input_parameter_variables(variables)
        self.fearures = set(fearures)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):

        X = super().fit(X)

        # check if all the variables are of type datetime.
        if not all([pd.is_datetime64_any_dtype(X[col]) for col in X.columns]):
            raise ValueError("Some variables are not datetime cannot transform data")

        self.input_shape_ = X.shape

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)
        for col in self.variables:
            if "week" in self.fearures:
                X[col + "week"] = X[col].dt.week
            if "month" in self.fearures:
                X[col + "month"] = X[col].dt.month
            if "quarter" in self.fearures:
                X[col + "quarter"] = X[col].dt.quarter
            if "semester" in self.fearures:
                X[col + "semester"] = np.where(X[col].dt.quarter.isin([1, 2]), 1, 2)

        return X