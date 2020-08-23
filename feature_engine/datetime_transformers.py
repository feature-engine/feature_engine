
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from feature_engine.dataframe_checks import _is_dataframe, _check_input_matches_training_df
from feature_engine.variable_manipulation import _define_variables

import pandas as pd
import numpy as np
import types
from typing import Union


class DateTimeTransformer(BaseEstimator, TransformerMixin):
    """

    """

    # global config
    feature_map = {
        "year": "year",
        "month": "month",
        "quarter": "quarter",
        "semester": lambda data, var: np.where(data[f"{var}_quarter"].isin([1, 2]), 1, 2),
        "day": "day",
        "day_of_week": "dayofweek",
        "is_weekend": lambda data, var: np.where(data[f"{var}_day_of_week"].isin([5, 6]), 1, 0),
        "hr": "hour",
        "min": "minute",
        "sec": "second",
    }

    def __init__(self, variables: Union[str, list] = None, errors: str = "raise",
                 features_to_add: Union[dict, list] = "default", keep_original: bool = False):
        self.variables = _define_variables(variables)
        self.errors = errors
        self.keep_original = keep_original
        if len(self.variables) == 0:
            raise ValueError("Variables cannot be empty. Please pass at least one variable to transform")

        # if features_to_add is empty list or dict
        if not features_to_add:
            raise ValueError("features_to_add cannot be empty, Please pass at lease one feature to add")

        # if features_to_add is dict then look for values
        if isinstance(features_to_add, dict):
            # check for type
            if not all([isinstance(features_to_add[x], list) for x in features_to_add.keys()]):
                raise ValueError("features_to_drop if dict type, should have lists as values")
            # check if features are supported for each key
            if not all(all(y in self.feature_map.keys() for y in features_to_add[x]) for x in features_to_add.keys()):
                raise ValueError(f"Some of the features in features_to_add are not supported. "
                                 f"supported features are {', '.join(self.feature_map.keys())}")
            # check if a col in variables but not in features_to_add
            if any([x for x in variables if x not in features_to_add.keys()]):
                raise ValueError(
                    f"columns are inconsistent between variables and features_to_add "
                    f"please verify that all the columns in variables are passed to features_to_add"
                )
            # assign
            self.features_to_add = features_to_add
        elif isinstance(features_to_add, list):
            # inspect the values
            if not all([x in self.feature_map.keys() for x in features_to_add]):
                raise ValueError(f"Some of the features in features_to_add are not supported. "
                                 f"supported features are {', '.join(self.feature_map.keys())}")
            # assign and make it a dict for easy parsing later
            self.features_to_add = {}
            for col in self.variables:
                self.features_to_add[col] = features_to_add

        # default case
        elif features_to_add == "default":
            self.features_to_add = {}
            for col in self.variables:
                self.features_to_add[col] = list(self.feature_map.keys())
        else:
            raise ValueError("features_to_drop has to be of type list or dict")

    def fit(self, X, y=None):
        """

        Parameters
        ----------
        X
        y

        Returns
        -------

        """
        # check input dataframe
        X = _is_dataframe(X)

        # check for non existent columns in both variables and features_to_add
        non_existent = [x for x in self.variables if x not in X.columns]
        if non_existent:
            raise KeyError(
                f"Columns '{', '.join(non_existent)}' not in the input dataframe, "
                f"please check the columns and enter a new list of features to transform"
            )

        for col in self.variables:
            try:
                # explicit cast
                X[col] = pd.to_datetime(X[col], errors=self.errors)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Cannot cast variable {col} to datetime, given string not likely a valid datetime"
                )
        # add input shape
        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """

        Parameters
        ----------
        X

        Returns
        -------

        """
        # check if fit is called prior
        check_is_fitted(self)

        # check input dataframe
        X = _is_dataframe(X)

        # check for input consistency
        _check_input_matches_training_df(X, self.input_shape_[1])

        for col in self.variables:
            # start decomposing
            for k in self.features_to_add[col]:
                try:
                    X[f"{col}_{k}"] = self.feature_map[k](X, col) if isinstance(self.feature_map[k], types.LambdaType) \
                        else getattr(pd.DatetimeIndex(data=X[col]), self.feature_map[k])
                except KeyError as ex:
                    # gather the missing keys that has dependencies
                    if str(ex) in [f"'{col}_day_of_week'", f"'{col}_quarter'"]:
                        missing_key = str(ex)[1+len(col)+1:-1]
                        X[f"{col}_{missing_key}"] = getattr(pd.DatetimeIndex(data=X[col]), self.feature_map[missing_key])
                        # apply decomposition
                        X[f"{col}_{k}"] = self.feature_map[k](X, col) if isinstance(self.feature_map[k], types.LambdaType) \
                            else getattr(pd.DatetimeIndex(data=X[col]), self.feature_map[k])
                        # if not in features_to_add then drop
                        if f"{col}_{missing_key}" not in self.features_to_add[col]:
                            X = X.drop(columns=[f"{col}_{missing_key}"])
                    else:
                        raise ex

            # drop original unless specified otherwise
            if not self.keep_original:
                X = X.drop(columns=[col])
        return X


