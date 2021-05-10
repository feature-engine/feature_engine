# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from feature_engine.base_transformers import BaseNumericalTransformer
from feature_engine.variable_manipulation import _check_input_parameter_variables


class ReciprocalTransformer(BaseNumericalTransformer):
    """
    The ReciprocalTransformer() applies the reciprocal transformation 1 / x
    to numerical variables.

    The ReciprocalTransformer() only works with numerical variables with non-zero
    values. If a variable contains the value 0, the transformer will raise an error.

    A list of variables can be passed as an argument. Alternatively, the
    transformer will automatically select and transform all numerical
    variables.

    Parameters
    ----------
    variables : list, default=None
        The list of numerical variables to transform. If None, the transformer will
        automatically find and select all numerical variables.

    Attributes
    ----------
    variables_:
        The group of variables that will be transformed.

    n_features_in_:
        The number of features in the train set used in fit

    Methods
    -------
    fit:
        This transformer does not learn parameters.
    transform:
        Apply the reciprocal 1 / x transformation.
    fit_transform:
        Fit to data, then transform it.
    """

    def __init__(
        self, variables: Union[None, int, str, List[Union[str, int]]] = None
    ) -> None:

        self.variables = _check_input_parameter_variables(variables)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn parameters.

        Parameters
        ----------
        X : Pandas DataFrame of shape = [n_samples, n_features].
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y : pandas Series, default=None
            It is not needed in this transformer. You can pass y or None.

        Raises
        ------
        TypeError
            - If the input is not a Pandas DataFrame
            - If any of the user provided variables are not numerical
        ValueError
            - If there are no numerical variables in the df or the df is empty
            - If the variable(s) contain null values
            - If some variables contain zero as values

        Returns
        -------
        self
        """

        # check input dataframe
        X = super().fit(X)

        # check if the variables contain the value 0
        if (X[self.variables_] == 0).any().any():
            raise ValueError(
                "Some variables contain the value zero, can't apply reciprocal "
                "transformation."
            )

        self.input_shape_ = X.shape
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the reciprocal 1 / x transformation.

        Parameters
        ----------
        X : Pandas DataFrame of shape = [n_samples, n_features]
            The data to be transformed.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            - If the variable(s) contain null values.
            - If the dataframe not of the same size as that used in fit().
            - If some variables contain zero values.

        Returns
        -------
        X : pandas dataframe
            The dataframe with the transformed variables.
        """

        # check input dataframe and if class was fitted
        X = super().transform(X)

        # check if the variables contain the value 0
        if (X[self.variables_] == 0).any().any():
            raise ValueError(
                "Some variables contain the value zero, can't apply reciprocal "
                "transformation."
            )

        # transform
        # for some reason reciprocal does not work with integers
        X.loc[:, self.variables_] = X.loc[:, self.variables_].astype("float")
        X.loc[:, self.variables_] = np.reciprocal(X.loc[:, self.variables_])

        return X

    # for the check_estimator tests
    def _more_tags(self):
        return {
            "_xfail_checks": {
                # =======  this tests fail because the transformers throw an error
                # when the values are 0. Nothing to do with the test itself but
                # mostly with the data created and used in the test
                # TODO: si if and how we can have these replaced
                "check_estimators_dtypes":
                    "transformers raise errors when data contains zeroes, thus this "
                    "check fails",

                "check_estimators_fit_returns_self":
                    "transformers raise errors when data contains zeroes, thus this "
                    "check fails",

                "check_pipeline_consistency":
                    "transformers raise errors when data contains zeroes, thus this "
                    "check fails",

                "check_estimators_overwrite_params":
                    "transformers raise errors when data contains zeroes, thus this "
                    "check fails",

                "check_estimators_pickle":
                    "transformers raise errors when data contains zeroes, thus this "
                    "check fails",

                "check_transformer_general":
                    "transformers raise errors when data contains zeroes, thus this "
                    "check fails",
                # ======================

                # Complex data in math terms, are values like 4i (imaginary numbers
                # so to speak). I've never seen such a thing in the dfs I've
                # worked with, so I do not need this test.
                "check_complex_data": "I dont think we need this check, if users "
                                      "disagree we can think how to introduce it "
                                      "at a later stage.",

                # check that estimators treat dtype object as numeric if possible
                "check_dtype_object":
                    "Transformers use dtypes to select between numerical and "
                    "categorical variables. Feature-engine trusts the user cast the "
                    "variables in they way they would like them treated.",

                # Not sure what the aim of this check is, it fails because FE does not
                # like the sklearn class _NotAnArray
                "check_transformer_data_not_an_array": "Not sure what this check is",

                # this test fails because the test uses dtype attribute of numpy, but
                # in feature engine the array is converted to a df, and it does not
                # have the dtype attribute.
                # need to understand why this test is useful an potentially have one
                # for the package. But some Feature-engine transformers DO change the
                # types
                "check_transformer_preserve_dtypes":
                    "Test not relevant, Feature-engine transformers can change "
                    "the types",

                # TODO: we probably need the test below!!
                "check_methods_sample_order_invariance":
                    "Test does not work on dataframes",

                # TODO: we probably need the test below!!
                # the test below tests that a second fit overrides a first fit.
                # the problem is that the test does not work with pandas df.
                "check_fit_idempotent": "Test does not work on dataframes",

                "check_fit1d": "Test not relevant, Feature-engine transformers only "
                               "work with dataframes",

                "check_fit2d_predict1d":
                    "Test not relevant, Feature-engine transformers only "
                    "work with dataframes",
            }
        }
