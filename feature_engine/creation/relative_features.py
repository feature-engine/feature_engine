from typing import List, Optional, Union

import pandas as pd
from sklearn.utils.validation import check_is_fitted

from feature_engine.creation.base_creation import BaseCreation
from feature_engine.docstrings import (
    Substitution,
    _drop_original_docstring,
    _feature_names_in_docstring,
    _fit_not_learn_docstring,
    _fit_transform_docstring,
    _missing_values_docstring,
    _n_features_in_docstring,
    _variables_numerical_docstring,
)
from feature_engine.validation import _return_tags
from feature_engine.variable_manipulation import _find_or_check_numerical_variables

_PERMITTED_FUNCTIONS = [
    "add",
    "sub",
    "mul",
    "div",
    "truediv",
    "floordiv",
    "mod",
    "pow",
]


@Substitution(
    variables=_variables_numerical_docstring,
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
    RelativeFeatures() applies basic mathematical operations between a group
    of variables and one or more reference features. It adds the resulting features
    to the dataframe.

    In other words, CombineWithReferenceFeature() sums, multiplies, subtracts or
    divides a group of features to / by a group of reference variables, and returns the
    result as new variables in the dataframe.

    The transformed dataframe will contain the additional features indicated in the
    new_variables_name list plus the original set of variables.

    More details in the :ref:`User Guide <relative_features>`.

    Parameters
    ----------
    variables: list
        The list of numerical variables to combine with the reference variables.

    reference: list
        The list of numerical reference variables that will be added to, multiplied
        with, or subtracted from the `variables_to_combine`, or used as denominator for
        division.

    func: list, default=['sub']
        The list of basic mathematical operations to be used in the transformation.

        If None, all of ['sub', 'div', 'add', 'mul'] will be performed. Alternatively,
        you can enter a list of operations to carry out. Each operation should
        be a string and must be one of the elements in `['sub', 'div', 'add', 'mul']`.

        Each operation will result in a new variable that will be added to the
        transformed dataset.

    new_variables_names: list, default=None
        Names of the new variables. If passing a list with the names for the new
        features (recommended), you must enter as many names as new features created
        by the transformer. The number of new features is the number of `operations`,
        times the number of `reference_variables`, times the number of
        `variables_to_combine`.

        If `new_variable_names` is None, the transformer will assign an arbitrary name
        to the features. The name will be var + operation + ref_var.

    {missing_values}

    {drop_original}

    Attributes
    ----------
    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {transform}

    {fit_transform}

    Notes
    -----
    Although the transformer in essence allows us to combine any feature with any of
    the allowed mathematical operations, its use is intended mostly for the creation
    of new features based on some domain knowledge. Typical examples within the
    financial sector are:

    - Ratio between income and debt to create the debt_to_income_ratio.
    - Subtraction of rent from income to obtain the disposable_income.
    """

    def __init__(
        self,
        variables: List[Union[str, int]],
        reference: List[Union[str, int]],
        func: List[str],
        new_variables_names: Optional[List[str]] = None,
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

        if not isinstance(func, list) or not any(
            fun not in _PERMITTED_FUNCTIONS for fun in func
        ):
            raise ValueError(
                "At least one of the entered functions is not supported. "
                "Supported functions are {}. ".format(", ".join(_PERMITTED_FUNCTIONS))
            )

        if new_variables_names is not None:
            if (
                not isinstance(new_variables_names, list)
                or not all(isinstance(var, str) for var in new_variables_names)
                or len(set(new_variables_names)) != len(new_variables_names)
            ):
                raise ValueError(
                    "new_variable_names should be None or a list of unique strings. "
                    f"Got {new_variables_names} instead."
                )

        if new_variables_names is not None:
            if len(new_variables_names) != len(reference) * len(variables) * len(func):
                raise ValueError(
                    "The number of new feature names must coincide with the number "
                    "of returned new features."
                )

        super().__init__(missing_values, drop_original)
        self.variables = variables
        self.reference = reference
        self.func = func
        self.new_variables_names = new_variables_names

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
        super().fit(X, y)

        # check variables are numerical
        self.reference = _find_or_check_numerical_variables(X, self.reference)

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
        X_new: Pandas dataframe, shape = [n_samples, n_features + n_operations *
        n_references]
            The dataframe with the new variables.
        """

        X = super().transform(X)

        # TODO: add transform functionality. Waiting for stackoverflow consultation.

        # # replace created variable names with user ones.
        # if self.new_variables_names:
        #     X.columns = self.feature_names_in_ + self.new_variables_names
        #
        # if self.drop_original:
        #     X.drop(
        #         columns=set(self.variables_to_combine + self.reference_variables),
        #         inplace=True,
        #     )

        return X

    def _sub(self, X):
        for reference in self.reference:
            varname = [str(var) + "_sub_" + str(reference) for var in self.variables]
            X[varname] = X[self.variables].sub(X[reference], axis=0)
        return X

    def _div(self, X):
        for reference in self.reference:
            if (X[reference] == 0).any().any():
                raise ValueError(
                    "Some of the reference variables contain 0 as values. Check and "
                    "remove those before using this transformer."
                )
            varname = [
                str(var) + "_div_" + str(reference) for var in self.variables_to_combine
            ]
            X[varname] = X[self.variables_to_combine].div(X[reference], axis=0)
        return X

    def _add(self, X):
        for reference in self.reference_variables:
            varname = [
                str(var) + "_add_" + str(reference) for var in self.variables_to_combine
            ]
            X[varname] = X[self.variables_to_combine].add(X[reference], axis=0)
        return X

    def _mul(self, X):
        for reference in self.reference_variables:
            varname = [
                str(var) + "_mul_" + str(reference) for var in self.variables_to_combine
            ]
            X[varname] = X[self.variables_to_combine].mul(X[reference], axis=0)
        return X

    def _truediv(self, X):

        for reference in self.reference:
            if (X[reference] == 0).any().any():
                raise ValueError(
                    "Some of the reference variables contain 0 as values. Check and "
                    "remove those before using this transformer."
                )
            varname = [
                str(var) + "_truediv_" + str(reference)
                for var in self.variables_to_combine
            ]
            X[varname] = X[self.variables_to_combine].truediv(X[reference], axis=0)
        return X

    def _floordiv(self, X):
        for reference in self.reference:
            if (X[reference] == 0).any().any():
                raise ValueError(
                    "Some of the reference variables contain 0 as values. Check and "
                    "remove those before using this transformer."
                )
            varname = [
                str(var) + "_truediv_" + str(reference)
                for var in self.variables_to_combine
            ]
            X[varname] = X[self.variables_to_combine].floordiv(X[reference], axis=0)
        return X

    def _mod(self, X):
        for reference in self.reference:
            if (X[reference] == 0).any().any():
                raise ValueError(
                    "Some of the reference variables contain 0 as values. Check and "
                    "remove those before using this transformer."
                )
            varname = [
                str(var) + "_truediv_" + str(reference)
                for var in self.variables_to_combine
            ]
            X[varname] = X[self.variables_to_combine].mod(X[reference], axis=0)
        return X

    def _pow(self, X):
        for reference in self.reference:
            varname = [
                str(var) + "_truediv_" + str(reference)
                for var in self.variables_to_combine
            ]
            X[varname] = X[self.variables_to_combine].por(X[reference], axis=0)
        return X

    def get_feature_names_out(self, input_features: Optional[List] = None) -> List:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features: list, default=None
            Input features. If `input_features` is `None`, then the names of all the
            variables in the transformed dataset (original + new variables) is returned.
            Alternatively, only the names for the new features derived from
            input_features will be returned.

        Returns
        -------
        feature_names_out: list
            The feature names.
        """
        check_is_fitted(self)

        # # Create names for all features or just the indicated ones.
        # if input_features is None:
        #     input_features_ = self.variables_
        # else:
        #     if not isinstance(input_features, list):
        #         raise ValueError(
        #             f"input_features must be a list. Got {input_features} instead."
        #         )
        #     if any([f for f in input_features if f not in self.variables_]):
        #         raise ValueError(
        #             "Some features in input_features were not used to create new "
        #             "variables. You can only get the names of the new features "
        #             "with this function."
        #         )
        #     # Create just indicated lag features.
        #     input_features_ = input_features
        #
        # # create the names for the periodic features
        # feature_names = [
        #     str(var) + suffix for var in input_features_ for suffix in
        #     ["_sin", "_cos"]
        # ]
        #
        # # Return names of all variables if input_features is None.
        # if input_features is None:
        #     if self.drop_original is True:
        #         # Remove names of variables to drop.
        #         original = [
        #             f for f in self.feature_names_in_ if f not in self.variables_
        #         ]
        #         feature_names = original + feature_names
        #     else:
        #         feature_names = self.feature_names_in_ + feature_names
        #
        # return feature_names
        return self

    def _more_tags(self):
        tags_dict = _return_tags()
        # add additional test that fails
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"

        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has 1 mandatory parameter"

        msg = "this transformer works with datasets that contain at least 2 variables. \
        Otherwise, there is nothing to combine"
        tags_dict["_xfail_checks"]["check_fit2d_1feature"] = msg
        return tags_dict
