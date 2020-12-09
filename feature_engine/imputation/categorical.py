# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import Optional, List, Union

import pandas as pd

from feature_engine.dataframe_checks import _is_dataframe
from feature_engine.imputation.base_imputer import BaseImputer
from feature_engine.variable_manipulation import (
    _find_or_check_categorical_variables,
    _check_input_parameter_variables,
)


class CategoricalImputer(BaseImputer):
    """
    The CategoricalImputer() replaces missing data in categorical variables
    by a string like 'Missing' or any other entered by the user. Alternatively, it
    replaces missing data by the most frequent category.

    The CategoricalVariableImputer() works only with categorical variables.

    The user can pass a list with the variables to be imputed. Alternatively,
    the CategoricalImputer() will automatically find and select all variables of type
    object.

    **Note**

    If you want to impute numerical variables with this transformer, you first need to
    cast them as object. It may well be that after the imputation, they are re-casted
    by pandas as numeric. Thus, if planning to do categorical encoding with
    feature-engine to this variables after the imputation, make sure to return the
    variables as object by setting `return_object=True`.


    Parameters
    ----------
    imputation_method : str, default=missing
        Desired method of imputation. Can be 'frequent' or 'missing'.

    fill_value : str, default='Missing'
        Only used when `imputation_method='missing'`. Can be used to set a
        user-defined value to replace the missing data.

    variables : list, default=None
        The list of variables to be imputed. If None, the imputer will find and
        select all object type variables.

    return_object: bool, default=False
        If working with numerical variables cast as object, decide
        whether to return the variables as numeric or re-cast them as object.
        Note that pandas will re-cast them automatically as numeric after the
        transformation with the mode.

    Attributes
    ----------
    imputer_dict_:
        Dictionary with most frequent category or string per variable.

    Methods
    -------
    fit:
        Learn more frequent category, or assign string to variable.
    transform:
        Impute missing data.
    fit_transform:
        Fit to the data, than transform it.
    """

    def __init__(
        self,
        imputation_method: str = "missing",
        fill_value: str = "Missing",
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_object: bool = False,
    ) -> None:

        if imputation_method not in ["missing", "frequent"]:
            raise ValueError(
                "imputation_method takes only values 'missing' or 'frequent'"
            )

        if not isinstance(fill_value, str):
            raise ValueError("parameter 'fill_value' should be string")

        self.imputation_method = imputation_method
        self.fill_value = fill_value
        self.variables = _check_input_parameter_variables(variables)
        self.return_object = return_object

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the most frequent category if the imputation method is set to frequent.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y : pandas Series, default=None
            y is not needed in this imputation. You can pass None or y.

        Raises
        ------
        TypeError
            - If the input is not a Pandas DataFrame.
            - If any user provided variable is not categorical
        ValueError
            If there are no categorical variables in the df or the df is empty

        Returns
        -------
        self
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find or check for categorical variables
        self.variables = _find_or_check_categorical_variables(X, self.variables)

        if self.imputation_method == "missing":
            self.imputer_dict_ = {var: self.fill_value for var in self.variables}

        elif self.imputation_method == "frequent":
            self.imputer_dict_ = {}

            for var in self.variables:
                mode_vals = X[var].mode()

                # careful: some variables contain multiple modes
                if len(mode_vals) == 1:
                    self.imputer_dict_[var] = mode_vals[0]
                else:
                    raise ValueError(
                        "Variable {} contains multiple frequent categories.".format(var)
                    )

        self.input_shape_ = X.shape

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # bring functionality from the BaseImputer
        X = super().transform(X)

        # add additional step to return variables cast as object
        if self.return_object:
            X[self.variables] = X[self.variables].astype("O")

        return X

    # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    transform.__doc__ = BaseImputer.transform.__doc__
