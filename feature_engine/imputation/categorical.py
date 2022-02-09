# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd

from feature_engine.dataframe_checks import _is_dataframe
from feature_engine.docstrings import (
    Substitution,
    _feature_names_in_docstring,
    _fit_transform_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine.imputation.base_imputer import BaseImputer
from feature_engine.validation import _return_tags
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_all_variables,
    _find_or_check_categorical_variables,
)


@Substitution(
    imputer_dict_=BaseImputer._imputer_dict_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    transform=BaseImputer._transform_docstring,
    fit_transform=_fit_transform_docstring,
)
class CategoricalImputer(BaseImputer):
    """
    The CategoricalImputer() replaces missing data in categorical variables by an
    arbitrary value or by the most frequent category.

    The CategoricalImputer() imputes by default only categorical variables
    (type 'object' or 'categorical'). You can pass a list of variables to impute, or
    alternatively, the encoder will find and impute all categorical variables.

    If you want to impute numerical variables with this transformer, there are 2 ways
    of doing it:

    **Option 1**: Cast your numerical variables as object in the input dataframe before
    passing it to the transformer.

    **Option 2**: Set `ignore_format=True`. Note that if you do this and do not pass the
    list of variables to impute, the imputer will automatically select and impute all
    variables in the dataframe.

    More details in the :ref:`User Guide <categorical_imputer>`.

    Parameters
    ----------
    imputation_method: str, default='missing'
        Desired method of imputation. Can be 'frequent' for frequent category imputation
        or 'missing' to impute with an arbitrary value.

    fill_value: str, int, float, default='Missing'
        User-defined value to replace missing data. Only used when
        `imputation_method='missing'`.

    variables: list, default=None
        The list of categorical variables that will be imputed. If None, the
        imputer will find and transform all variables of type object or categorical by
        default. You can also make the transformer accept numerical variables, see the
        parameter `ignore_format` below.

    return_object: bool, default=False
        If working with numerical variables cast as object, decide
        whether to return the variables as numeric or re-cast them as object.
        Note that pandas will re-cast them automatically as numeric after the
        transformation with the mode or with an arbitrary number.

    ignore_format: bool, default=False
        Whether the format in which the categorical variables are cast should be
        ignored. If false, the imputer will automatically select variables of type
        object or categorical, or check that the variables entered by the user are of
        type object or categorical. If True, the imputer will select all variables or
        accept all variables entered by the user, including those cast as numeric.

    Attributes
    ----------
    {imputer_dict_}

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Learn the most frequent category or assign arbitrary value to variable.

    {transform}

    {fit_transform}

    """

    def __init__(
        self,
        imputation_method: str = "missing",
        fill_value: Union[str, int, float] = "Missing",
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_object: bool = False,
        ignore_format: bool = False,
    ) -> None:

        if imputation_method not in ["missing", "frequent"]:
            raise ValueError(
                "imputation_method takes only values 'missing' or 'frequent'"
            )

        if not isinstance(ignore_format, bool):
            raise ValueError("ignore_format takes only booleans True and False")

        self.imputation_method = imputation_method
        self.fill_value = fill_value
        self.variables = _check_input_parameter_variables(variables)
        self.return_object = return_object
        self.ignore_format = ignore_format

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the most frequent category if the imputation method is set to frequent.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y: pandas Series, default=None
            y is not needed in this imputation. You can pass None or y.
        """

        # check input dataframe
        X = _is_dataframe(X)

        # check or select the the right variables
        if not self.ignore_format:
            # find categorical variables or check variables entered by user are
            # categorical
            self.variables_: List[
                Union[str, int]
            ] = _find_or_check_categorical_variables(X, self.variables)
        else:
            # select all variables or check variables entered by the user
            self.variables_ = _find_all_variables(X, self.variables)

        if self.imputation_method == "missing":
            self.imputer_dict_ = {var: self.fill_value for var in self.variables_}

        elif self.imputation_method == "frequent":
            self.imputer_dict_ = {}

            for var in self.variables_:
                mode_vals = X[var].mode()

                # careful: some variables contain multiple modes
                if len(mode_vals) == 1:
                    self.imputer_dict_[var] = mode_vals[0]
                else:
                    raise ValueError(
                        "Variable {} contains multiple frequent categories.".format(var)
                    )

        self.feature_names_in_ = X.columns.to_list()
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        X = self._check_transform_input_and_state(X)

        # replaces missing data with the learned parameters
        if self.imputation_method == "frequent":
            for variable in self.imputer_dict_:
                X[variable].fillna(self.imputer_dict_[variable], inplace=True)

        else:
            for variable in self.imputer_dict_:
                if pd.api.types.is_categorical_dtype(X[variable]):
                    # if variable is of type category, we first need to add the new
                    # category, and then fill in the nan
                    X[variable].cat.add_categories(
                        self.imputer_dict_[variable], inplace=True
                    )

                X[variable].fillna(self.imputer_dict_[variable], inplace=True)

        # add additional step to return variables cast as object
        if self.return_object:
            X[self.variables_] = X[self.variables_].astype("O")

        return X

    # Get docstring from BaseClass
    transform.__doc__ = BaseImputer.transform.__doc__

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["allow_nan"] = True
        tags_dict["variables"] = "categorical"
        return tags_dict
