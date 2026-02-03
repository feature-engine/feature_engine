from typing import List, Union

import pandas as pd

from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.init_parameters.selection import (
    _confirm_variables_docstring,
)
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.selection._docstring import (
    _get_support_docstring,
    _variables_all_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import _check_contains_na, check_X
from feature_engine.selection.base_selector import BaseSelector
from feature_engine.tags import _return_tags

from .base_selection_functions import _select_all_variables

Variables = Union[None, int, str, List[Union[str, int]]]


@Substitution(
    confirm_variables=_confirm_variables_docstring,
    variables=_variables_all_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
    get_support=_get_support_docstring,
)
class DropConstantFeatures(BaseSelector):
    """
    DropConstantFeatures() drops constant and quasi-constant variables from a dataframe.
    Constant variables show the same value in all the observations in the dataset.
    Quasi-constant variables show the same value in almost all the observations in the
    dataset.

    This transformer works with numerical and categorical variables. The user can
    indicate a list of variables to examine. Alternatively, the transformer will
    evaluate all the variables in the dataset.

    The transformer will first identify and store the constant and quasi-constant
    variables. Next, the transformer will drop these variables from a dataframe.

    More details in the :ref:`User Guide <drop_constant>`.

    Parameters
    ----------
    {variables}

    tol: float,int,  default=1
        Threshold to detect constant/quasi-constant features. Variables showing the
        same value in a percentage of observations greater than tol will be considered
        constant / quasi-constant and dropped. If tol=1, the transformer removes
        constant variables. Else, it will remove quasi-constant variables. For example,
        if tol=0.98, the transformer will remove variables that show the same value in
        98% of the observations.

    missing_values: str, default=raises
        Whether the missing values should be raised as error, ignored or included as an
        additional value of the variable. Takes values 'raise', 'ignore', 'include'.

    {confirm_variables}

    Attributes
    ----------
    features_to_drop_:
        List with constant and quasi-constant features.

    {variables_}:

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Find constant and quasi-constant features.

    {fit_transform}

    {get_support}

    transform:
        Remove constant and quasi-constant features.

    Notes
    -----
    This transformer is a similar concept to the VarianceThreshold from Scikit-learn,
    but it evaluates number of unique values instead of variance.

    See Also
    --------
    sklearn.feature_selection.VarianceThreshold

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.selection import DropConstantFeatures
    >>> X = pd.DataFrame(dict(x1 = [1,1,1,1],
    >>>                     x2 = ["a", "a", "b", "c"],
    >>>                     x3 = [True, False, False, True]))
    >>> dcf = DropConstantFeatures()
    >>> dcf.fit_transform(X)
        x2     x3
    0  a   True
    1  a  False
    2  b  False
    3  c   True

    Additionally, you can set the Threshold for quasi-constant features:

    >>> X = pd.DataFrame(dict(x1 = [1,1,1,1],
    >>>                      x2 = ["a", "a", "b", "c"],
    >>>                      x3 = [True, False, False, False]))
    >>> dcf = DropConstantFeatures(tol = 0.75)
    >>> dcf.fit_transform(X)
        x2
    0  a
    1  a
    2  b
    3  c
    """

    def __init__(
        self,
        variables: Variables = None,
        tol: float = 1,
        missing_values: str = "raise",
        confirm_variables: bool = False,
    ):

        if (
            not isinstance(tol, (float, int))
            or isinstance(tol, bool)
            or tol < 0
            or tol > 1
        ):
            raise ValueError("tol must be a float or integer between 0 and 1")

        if missing_values not in ["raise", "ignore", "include"]:
            raise ValueError(
                "missing_values takes only values 'raise', 'ignore' or " "'include'."
            )

        super().__init__(confirm_variables)

        self.tol = tol
        self.variables = _check_variables_input_value(variables)
        self.missing_values = missing_values

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Find constant and quasi-constant features.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The input dataframe.
        y: None
            y is not needed for this transformer. You can pass y or None.
        """

        # check input dataframe
        X = check_X(X)

        self.variables_ = _select_all_variables(
            X, self.variables, self.confirm_variables
        )

        if self.missing_values == "raise":
            # check if dataset contains na
            _check_contains_na(X, self.variables_)

        if self.missing_values == "include":
            X[self.variables_] = X[self.variables_].fillna("missing_values")

        # find constant features
        if self.tol == 1:
            self.features_to_drop_ = [
                feature for feature in self.variables_ if X[feature].nunique() == 1
            ]

        # find constant and quasi-constant features
        else:
            self.features_to_drop_ = []

            for feature in self.variables_:
                # find most frequent value / category in the variable
                predominant = (
                    (X[feature].value_counts() / float(len(X)))
                    .sort_values(ascending=False)
                    .values[0]
                )

                if predominant >= self.tol:
                    self.features_to_drop_.append(feature)

        # check we are not dropping all the columns in the df
        if len(self.features_to_drop_) == len(X.columns):
            raise ValueError(
                "The resulting dataframe will have no columns after dropping all "
                "constant or quasi-constant features. Try changing the tol value."
            )

        # save input features
        self._get_feature_names_in(X)

        return self

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["allow_nan"] = True
        tags_dict["variables"] = "all"
        # add additional test that fails
        tags_dict["_xfail_checks"][
            "check_fit2d_1sample"
        ] = "the transformer raises an error when dropping all columns, ok to fail"
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
