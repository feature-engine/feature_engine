from typing import List, Optional, Union

import pandas as pd
from sklearn.utils.validation import check_is_fitted
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from feature_engine.creation.base_creation import BaseCreation
from feature_engine._docstrings.methods import (
    _fit_not_learn_docstring,
    _fit_transform_docstring,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.class_inputs import (
    _variables_numerical_docstring,
    _drop_original_docstring,
    _missing_values_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.variable_manipulation import _find_or_check_numerical_variables


class DecisionTreeCreation(BaseCreation):
    """
    DecisionTreeCreation() creates a new variable by applying user-indicated variables with
    a decision tree. The class uses either scikit-learn's DecisionTreeClassifier or
    DecisionTreeRegressor, pending the target value.

    Currently, scikit-learn decision-tree classes do not support categorical variables.
    Categorical variables must be converted to numerical values. There are criticisms of
    using OneHotEncoder as sparse matrices can be detrimental to a decision tree's performance.






    """
    def __init__(
        self,
        variables: List[Union[str, int]],
        new_variable_name: Optional[str] = None,
        regression: bool = True,
        max_depth: int = 3,
        missing_value: str = "raise",
        drop_original: bool = False,
    ) -> None:

        if (
            not isinstance(variables, list)
            or not all(isinstance(var, (int, str)) for var in variables)
            or len(variables) < 2
            or len(set(variables)) != len(variables)
        ):
            raise ValueError(
                "variables must a list of string or integers with a least 2 "
                f"distinct variables. Got {variables} instead."
            )

        # TODO: Does the transfomer generate 1 or > 1 new variables?
        if new_variable_name is not None:
            if not isinstance(new_variable_name, str):
                raise ValueError(
                    f"new_variable_name must a be a string. Got {new_variable_name} "
                    f"instead."
                )

        if not isinstance(regression, bool):
            raise ValueError(
                f"regression must be a boolean value. Got {regression} instead."
            )

        if not isinstance(max_depth, int):
            raise ValueError(
                f"max_depth must be an integer. Got {max_depth} instead."
            )

        super().__init__(missing_value, drop_original)
        self.variables = variables
        self.new_variable_name = new_variable_name
        self.regression = regression
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        The transformer learns the target variable values associated with
        the user-provided features using a decision tree.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just
            the variables to transform.

        y: pandas Series or np.array = [n_samples,]
            The target variable that is used to train the decision tree.
        """
        # common checks and attributes
        X = super().fit(X, y)