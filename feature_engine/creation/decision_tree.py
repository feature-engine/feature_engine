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


    """
    def __init__(
        self,
        variables: List[Union[str, int]],
        new_variable_name: Optional[str] = None,
        strategy: str = "regressor",
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

        if strategy not in ("classifier", "regressor"):
            raise ValueError(
                "strategy must be either 'classifier' or 'regressor'. "
                f"Got {strategy} instead."
            )

        super().__init__(missing_value, drop_original)
        self.variables = variables
        self.new_variable_name = new_variable_name
        self.strategy = strategy
    