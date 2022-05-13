from itertools import combinations
from typing import List, Optional, Tuple, Union

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
        variables: List[Union[str, int]] = None,
        output_features: Union[int, List[int], Tuple[tuple, ...]] = None,
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

        # checks for output_features
        if isinstance(output_features, int):
            if len(variables) > output_features:
                raise ValueError(
                    "If output_features is an integer, the value cannot be greater than "
                    f"the length of variables. Got {output_features} for output_features "
                    f"and {len(variables)} for the length of variables."
                )

        if isinstance(output_features, list):
            if (
                max(output_features) > len(variables)
                or not all(isinstance(feature, int) for feature in output_features)
            ):
                raise ValueError(
                    "output_features must be a list solely comprised of integers and the "
                    "maximum integer cannot be greater than the length of variables. Got "
                    f"{output_features} for output_features and {len(variables)} for the "
                    f"length of variables."
                )

        if isinstance(output_features, tuple):
            num_combos = 0
            for n in range(len(variables)):
                num_combos += len(list(combinations(variables, n + 1)))
            if (
                not all(isinstance(feature, tuple) for feature in output_features)
                or len(output_features) > num_combos
            ):
                raise ValueError(
                    "output_features must a tuple solely comprised of tuples and the maximum "
                    f"number of tuples cannot exceed {num_combos}. Got {output_features} instead."
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
        self.output_features = output_features
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