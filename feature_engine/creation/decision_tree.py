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
        regression: bool = True,
        max_depth: int = 3,
        missing_value: str = "raise",
        drop_original: bool = False,
    ) -> None:

        if (
            not isinstance(variables, list)
            or not all(isinstance(var, (int, str)) for var in variables)
            or len(set(variables)) != len(variables)
        ):
            raise ValueError(
                "variables must a list of strings or integers comprise of "
                f"distinct variables. Got {variables} instead."
            )

        if (
                not isinstance(output_features, (int, list, tuple))
                and output_features is not None
        ):
            raise ValueError(
                f"output_features must an integer, list or tuple. Got {output_features} instead."
            )

        if isinstance(output_features, int):
            if output_features > len(variables):
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
            for n in range(1, len(variables) + 1):
                num_combos += len(list(combinations(variables, n)))
            if (
                    not all(isinstance(feature, (str, tuple)) for feature in output_features)
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
        # TODO: We don't need to check_variables b/c BaseCreation fit() performs action
        X = super().fit(X, y)
        self._check_dependent_variable_not_fitted_by_estimator(X, y)

        self.variable_combinations_ = self._create_variable_combinations()
        self.variable_combination_indices_ = {}
        self.fitted_estimators_ = {}

        for idx, combo in enumerate(self.variable_combinations_):
            self.variable_combination_indices_[f"estimator_{idx}"] = combo
            estimator = self._make_decision_tree()
            self.fitted_estimators_[f"estimator_{idx}"] = estimator.fit(X[combo], y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Creates new features using scikit-learn's decision tree.

        Parameters
        ----------
        X: Pandas DataFrame of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: Pandas dataframe.
            The original dataframe plus the additional features.
        """
        X = super().transform(X)
        self.new_features_names_ = self._create_new_features_names()

        for idx, estimator in self.fitted_estimators_.items():
            new_feature = self.new_features_names_[idx]
            X[new_feature] = estimator.predict(
                X[self.variable_combination_indices_[idx]]
            )

        return X

    def _make_decision_tree(self):
        """Instantiate decision tree."""
        if self.regression is True:
            est = DecisionTreeRegressor(max_depth=self.max_depth)
        else:
            est = DecisionTreeClassifier(max_depth=self.max_depth)

        return est

    def _create_variable_combinations(self):
        """
        Create a list of the different combinations of variables that will be
        used to create new features.
        """
        variable_combinations = []
        if isinstance(self.output_features, tuple):
            for feature in self.output_features:
                if isinstance(feature, str):
                    variable_combinations.append([feature])
                else:
                    variable_combinations.append(list(feature))

        # if output_features is None, int or list.
        else:
            if self.output_features is None:
                for num in range(1, len(self.variables) + 1):
                    variable_combinations += list(combinations(self.variables, num))

            elif isinstance(self.output_features, int):
                for num in range(1, self.output_features + 1):
                    variable_combinations += list(combinations(self.variables, num))

            # output_feature is a list
            else:
                for num in self.output_features:
                    variable_combinations += list(combinations(self.variables, num))

            # transform all elements to lists to slice X dataframe
            variable_combinations = [list(var) for var in variable_combinations]

        return variable_combinations

    def _create_new_features_names(self):
        """Generate a dictionary of the names for the new features"""
        new_features_names = {}

        for idx, combo in self.variable_combination_indices_.items():
            if len(combo) == 1:
                new_features_names[idx] = f"{combo[0]}_tree"

            else:
                combo_joined = "_".join(combo)
                new_features_names[idx] = f"{combo_joined}_tree"

        return new_features_names

    def _check_dependent_variable_not_fitted_by_estimator(
            self, X: pd.DataFrame, y: pd.Series
    ) -> None:
        """
        Raise error if one of the variables to be fitted by the decision tree
        is the dependent variable.
        """
        # TODO: Is this neccessary?
        # Wouldn't doing so be a circular reference?
        for variable in self.variables_:
            if X[variable] == y:
                raise ValueError(
                    "Dependent variable cannot also be one of the estimator's "
                    "fitted variables."
                )