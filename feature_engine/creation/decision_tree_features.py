from itertools import combinations
from typing import List, Optional, Tuple, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from feature_engine._docstrings.class_inputs import (
    _drop_original_docstring,
    _variables_numerical_docstring,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_X_matches_training_df,
    check_X,
    check_X_y,
)
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_numerical_variables,
)


@Substitution(
    variables=_variables_numerical_docstring,
    drop_original=_drop_original_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
)
class DecisionTreeFeatures(BaseEstimator, TransformerMixin):
    """
    DecisionTreeFeatures() creates a new variable by using a decision tree.
    The class allows for the use scikit-learn's DecisionTreeClassifier or
    DecisionTreeRegressor.

    Currently, scikit-learn decision-tree classes do not support categorical variables.
    Categorical variables must be converted to numerical values. There are criticisms
    of using OneHotEncoder as sparse matrices can be detrimental to a decision tree's
    performance.

    Parameters
    ----------
    {variables}

    output_features: integer, list or tuple, default=None
        Assign the permutations of variables that will be used to create the new
        feature(s).

        If the user passes an integer, then that number corresponds to the largest
        size of combinations to be used to create the new features:

            If the user passes 3 variables, ["var_A", "var_B", "var_C"], then
                - output_features = 1 returns new features based on the predictions
                    of each individual variable, generating 3 new features.
                - output_features = 2 returns all possible combinations of 2
                    variables, i.e., ("var_A", "var_B"), ("var_A", "var_C"), and
                    ("var_B", "var_C"), in addition to the 3 new variables create
                    by output_features = 1. Resulting in a total of 6 new features.
                - output_features = 3 returns one new feature based on ["var_A",
                    "var_B", "var_C"] in addition to the 6 new features created by
                    output_features = 1 and output_features = 2. Resulting in a total
                    of 7 new features.
                - output_features >= 4 returns an error, a larger size of combination
                    than number of variables provided by user.

        If the user passes a list, the list must be comprised of integers and the
        greatest integer cannot be greater than the number of variables passed by
        the user. Each integer creates all the possible combinations of that size.

            If the user passes 4 variables, ["var_A", "var_B", "var_C", "var_D"]
            and output_features = [2,3] then the following combinations will be
            used to create new features: ("var_A", "var_B"), ("var_A", "var_C"),
            ("var_A", "var_D"), ("var_B", "var_C"), ("var_B", "var_D"), ("var_C",
            "var_D"), ("var_A", "var_B", "var_C"), ("var_A", "var_B", "var_D"),
            ("var_A", "var_C", "var_D"), and ("var_B", "var_C", "var_D").

        If the user passes a tuple, it must be comprised of strings and/or tuples
        that indicate how to combine the variables, e.g. output_features =
        ("var_C", ("var_A", "var_C"), "var_C", ("var_B", "var_D").

        If the user passes None, then all possible combinations will be
        created. This is analogous to the user passing an integer that is equal
        to the number of provided variables when the class is initiated.

    regression: boolean, default = True
        Select whether the decision tree is performing a regression or
        classification.

    max_depth: integer, default = 3
        The maximum depth of the tree. Used to mitigate the risk of over-fitting.

    {drop_original}

    Attributes
    ----------    with pytest.raises(ValueError):
        transformer = DecisionTreeFeatures(
            variables=["Age", "Marks", "Avg_5k_run_minutes", "Height_cm"],
            output_features=3,
            regression=True,
            max_depth=3,
            random_state=_random_state,
            drop_original=True
        )

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Builds a decision tree estimator(s).

    transform:
        Adds new features.

    {fit_transform}

    Notes
    -----


    """
    def __init__(
        self,
        variables: List[Union[str, int]] = None,
        # TODO: What is the correct typing hint?
        output_features: Union[int, List[int], Tuple[tuple, ...]] = None,
        regression: bool = True,
        max_depth: int = 3,
        random_state: int = 0,
        drop_original: bool = False,
    ) -> None:

        if not isinstance(regression, bool):
            raise ValueError(
                f"regression must be a boolean value. Got {regression} instead."
            )

        if not isinstance(max_depth, int) or isinstance(max_depth, bool):
            raise ValueError(
                f"max_depth must be an integer or None. Got {max_depth} instead."
            )

        if not isinstance(random_state, int) or isinstance(random_state, bool):
            raise ValueError(
                f"random_state must be an integer. Got {random_state} instead."
            )

        if not isinstance(drop_original, bool):
            raise ValueError(
                "drop_original takes only boolean values True and False. "
                f"Got {drop_original} instead."
            )
        self.variables = _check_input_parameter_variables(variables)
        self.output_features = output_features
        self.regression = regression
        self.max_depth = max_depth
        self.random_state = random_state
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: pd.Series):
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

        # only numerical variables
        self.variables_ = _find_or_check_numerical_variables(X, self.variables)

        # validate output_features param
        self._validate_strategy()

        # basic checks
        X = check_X(X)
        X, y = check_X_y(X, y)
        _check_contains_inf(X, self.variables_)

        # get all sets of variables that will be used to create new features
        self.variable_combinations_ = self._create_variable_combinations()
        self._estimators = []

        # fit a decision tree for each set of variables
        for combo in self.variable_combinations_:
            estimator = self._make_decision_tree()
            self._estimators.append(estimator.fit(X[combo], y))

        # create a tuple of tuples
        # inner tuple's elements: (1) variable combination; and fitted estimator
        self.output_features_ = tuple(
            [
                (combo, estimator) for combo, estimator in zip(
                    self.variable_combinations_, self._estimators
                )
            ]
        )

        # save input features
        self.feature_names_in_ = X.columns.tolist()

        # save train set shape
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features using a decision tree.

        Parameters
        ----------
        X: Pandas DataFrame of shame = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: Pandas dataframe.
            Either the original dataframe plus the new features or
            a dataframe of only the new features.

        """
        check_is_fitted(self)
        _check_X_matches_training_df(X, self.n_features_in_)

        # get new feature names for dataframe column names
        self.feature_names_ = self.get_feature_names_out(
            input_features=self.variable_combinations_
        )

        # create new features and add them to the original dataframe
        for (var_combo, estimator), name in zip(
                self.output_features_, self.feature_names_
        ):
            X[name] = estimator.predict(X[var_combo])

        if self.drop_original:
            X.drop(columns=self.variables_, inplace=True)

        return X

    def _make_decision_tree(self):
        """Instantiate decision tree."""
        if self.regression is True:
            est = DecisionTreeRegressor(max_depth=self.max_depth,
                                        random_state=self.random_state)
        else:
            est = DecisionTreeClassifier(max_depth=self.max_depth,
                                         random_state=self.random_state)

        return est

    def _create_variable_combinations(self):
        """
        Create a list of the different combinations of variables that are used
        to create new features.
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
                for num in range(1, len(self.variables_) + 1):
                    variable_combinations += list(combinations(self.variables_, num))

            elif isinstance(self.output_features, int):
                for num in range(1, self.output_features + 1):
                    variable_combinations += list(combinations(self.variables_, num))

            # output_feature is a list
            else:
                for num in self.output_features:
                    variable_combinations += list(combinations(self.variables_, num))

            # transform all elements to lists to slice X dataframe
            variable_combinations = [list(var) for var in variable_combinations]

        return variable_combinations

    def get_feature_names_out(self, input_features: Optional[List] = None) -> List:
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features: list, default=None
            If input_features is None, then the names of all the
            variables in the transformed dataset (original + new variables) is returned.
            Alternatively, only the names for the new features derived from
            input_features will be returned.

        Returns
        -------
        feature_names_out: list
            The feature names.
        """
        feature_names = []

        for combo in self.variable_combinations_:
            if len(combo) == 1:
                feature_names.append(f"{combo[0]}_tree")

            else:
                combo_joined = "_".join(combo)
                feature_names.append(f"{combo_joined}_tree")

        if input_features is None:
            feature_names = self.variables + feature_names

        return feature_names

    def _get_distinct_output_features(self) -> List:
        """
        Get unique values from output_features when it is a tuple.

        """
        unique_features = []

        # transform all elements in output_features to a list
        features_list = [
            list(features) if type(features) is tuple else [features]
            for features in self.output_features
        ]

        # merge all lists into 1 list
        for lst in features_list:
            unique_features += lst

        # get unique values
        unique_features = list(set(unique_features))

        return unique_features

    def _validate_strategy(self) -> None:
        """
        Validates output_features.
        """
        if (
                not isinstance(self.output_features, (int, list, tuple))
                and self.output_features is not None
        ):
            raise ValueError(
                f"output_features must an integer, list or tuple. Got "
                f"{self.output_features} instead."
            )

        # check user is not creating combinations comprised of more variables
        # than the number of variables provided in the 'variables' param
        if isinstance(self.output_features, int):
            if self.output_features > len(self.variables_):
                raise ValueError(
                    "If output_features is an integer, the value cannot be "
                    f"greater than the number of variables. Got {self.output_features} "
                    f"for output_features and {len(self.variables_)} for the number "
                    "of variables."
                )

        if isinstance(self.output_features, list):
            # Check (1) the list is only comprised of integers or
            # (2) user is not creating combinations comprised of more variables
            # than the number of variables allowed by the 'variables' param
            if (
                     not all(
                        isinstance(feature, int) for feature in self.output_features
                    )
                    or max(self.output_features) > len(self.variables_)
            ):
                raise ValueError(
                    "output_features must be a list comprised of integers. The "
                    "maximum integer cannot be greater than the number of variables "
                    f"passed in the 'variable' param. Got {self.output_features} for "
                    f"output_features and {len(self.variables_)} for the number of "
                    f"variables. "
                )

        if isinstance(self.output_features, tuple):

            # confirm all features in output_features are included in variables
            unique_output_features = self._get_distinct_output_features()
            if not all(
                    feature in self.variables for feature in unique_output_features
            ):
                raise ValueError(
                    "output_features contains features that are not in variables. "
                    f"Got {unique_output_features} for unique output features. "
                    f"Got {self.variables_} for variables."
                )

            # calculate maximum number of subsequences/combinations of variables
            num_combos = 0
            for n in range(1, len(self.variables_) + 1):
                # combinations returns all subsequences of 'n' length from 'variables'
                num_combos += len(list(combinations(self.variables_, n)))

            # check (1) each element in output_features is either a string or tuple or
            # (2) user only passes strings and tuples.
            if (
                    not all(
                        isinstance(feature, (str, tuple)) for feature in
                        self.output_features
                    )
                    or len(self.output_features) > num_combos
            ):
                raise ValueError(
                    "output_features must be comprised of tuples and strings."
                    f"output_features cannot contain more feature combinations than "
                    f"{num_combos}. Got {self.output_features} instead."
                )

        return self
