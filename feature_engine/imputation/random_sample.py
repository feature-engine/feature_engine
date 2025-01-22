# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_transform_docstring,
    _transform_imputers_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import check_X
from feature_engine.imputation.base_imputer import BaseImputer
from feature_engine.tags import _return_tags
from feature_engine.variable_handling import check_all_variables, find_all_variables


# for RandomSampleImputer
def _define_seed(
    X: pd.DataFrame,
    index: int,
    seed_variables: Union[str, int, List[Union[str, int]]],
    how: str = "add",
) -> int:
    # determine seed by adding or multiplying the value of 1 or
    # more variables
    if how == "add":
        internal_seed = int(np.round(X.loc[index, seed_variables].sum(), 0))
    elif how == "multiply":
        internal_seed = int(np.round(X.loc[index, seed_variables].product(), 0))
    return internal_seed


@Substitution(
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    transform=_transform_imputers_docstring,
    fit_transform=_fit_transform_docstring,
)
class RandomSampleImputer(BaseImputer):
    """
    The RandomSampleImputer() replaces missing data with a random sample extracted from
    the variables in the training set.

    The RandomSampleImputer() works with both numerical and categorical variables.

    **Note**

    The Random samples used to replace missing values may vary from execution to
    execution. This may affect the results of your work. Thus, it is advisable to set a
    seed.

    More details in the :ref:`User Guide <random_sample_imputer>`.

    Parameters
    ----------
    variables: list, default=None
        The list of variables to be imputed. If None, the imputer will select
        all variables in the train set.

    random_state: int, str or list, default=None
        The random_state can take an integer to set the seed when extracting the
        random samples. Alternatively, it can take a variable name or a list of
        variables, which values will be used to determine the seed, observation per
        observation.

    seed: str, default='general'
        Indicates whether the seed should be set for each observation with missing
        values, or if one seed should be used to impute all observations in one go.

        **'general'**: one seed will be used to impute the entire dataframe. This is
        equivalent to setting the seed in pandas.sample(random_state).

        **'observation'**: the seed will be set for each observation using the values
        of the variables indicated in the random_state for that particular
        observation.

    seeding_method: str, default='add'
        If more than one variable are indicated to seed the random sampling per
        observation, you can choose to combine those values as an addition or a
        multiplication. Can take the values 'add' or 'multiply'.

    Attributes
    ----------
    X_:
        Copy of the training dataframe from which to extract the random samples.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Make a copy of the train set

    {fit_transform}

    {transform}

    Examples
    --------

    >>> import pandas as pd
    >>> import numpy as np
    >>> from feature_engine.imputation import RandomSampleImputer
    >>> X = pd.DataFrame(dict(
    >>>        x1 = [np.nan,1,1,0,np.nan],
    >>>        x2 = ["a", np.nan, "b", np.nan, "a"],
    >>>        ))
    >>> rsi = RandomSampleImputer()
    >>> rsi.fit(X)
    >>> rsi.transform(X)
        x1 x2
    0  1.0  a
    1  1.0  b
    2  1.0  b
    3  0.0  a
    4  1.0  a
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        random_state: Union[None, int, str, List[Union[str, int]]] = None,
        seed: str = "general",
        seeding_method: str = "add",
    ) -> None:

        if seed not in ["general", "observation"]:
            raise ValueError("seed takes only values 'general' or 'observation'")

        if seeding_method not in ["add", "multiply"]:
            raise ValueError("seeding_method takes only values 'add' or 'multiply'")

        if seed == "general" and random_state:
            if not isinstance(random_state, int):
                raise ValueError(
                    "if seed == 'general' then random_state must take an integer"
                )

        if seed == "observation" and not random_state:
            raise ValueError(
                "if seed == 'observation' the random state must take the name of one "
                "or more variables which will be used to seed the imputer"
            )

        self.variables = _check_variables_input_value(variables)
        self.random_state = random_state
        self.seed = seed
        self.seeding_method = seeding_method

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Makes a copy of the train set. Only stores a copy of the variables to impute.
        This copy is then used to randomly extract the values to fill the missing data
        during transform.

        Parameters
        ----------

        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y: None
            y is not needed in this imputation. You can pass None or y.
        """

        # check input dataframe
        X = check_X(X)

        # find variables to impute
        if self.variables is None:
            self.variables_ = find_all_variables(X)
        else:
            self.variables_ = check_all_variables(X, self.variables)

        # take a copy of the selected variables
        self.X_ = X[self.variables_].copy()

        # check the variables assigned to the random state
        if self.seed == "observation":
            self.random_state = _check_variables_input_value(self.random_state)
            if isinstance(self.random_state, (int, str)):
                self.random_state = [self.random_state]
            if self.random_state and any(
                var for var in self.random_state if var not in X.columns
            ):
                raise ValueError(
                    "There are variables assigned as random state which are not part "
                    "of the training dataframe."
                )

        self._get_feature_names_in(X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace missing data with random values taken from the train set.

        Parameters
        ----------

        X: pandas dataframe of shape = [n_samples, n_features]
            The dataframe to be transformed.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features]
            The dataframe without missing values in the transformed variables.
        """

        X = self._transform(X)

        # random sampling with a general seed
        if self.seed == "general":
            for feature in self.variables_:
                if X[feature].isnull().sum() > 0:
                    # determine number of data points to extract at random
                    n_samples = X[feature].isnull().sum()

                    # extract values
                    random_sample = (
                        self.X_[feature]
                        .dropna()
                        .sample(n_samples, replace=True, random_state=self.random_state)
                    )
                    # re-index: pandas needs this to add the values to the right
                    # observations
                    random_sample.index = X[X[feature].isnull()].index

                    # replace na
                    X.loc[X[feature].isnull(), feature] = random_sample

        # random sampling observation per observation
        elif self.seed == "observation" and self.random_state:
            for feature in self.variables_:
                if X[feature].isnull().sum() > 0:

                    # loop over each observation with missing data
                    for i in X[X[feature].isnull()].index:
                        # find the seed using additional variables
                        internal_seed = _define_seed(
                            X, i, self.random_state, how=self.seeding_method
                        )

                        # extract 1 value at random
                        random_sample = (
                            self.X_[feature]
                            .dropna()
                            .sample(1, replace=True, random_state=internal_seed)
                        )
                        random_sample = random_sample.values[0]

                        # replace the missing data point
                        X.loc[i, feature] = random_sample
        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["allow_nan"] = True
        tags_dict["variables"] = "all"
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
