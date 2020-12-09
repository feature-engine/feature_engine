# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import Optional, List, Union

import pandas as pd
import numpy as np

from feature_engine.dataframe_checks import _is_dataframe
from feature_engine.imputation.base_imputer import BaseImputer
from feature_engine.variable_manipulation import _check_input_parameter_variables


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


class RandomSampleImputer(BaseImputer):
    """
    The RandomSampleImputer() replaces missing data in each feature with a random
    sample extracted from the variables in the training set.
    The RandomSampleImputer() works with both numerical and categorical variables.

    **Note**

    Random samples will vary from execution to execution. This may affect
    the results of your work. Remember to set a seed before running the
    RandomSampleImputer().

    There are 2 ways in which the seed can be set with the RandomSampleImputer():

    If seed = 'general' then the random_state can be either None or an integer.
    The seed will be used as the random_state and all observations will be
    imputed in one go. This is equivalent to `pandas.sample(n, random_state=seed)`
    where n is the number of observations with missing data.

    If seed = 'observation', then the random_state should be a variable name
    or a list of variable names. The seed will be calculated observation per
    observation, either by adding or multiplying the seeding variable values, and
    passed to the random_state. Then, a value will be extracted from the train set
    using that seed and  used to replace the NAN in particular observation. This is the
    equivalent of `pandas.sample(1, random_state=var1+var2)` if the 'seeding_method' is
    set to 'add' or `pandas.sample(1, random_state=var1*var2)` if the 'seeding_method'
    is set to 'multiply'.

    For more details on why this functionality is important refer to the course
    Feature Engineering for Machine Learning in Udemy:
    https://www.udemy.com/feature-engineering-for-machine-learning/

    Note, if the variables indicated in the random_state list are not numerical
    the imputer will return an error. Note also that the variables indicated as seed
    should not contain missing values.

    This estimator stores a copy of the training set when the fit() method is
    called. Therefore, the object can become quite heavy. Also, it may not be GDPR
    compliant if your training data set contains Personal Information. Please check
    if this behaviour is allowed within your organisation.

    Parameters
    ----------
    random_state : int, str or list, default=None
        The random_state can take an integer to set the seed when extracting the
        random samples. Alternatively, it can take a variable name or a list of
        variables, which values will be used to determine the seed observation per
        observation.

    seed : str, default='general'
        Indicates whether the seed should be set for each observation with missing
        values, or if one seed should be used to impute all variables in one go.

        **general**: one seed will be used to impute the entire dataframe. This is
        equivalent to setting the seed in pandas.sample(random_state).

        **observation**: the seed will be set for each observation using the values
        of the variables indicated in the random_state for that particular
        observation.

    seeding_method : str, default='add'
        If more than one variable are indicated to seed the random sampling per
        observation, you can choose to combine those values as an addition or a
        multiplication. Can take the values 'add' or 'multiply'.

    variables : list, default=None
        The list of variables to be imputed. If None, the imputer will select
        all variables in the train set.

    Attributes
    ----------
    X_ :
        Copy of the training dataframe from which to extract the random samples.

    Methods
    -------
    fit:
        Make a copy of the dataframe
    transform:
        Impute missing data.
    fit_transform:
        Fit to the data, then transform it.
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

        self.variables = _check_input_parameter_variables(variables)
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

        X : pandas dataframe of shape = [n_samples, n_features]
            The training dataset. Only a copy of the indicated variables will be stored
            in the transformer.

        y : None
            y is not needed in this imputation. You can pass None or y.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame

        Returns
        -------
        self
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find variables to impute
        if not self.variables:
            self.variables = [var for var in X.columns]
        else:
            self.variables = self.variables

        # take a copy of the selected variables
        self.X_ = X[self.variables].copy()

        # check the variables assigned to the random state
        if self.seed == "observation":
            self.random_state = _check_input_parameter_variables(self.random_state)
            if isinstance(self.random_state, (int, str)):
                self.random_state = [self.random_state]
            if self.random_state and any(
                var for var in self.random_state if var not in X.columns
            ):
                raise ValueError(
                    "There are variables assigned as random state which are not part "
                    "of the training dataframe."
                )
        self.input_shape_ = X.shape

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace missing data with random values taken from the train set.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The dataframe to be transformed.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame

        Returns
        -------
        X : pandas dataframe of shape = [n_samples, n_features]
            The dataframe without missing values in the transformed variables.
        """

        X = self._check_transform_input_and_state(X)

        # random sampling with a general seed
        if self.seed == "general":
            for feature in self.variables:
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
            for feature in self.variables:
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
