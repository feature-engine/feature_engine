# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import _is_dataframe, _check_input_matches_training_df
from feature_engine.variable_manipulation import _define_variables


# for RandomSampleImputer
def _define_seed(X, index, seed_variables, how='add'):
    # determine seed by adding or multiplying the value of 1 or
    # more variables
    if how == 'add':
        internal_seed = int(np.round(X.loc[index, seed_variables].sum(), 0))
    elif how == 'multiply':
        internal_seed = int(np.round(X.loc[index, seed_variables].product(), 0))
    return internal_seed


class RandomSampleImputer(BaseEstimator, TransformerMixin):
    """
    The RandomSampleImputer() replaces missing data in each feature with a random
    sample extracted from the variables in the training set.
    The RandomSampleImputer() works with both numerical and categorical variables.
    Note: random samples will vary from execution to execution. This may affect
    the results of your work. Remember to set a seed before running the
    RandomSampleImputer().

    There are 2 ways in which the seed can be set with the RandomSampleImputer():
    If seed = 'general' then the random_state can be either None or an integer.
    The seed will be used as the random_state and all observations will be
    imputed in one go. This is equivalent to pandas.sample(n, random_state=seed).

    If seed = 'observation', then the random_state should be a variable name
    or a list of variable names. The seed will be calculated, observation per
    observation, either by adding or multiplying the seeding variable values for that
    observation, and passed to the random_state. Thus, a value will be extracted using
    that seed, and used to replace that particular observation. This is the equivalent
    of pandas.sample(1, random_state=var1+var2) if the 'seeding_method' is set to 'add'
    or pandas.sample(1, random_state=var1*var2) if the 'seeding_method' is set to
    'multiply'.

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
    The imputer replaces missing data with a random sample from the training set.

    Parameters
    ----------

    random_state : int, str or list, default=None
        The random_state can take an integer to set the seed when extracting the
        random samples. Alternatively, it can take a variable name or a list of
        variables, which values will be used to determine the seed observation per
        observation.

    seed: str, default='general'
        Indicates whether the seed should be set for each observation with missing
        values, or if one seed should be used to impute all variables in one go.

        general: one seed will be used to impute the entire dataframe. This is
        equivalent to setting the seed in pandas.sample(random_state).

        observation: the seed will be set for each observation using the values
        of the variables indicated in the random_state for that particular
        observation.

    seeding_method : str, default='add'
        If more than one variable are indicated to seed the random sampling per
        observation, you can choose to combine those values as an addition or a
        multiplication. Can take the values 'add' or 'multiply'.

    variables : list, default=None
        The list of variables to be imputed. If None, the imputer will select
        all variables in the train set.
    """

    def __init__(self, variables=None, random_state=None, seed='general', seeding_method='add'):

        if seed not in ['general', 'observation']:
            raise ValueError("seed takes only values 'general' or 'observation'")

        if seeding_method not in ['add', 'multiply']:
            raise ValueError("seeding_method takes only values 'add' or 'multiply'")

        if seed == 'general' and random_state:
            if not isinstance(random_state, int):
                raise ValueError("if seed == 'general' the random state must take an integer")

        if seed == 'observation' and not random_state:
            raise ValueError("if seed == 'observation' the random state must take the name of one or more variables "
                             "which will be used to seed the imputer")

        self.variables = _define_variables(variables)
        self.random_state = random_state
        self.seed = seed
        self.seeding_method = seeding_method

    def fit(self, X, y=None):
        """
        Makes a copy of the variables to impute in the training dataframe from
        which it will randomly extract the values to fill the missing data
        during transform.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just he variables to impute.

        y : None
            y is not needed in this imputation. You can pass None or y.

        Attributes
        ----------

        X_ : dataframe.
            Copy of the training dataframe from which to extract the random samples.
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
        if self.seed == 'observation':
            self.random_state = _define_variables(self.random_state)
            if len([var for var in self.random_state if var not in X.columns]) > 0:
                raise ValueError("There are variables assigned as random state which are not part of the training "
                                 "dataframe.")
        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        Replaces missing data with random values taken from the train set.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The dataframe to be transformed.

        Returns
        -------

        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe without missing values in the transformed variables.
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # Check that input data contains same number of columns as dataframe used to fit
        _check_input_matches_training_df(X, self.input_shape_[1])

        # random sampling with a general seed
        if self.seed == 'general':
            for feature in self.variables:
                if X[feature].isnull().sum() > 0:
                    # determine number of data points to extract at random
                    n_samples = X[feature].isnull().sum()

                    # extract values
                    random_sample = self.X_[feature].dropna().sample(n_samples,
                                                                     replace=True,
                                                                     random_state=self.random_state
                                                                     )
                    # re-index: pandas needs this to add values in the correct observations
                    random_sample.index = X[X[feature].isnull()].index

                    # replace na
                    X.loc[X[feature].isnull(), feature] = random_sample

        # random sampling observation per observation
        elif self.seed == 'observation':
            for feature in self.variables:
                if X[feature].isnull().sum() > 0:

                    # loop over each observation with missing data
                    for i in X[X[feature].isnull()].index:
                        # find the seed using additional variables
                        internal_seed = _define_seed(X, i, self.random_state, how=self.seeding_method)

                        # extract 1 value at random
                        random_sample = self.X_[feature].dropna().sample(1,
                                                                         replace=True,
                                                                         random_state=internal_seed
                                                                         )
                        random_sample = random_sample.values[0]

                        # replace the missing data point
                        X.loc[i, feature] = random_sample
        return X
