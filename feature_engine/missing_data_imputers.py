# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import deprecated

from feature_engine.dataframe_checks import _is_dataframe, _check_input_matches_training_df
from feature_engine.variable_manipulation import _find_categorical_variables, _define_variables, \
    _find_numerical_variables
from feature_engine.base_transformers import BaseImputer


# for RandomSampleImputer
def _define_seed(X, index, seed_variables, how='add'):
    # determine seed by adding or multiplying the value of 1 or
    # more variables
    if how == 'add':
        internal_seed = int(np.round(X.loc[index, seed_variables].sum(), 0))
    elif how == 'multiply':
        internal_seed = int(np.round(X.loc[index, seed_variables].product(), 0))
    return internal_seed


class MeanMedianImputer(BaseImputer):
    """
    The MeanMedianImputer() transforms features by replacing missing data by the mean
    or median value of the variable.

    The MeanMedianImputer() works only with numerical variables.

    Users can pass a list of variables to be imputed as argument. Alternatively, the
    MeanMedianImputer() will automatically find and select all variables of type numeric.

    The imputer first calculates the mean / median values of the variables (fit).

    The imputer then replaces the missing data with the estimated mean / median (transform).

    Parameters
    ----------

    imputation_method : str, default=median
        Desired method of imputation. Can take 'mean' or 'median'.

    variables : list, default=None
        The list of variables to be imputed. If None, the imputer will select
        all variables of type numeric.
    """

    def __init__(self, imputation_method='median', variables=None):

        if imputation_method not in ['median', 'mean']:
            raise ValueError("imputation_method takes only values 'median' or 'mean'")

        self.imputation_method = imputation_method
        self.variables = _define_variables(variables)

    def fit(self, X, y=None):
        """
        Learns the mean or median values.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            User can pass the entire dataframe, not just the variables that need imputation.

        y : pandas series or None, default=None
            y is not needed in this imputation. You can pass None or y.

        Attributes
        ----------

        imputer_dict_: dictionary
            The dictionary containing the mean / median values per variable. These
            values will be used by the imputer to replace missing data.
            The imputer_dict_ is created when fitting the imputer.
        """
        # check input dataframe
        X = _is_dataframe(X)

        # find or check for numerical variables
        self.variables = _find_numerical_variables(X, self.variables)

        # find imputation parameters: mean or median
        if self.imputation_method == 'mean':
            self.imputer_dict_ = X[self.variables].mean().to_dict()

        elif self.imputation_method == 'median':
            self.imputer_dict_ = X[self.variables].median().to_dict()

        self.input_shape_ = X.shape

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise none of this is necessary
    def transform(self, X):
        X = super().transform(X)
        return X

    transform.__doc__ = BaseImputer.transform.__doc__


class EndTailImputer(BaseImputer):
    """
    The EndTailImputer() transforms features by replacing missing data by a
    value at either tail of the distribution.

    The EndTailImputer() works only with numerical variables.

    The user can indicate the variables to be imputed in a list. Alternatively, the
    EndTailImputer() will automatically find and select all variables of type numeric.

    The imputer first calculates the values at the end of the distribution for each variable
    (fit). The values at the end of the distribution are determined using the Gaussian limits,
    the the IQR proximity rule limits, or a factor of the maximum value:

    Gaussian limits:
        right tail: mean + 3*std

        left tail: mean - 3*std

    IQR limits:
        right tail: 75th quantile + 3*IQR

        left tail:  25th quantile - 3*IQR

    where IQR is the inter-quartile range = 75th quantile - 25th quantile

    Maximum value:
        right tail: max * 3

        left tail: not applicable

    You can change the factor that multiplies the std, IQR or the maximum value
    using the parameter 'fold'.

    The imputer then replaces the missing data with the estimated values (transform).

    Parameters
    ----------

    distribution : str, default=gaussian
        Method to be used to find the replacement values. Can take 'gaussian',
        'skewed' or 'max'.

        gaussian: the imputer will use the Gaussian limits to find the values
        to replace missing data.

        skewed: the imputer will use the IQR limits to find the values to replace
        missing data.

        max: the imputer will use the maximum values to replace missing data. Note
        that if 'max' is passed, the parameter 'tail' is ignored.

    tail : str, default=right
        Indicates if the values to replace missing data should be selected from the right
        or left tail of the variable distribution. Can take values 'left' or 'right'.

    fold: int, default=3
        Factor to multiply the std, the IQR or the Max values. Recommended values
        are 2 or 3 for Gaussian, or 1.5 or 3 for skewed.

    variables : list, default=None
        The list of variables to be imputed. If None, the imputer will find and
        select all variables of type numeric.
    """

    def __init__(self, distribution='gaussian', tail='right', fold=3, variables=None):

        if distribution not in ['gaussian', 'skewed', 'max']:
            raise ValueError("distribution takes only values 'gaussian', 'skewed' or 'max'")

        if tail not in ['right', 'left']:
            raise ValueError("tail takes only values 'right' or 'left'")

        if fold <= 0:
            raise ValueError("fold takes only positive numbers")

        self.distribution = distribution
        self.tail = tail
        self.fold = fold
        self.variables = _define_variables(variables)

    def fit(self, X, y=None):
        """
        Learns the values at the end of the variable distribution.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            The user can pass the entire dataframe, not just the variables that need imputation.

        y : None
            y is not needed in this imputation. You can pass None or y.

        Attributes
        ----------

        imputer_dict_: dictionary
            The dictionary containing the values at the end of the distribution
            per variable. These values will be used by the imputer to replace missing
            data.
        """
        # check input dataframe
        X = _is_dataframe(X)

        # find or check for numerical variables
        self.variables = _find_numerical_variables(X, self.variables)

        # estimate imputation values
        if self.distribution == 'max':
            self.imputer_dict_ = (X[self.variables].max() * self.fold).to_dict()

        elif self.distribution == 'gaussian':
            if self.tail == 'right':
                self.imputer_dict_ = (X[self.variables].mean() + self.fold * X[self.variables].std()).to_dict()
            elif self.tail == 'left':
                self.imputer_dict_ = (X[self.variables].mean() - self.fold * X[self.variables].std()).to_dict()

        elif self.distribution == 'skewed':
            IQR = X[self.variables].quantile(0.75) - X[self.variables].quantile(0.25)
            if self.tail == 'right':
                self.imputer_dict_ = (X[self.variables].quantile(0.75) + (IQR * self.fold)).to_dict()
            elif self.tail == 'left':
                self.imputer_dict_ = (X[self.variables].quantile(0.25) - (IQR * self.fold)).to_dict()

        self.input_shape_ = X.shape

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise none of this is necessary
    def transform(self, X):
        X = super().transform(X)
        return X

    transform.__doc__ = BaseImputer.transform.__doc__


class ArbitraryNumberImputer(BaseImputer):
    """
    The ArbitraryNumberImputer() replaces missing data in each variable
    by an arbitrary value determined by the user.

    Parameters
    ----------

    arbitrary_number : int or float, default=999
        the number to be used to replace missing data.

    variables : list, default=None
        The list of variables to be imputed. If None, the imputer will find and
        select all numerical type variables.
    """

    def __init__(self, arbitrary_number=999, variables=None):

        if isinstance(arbitrary_number, int) or isinstance(arbitrary_number, float):
            self.arbitrary_number = arbitrary_number
        else:
            raise ValueError('arbitrary_number must be numeric of type int or float')

        self.variables = _define_variables(variables)

    def fit(self, X, y=None):
        """
        Checks that the variables are numerical.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            User can pass the entire dataframe, not just the variables to impute.

        y : None
            y is not needed in this imputation. You can pass None or y.
        """
        # check input dataframe
        X = _is_dataframe(X)

        # find or check for numerical variables
        self.variables = _find_numerical_variables(X, self.variables)

        # create the imputer dictionary
        self.imputer_dict_ = {var: self.arbitrary_number for var in self.variables}

        self.input_shape_ = X.shape

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise none of this is necessary
    def transform(self, X):
        X = super().transform(X)
        return X

    transform.__doc__ = BaseImputer.transform.__doc__


class CategoricalVariableImputer(BaseImputer):
    """
    The CategoricalVariableImputer() replaces missing data in categorical variables
    by the string 'Missing' or by the most frequent category.

    The CategoricalVariableImputer() works only with categorical variables.

    The user can pass a list with the variables to be imputed. Alternatively,
    the CategoricalVariableImputer() will automatically find and select all
    variables of type object.

    Parameters
    ----------

    imputation_method : str, default=missing
        Desired method of imputation. Can take 'missing' or 'frequent'.

    variables : list, default=None
        The list of variables to be imputed. If None, the imputer will find and
        select all object type variables.

    return_object: bool, default=False
        If working with numerical variables cast as object, decide
        whether to return the variables as numeric or re-cast them as object.
        Note that pandas will re-cast them automatically as numeric after the
        transformation with the mode.

        Tip: return the variables as object if planning to do categorical encoding
        with feature-engine.
    """

    def __init__(self, imputation_method='missing', variables=None, return_object=False):

        if imputation_method not in ['missing', 'frequent']:
            raise ValueError("imputation_method takes only values 'missing' or 'frequent'")

        self.imputation_method = imputation_method
        self.variables = _define_variables(variables)
        self.return_object = return_object

    def fit(self, X, y=None):
        """
        Learns the most frequent category if the imputation method is set to frequent.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the selected variables.

        y : None
            y is not needed in this imputation. You can pass None or y.

        Attributes
        ----------

        imputer_dict_: dictionary
            The dictionary mapping each variable to the most frequent category, or to
            the value 'Missing' depending on the imputation_method. The most frequent
            category is calculated when fitting the transformer.
        """
        # check input dataframe
        X = _is_dataframe(X)

        # find or check for categorical variables
        self.variables = _find_categorical_variables(X, self.variables)

        # find imputation parameters
        if self.imputation_method == 'missing':
            self.imputer_dict_ = {var: 'Missing' for var in self.variables}

        elif self.imputation_method == 'frequent':
            self.imputer_dict_ = {}

            for var in self.variables:
                mode_vals = X[var].mode()

                # careful: some variables contain multiple modes
                if len(mode_vals) == 1:
                    self.imputer_dict_[var] = mode_vals[0]
                else:
                    raise ValueError('The variable {} contains multiple frequent categories.'.format(var))

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        # bring functionality from the BaseImputer
        X = super().transform(X)

        # add additional step to return variables cast as object
        if self.return_object:
            X[self.variables] = X[self.variables].astype('O')
        return X

    # Ugly work around to import the docstring for Sphinx, otherwise none of this is necessary
    transform.__doc__ = BaseImputer.transform.__doc__


@deprecated("Class 'FrequentCategoryImputer' was integrated into the "
            "class 'CategoricalVariableImputer' in version 0.4 and "
            " will be removed in version 0.5. "
            "To perform Frequent category imputation please use: "
            "CategoricalVariableImputer(imputation_method='frequent')")
class FrequentCategoryImputer(CategoricalVariableImputer):
    def __init__(self, variables=None):
        self.imputation_method = 'frequent'
        self.variables = _define_variables(variables)


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

        # Check that the dataframe contains the same number of columns than the dataframe
        # used to fit the imputer.
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


class AddMissingIndicator(BaseEstimator, TransformerMixin):
    """
    The AddMissingIndicator() adds an additional column or binary variable that
    indicates if data is missing.

    AddMissingIndicator() will add as many missing indicators as variables
    indicated by the user, or variables with missing data in the train set.

    The AddMissingIndicator() works for both numerical and categorical variables.
    The user can pass a list with the variables for which the missing indicators
    should be added as a list. Alternatively, the imputer will select and add missing
    indicators to all variables in the training set that show missing data.

    Parameters
    ----------

    how : string, defatul='missing_only'
        Indicates if missing indicators should be added to variables with missing
        data or to all variables.

        missing_only: indicators will be created only for those variables that showed
        missing data during fit.

        all: indicators will be created for all variables

    variables : list, default=None
        The list of variables to be imputed. If None, the imputer will find and
        select all variables with missing data.
        Note: the transformer will first select all variables or all user entered variables
        and if how=missing_only, it will re-select from the original group only those
        that show missing data in during fit.
    """

    def __init__(self, how='missing_only', variables=None):

        if how not in ['missing_only', 'all']:
            raise ValueError("how takes only values 'missing_only' or 'all'")

        self.variables = _define_variables(variables)
        self.how = how

    def fit(self, X, y=None):
        """
        Learns the variables for which the missing indicators will be created.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y : None
            y is not needed in this imputation. You can pass None or y.

        Attributes
        ----------

        variables_: list
            the lit of variables for which the missing indicator will be created.
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find variables for which indicator should be added
        if self.how == 'missing_only':
            if not self.variables:
                self.variables_ = [var for var in X.columns if X[var].isnull().sum() > 0]
            else:
                self.variables_ = [var for var in self.variables if X[var].isnull().sum() > 0]

        elif self.how == 'all':
            if not self.variables:
                self.variables_ = [var for var in X.columns]
            else:
                self.variables_ = self.variables

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        Adds the binary missing indicators.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The dataframe to be transformed.

        Returns
        -------

        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe containing the additional binary variables.
            Binary variables are named with the original variable name plus
            '_na'.
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # Check that the dataframe contains the same number of columns than the dataframe
        # used to fit the imputer.
        _check_input_matches_training_df(X, self.input_shape_[1])

        X = X.copy()
        for feature in self.variables_:
            X[feature + '_na'] = np.where(X[feature].isnull(), 1, 0)

        return X


@deprecated("Class 'AddNaNBinaryImputer' was renamed to AddMissingIndicator "
            "in version 0.4 and will be removed in version 0.5. "
            "To add a missing indicator please use: AddMissingIndicator()")
class AddNaNBinaryImputer(AddMissingIndicator):
    pass
