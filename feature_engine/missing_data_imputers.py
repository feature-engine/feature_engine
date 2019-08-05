# Authors: Soledad Galli <solegalli1@gmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
#import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.base_transformers import BaseNumericalImputer, BaseNumericalTransformer
from feature_engine.base_transformers import BaseCategoricalTransformer, _define_variables



# Numerical imputers
class MeanMedianImputer(BaseNumericalImputer):
    """ 
    The MeanMedianImputer() transforms features by replacing missing data in each
    variable by the mean or median value of the variable.
    
    The MeanMedianImputer() works only with numerical variables.
    
    A list of variables to impute can be indicated in a list. If no variable list
    is passed the MeanMedianImputer() will automatically find and select all
    variables of type numeric.
    
    The estimator first calculates the mean / median values for the
    indicated variables (fit).
    
    The estimator then fills the missing data with the estimated mean / median 
    (transform).
    
    Parameters
    ----------
    
    imputation_method : str, default=median 
        Desired method of imputation. Can take 'mean' or 'median'.
        
    Attributes
    ----------
    
    variables : list, default=None
        The list of variables to be imputed. If None, the imputer will find and 
        select all numerical type variables.
       
    imputer_dict_: dictionary
        The dictionary containing the mean / median values to use for each variable
        to replace missing data. It is calculated when fitting the imputer.
    """
    
    def __init__(self, imputation_method='median', variables = None):
        
        if imputation_method not in ['median', 'mean']:
            raise ValueError("Imputation method takes only values 'median' or 'mean'")
            
        self.imputation_method = imputation_method
        self.variables = _define_variables(variables)
        

    def fit(self, X, y=None):
        """ 
        Learns the mean or median values that should be used to replace mising
        data in each variable.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just seleted variables.
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking. You can pass None or y.
        """
        # brings the variables from the BaseImputer
        super().fit(X, y)
            
        if self.imputation_method == 'mean':
            self.imputer_dict_ = X[self.variables].mean().to_dict()
            
        elif self.imputation_method == 'median':
            self.imputer_dict_ = X[self.variables].median().to_dict()
        
        self.input_shape_ = X.shape    
        
        return self



class EndTailImputer(BaseNumericalImputer):
    """ 
    The EndTailImputer() transforms features by replacing missing data in each
    feature for a given value at the tail of the distribution.
    
    The EndTailImputer() works only with numerical variables. A list of variables
    to impute can be indicated in a list. If no variable list is passed the
    EndTailImputer() will automatically find and select all variables of type numeric.
    
    The estimator first calculates the values at the end of the distribution
    for the indicated features. The values at the end of the distribution can 
    be given by the Gaussian limits or the IQR proximity rule limits.
    
    Gaussian limits:
        right tail: mean + 3* std
        left tail: mean - 3* std
        
    IQR limits:
        right tail: 75th Quantile + 3* IQR
        left tail:  25th quantile - 3* IQR
        
    where IQR is the inter-quantal range = 75th Quantile - 25th quantile
        
    You can select how far out you want to place your missing values by tuning
    the number by which you multiply the std or the IQR, using the parameter
    'fold'.
    
    The encoder then fills the missing data with the estimated values.
    
    Parameters
    ----------
    
    distribution : str, default=gaussian 
        Desired distribution. Can take 'gaussian' or 'skewed'.
        gaussian: the encoder will use the Qaussian limits  to find the values
        to replace missing data.
        skewed: the encoder will use the IQR limits to find the values to replace
        missing data.
        
    tail : str, default=right
        Whether the missing values will be placed at the right or left tail of 
        the variable distribution. Can take 'left' or 'right'.
        
    fold: int, default=3
        How far out to place the missing data. Fold will multiply the std or the
        IQR. Recommended values, 2 or 3 for Gaussian, 1.5 or 3 for skewed.
    
    variables : list, default=None
        The list of variables to be imputed. If None, the imputer will find and
        select all numerical type variables.
        
    Attributes
    ----------
    
    imputer_dict_: dictionary
        The dictionary containing the values at end of the distribution to use
        to replace missing data for each variable. The values are calculated
        when fitting the imputer.
        
    """
    
    def __init__(self, distribution='gaussian', tail='right', fold=3, variables = None):
        
        if distribution not in ['gaussian', 'skewed']:
            raise ValueError("distribution takes only values 'gaussian' or 'skewed'")
            
        if tail not in ['right', 'left']:
            raise ValueError("tail takes only values 'right' or 'left'")
            
        if fold <=0 :
            raise ValueError("fold takes only positive numbers")
            
        self.distribution = distribution
        self.tail = tail
        self.fold = fold
        
        self.variables = _define_variables(variables)


    def fit(self, X, y=None, ):
        """
        Learns the values that should be used to replace mising data in each 
        variable.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just selected variables.
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking.
        """
        # brings the variables from the BaseImputer
        super().fit(X, y)

        # estimate the end values
        if self.tail == 'right':
            if self.distribution == 'gaussian':
                self.imputer_dict_ = (X[self.variables].mean()+self.fold*X[self.variables].std()).to_dict()
                
            elif self.distribution == 'skewed':
                IQR = X[self.variables].quantile(0.75) - X[self.variables].quantile(0.25)
                self.imputer_dict_ = (X[self.variables].quantile(0.75) + (IQR * self.fold)).to_dict()
        
        elif self.tail == 'left':
            if self.distribution == 'gaussian':
                self.imputer_dict_ = (X[self.variables].mean()-self.fold*X[self.variable_].std()).to_dict()
                
            elif self.distribution == 'skewed':
                IQR = X[self.variables].quantile(0.75) - X[self.variables].quantile(0.25)
                self.imputer_dict_ = (X[self.variables].quantile(0.25) - (IQR * self.fold)).to_dict()        
        
        self.input_shape_ = X.shape  
              
        return self



class CategoricalVariableImputer(BaseCategoricalTransformer):
    """
    The CategoricalVariableImputer() replaces missing data in categorical variables
    by the string 'Missing'.
    
    The CategoricalVariableImputer() works only with categorical variables.
    
    A list of variables to impute can be indicated. If no variable list is passed
    the CategoricalVariableImputer() will automatically find and select all
    variables of type object.
       
    Parameters
    ----------
    
    variables : list, default=None
        The list of variables to be imputed. If None, the imputer will find and
        select all object type variables.
    """
    
    def __init__(self, variables=None):
        
        self.variables = _define_variables(variables)
            

    def fit(self, X, y=None):
        """ 
        Checks that the selected variables are categorical.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just seleted variables
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking.

        """
        super().fit(X,y)

        self.input_shape_ = X.shape  
        
        return self

    def transform(self, X):
        """
        Replaces missing values with the new Label 'Missing'.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.
        
        Returns
        -------
        X_transformed : dataframe of shape = [n_samples, n_features]
            The dataframe containing no missing values for the selected
            variables
        """
        
        # Check is fit had been called
        check_is_fitted(self, ['input_shape_'])
        
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the imputer')

        X = X.copy()
        X.loc[:, self.variables] = X[self.variables].fillna('Missing')
        
        return X
    

class FrequentCategoryImputer(BaseCategoricalTransformer):
    """
    The FrequentCategoryImputer() replaces missing data in categorical variables
    by the most frequent category.
    
    The FrequentCategoryImputer() works only with categorical variables.
    
    A list of variables to impute can be indicated. If no variable list is passed
    the FrequentCategoryImputer() will automatically find and select all
    variables of type object.
    
    The encoder first finds the most frequent category per variable (fit).
    
    The encoder then fills the missing data with the most frequent category 
    (transform).
    
    Parameters
    ----------
    
    variables : list, default=None
        The list of variables to be imputed. If None, the transformer will find
        and select all object type variables.
        
    Attributes
    ----------   
    
    imputer_dict_: dictionary
        The dictionary mapping each variable to the most frequent category which
        will be used to fill missing data. These are calculated when fitting the
        transformer.   
    """
    
    def __init__(self, variables=None):
        
        self.variables = _define_variables(variables)

    def fit(self, X, y=None):
        """
        Learns the most frequent category for each variable.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just seleted variables
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking.

        """
        super().fit(X,y)
            
        self.imputer_dict_ = {}
        
        for var in self.variables:
            self.imputer_dict_[var] = X[var].mode()[0]
                    
        self.input_shape_ = X.shape   
        
        return self

    def transform(self, X):
        """ 
        Replaces missing data with the most frequent category of the variable.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.
            
        Returns
        -------
        
        X_transformed : dataframe of shape = [n_samples, n_features]
            The dataframe containing no missing values for the selected
            variables
        """
        
        # Check is fit had been called
        check_is_fitted(self, ['imputer_dict_'])
        
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the imputer')

        X = X.copy()
        
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
            
        return X



class RandomSampleImputer(BaseEstimator, TransformerMixin):
    """ 
    The RandomSampleImputer() replaces missing data in each feature with a random
    sample extracted from the variables in the training set.
    
    The RandomSampleImputer() works with both numerical and categorical variables.
    
    Note: random samples will vary from execution to execution. This may affect
    the results of your work. Remember to set a seed before running the
    RandomSampleImputer().
    
    There are 2 ways in which the seed can be set with the RandomSampleImputer():
    If the seed = 'general' then the random_state can be either None or an integer.
    The seed will be initialised to the random_state and all observations will be
    imputed in one go.
    If the seed = 'observation', then the random_state should be a variable name
    or a list of variable names. The seed will be calculated, observation per 
    observation, either adding or multiplying the variable values indicated in the
    list passed to the random_state.
    
    For more details on why this functionality is important refer to the course 
    Feature Engineering for Machine Learning in Udemy:
    https://www.udemy.com/feature-engineering-for-machine-learning/
    
    Note, if the variables indicated in the random_state list are not numerical
    the imputer will return an error.
    
    This estimator stores a copy of the training set when the fit() method is 
    called. Therefore, the object can become quite heavy. Also, it may not be GDPR
    compliant if your training dataset contains Personal Information.
    
    The transformer fills the missing data with a random sample from
    the training set.
    
    Parameters
    ----------
    
    random_state : int, str or list, default=None
        The random_state can take an integer to set the seed when extracting the
        random samples. Alternatively, it can take a variable name or a list of
        variables, which values will be used to set the seed, observation per 
        observation.
        
    seed: str, default='general'
        Indicates wheter the seed should be set for each observation to impute
        or one seed should be used for a batch of imputations.
        general: one seed will be used to impute the entire dataframe. This is 
        the equivalent of setting the seed in pandas.sample(random_state)
        observation: the seed will be set per each observation using the values
        of the variables indicated in the random_state.
        
    seeding_method : str, default='add'
        If more than one variable are indicated to seed the randoms sampling per
        observation, you can choose to combine those values as an addition or a 
        multiplication. Can take the values 'add' or 'multiply'.
        
    variables : list, default=None
        The list of variables to be imputed. If None, the imputer will select 
        all present in the train set.
        
    Attributes
    ----------
    
    X : dataframe.
        Copy of the training dataframe from which to extract the random samples.
        
    """
    
    def __init__(self, variables = None, random_state = None, seed = 'general',
                 seeding_method = 'add'):
        
        if seed not in ['general', 'observation']:
            raise ValueError("seed takes only values 'general' or 'observation'")
            
        if seeding_method not in ['add', 'multiply']:
            raise ValueError("seeding_method takes only values 'add' or 'multiply'")
            
        if seed == 'general' and random_state:
            if not isinstance(random_state, int):
                raise ValueError("if the seed == 'general' the random state must take an integer")
            
        if isinstance(random_state, str):
            random_state = list(random_state)
        
        self.variables = _define_variables(variables)
        self.random_state = random_state
        self.seed = seed
        self.seeding_method = seeding_method


    def fit(self, X, y=None):
        """ 
        Makes a copy of the indicated variables in the training dataframe, from
        which it will randomly extract the values during transform.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just seleted variables.    
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking.
        """
           
        if not self.variables:
            self.X = X.copy()
            self.variables = [x for x in X.columns]
        else:
            self.X = X[self.variables].copy()
            
        self.input_shape_ = X.shape         
        
        return self
    

    def transform(self, X):
        """
        Replaces missing data with random values taken from the train set.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.
            
        Returns
        -------
        
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe containing NO missing values for all indicated variables
        """

        # Check is fit had been called
        check_is_fitted(self, ['X'])
        
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the imputer')

        X = X.copy()
        
        # random sampling with a general seed
        if self.seed=='general':
            for feature in self.variables:
                if X[feature].isnull().sum()>0:
                    n_samples = X[feature].isnull().sum()
                    
                    random_sample = self.X[feature].dropna().sample(n_samples,
                                                                    replace=True, 
                                                                    random_state=self.random_state
                                                                    )
                    
                    random_sample.index = X[X[feature].isnull()].index
                    X.loc[X[feature].isnull(), feature] = random_sample
                    
        # random sampling observation per observation
        elif self.seed=='observation':
            for feature in self.variables:
                if X[feature].isnull().sum()>0:
                    df_ls = []
                    for index, row in X[X[feature].isnull()].iterrows():

                        if self.seeding_method=='add':
                            random_state = int(row[self.random_state].sum())
                        elif self.seeding_method == 'multiply':
                            random_state = int(row[self.random_state].multiply())
                            
                        random_sample = self.X[feature].dropna().sample(1,
                                                                    replace=True,
                                                                    random_state=random_state
                                                                    )
                    
                        random_sample.index = [row.name]
                        df_ls.append(random_sample)
                        
                    random_sample = pd.concat(df_ls, axis=0)    
                    X.loc[X[feature].isnull(), feature] = random_sample                
        return X
    
       

class AddNaNBinaryImputer(BaseEstimator, TransformerMixin):
    """
    The AddNaNBinaryImputer() adds an additional column or binary variable to
    indicating if data is missing for the selected variables. AddNaNBinaryImputer()
    will add as many missing indicators as variables are selected.
    
    The AddNaNBinaryImputer() works with both numerical and categorical variables.
    A list of variables can be indicated. If None, the imputer will select and 
    add missing indicators to all variables present in the training set.
        
    Parameters
    ----------
    
    variables : list, default=None
        The list of variables to be imputed. If None, the imputer will find and
        select all variables.     
    """
    
    def __init__(self, variables=None):
        self.variables = _define_variables(variables)
            

    def fit(self, X, y=None):
        """ 
        Learns the variables for which the additionalmissing indicator will be
        created.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking.

        """
        if not self.variables:
            self.variables = [var for var in X.columns]

        self.input_shape_ = X.shape
        
        return self

    def transform(self, X):
        """ 
        Adds the additional binary missing indicator variables indicating the
        presence of missing data per observation.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.
            
        Returns
        -------
        
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe containing the additional binary variables
        """
        # Check is fit had been called
        check_is_fitted(self, ['input_shape_'])
        
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the imputer')
            
        X = X.copy()
        for feature in self.variables:
            X[str(feature)+'_na'] = np.where(X[feature].isnull(),1,0)
        
        return X
    
 
class ArbitraryNumberImputer(BaseNumericalTransformer):
    """
    The ArbitraryNumberImputer() transforms features by replacing missing data
    in each feature for a given arbitrary value determined by the user.
       
    Parameters
    ----------
    
    arbitrary_number : int or float, default=-999
        the number to be used to replace missing data.
        
    variables : list, default=None
        The list of variables to be imputed. If None, the imputer will find and 
        select all numerical type variables.        
    """
    
    def __init__(self, arbitrary_number = -999, variables = None):
        
        if isinstance(arbitrary_number, int) or isinstance(arbitrary_number, float):
            self.arbitrary_number = arbitrary_number
        else:
            raise ValueError('Arbitrary number must be numeric of type int or float')
        
        self.variables = _define_variables(variables)
        

    def fit(self, X, y=None):
        """
        Checks that the variables are numerical.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the variables to impute.
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking.
        """
        super().fit(X,y)
                  
        self.input_shape_ = X.shape
        
        return self

    def transform(self, X):
        """
        Replaces missing data with the arbitrary values.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.
            
        Returns
        -------
        
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe containing no missing values for the selected
            variables
        """
        # Check is fit had been called
        check_is_fitted(self, ['input_shape_'])
        
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the imputer')
        
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna(self.arbitrary_number)
        
        return X