# Authors: Soledad Galli <solegalli1@gmail.com>
# License: BSD 3 clause

import numpy as np
#import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted



class BaseNumericalImputer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y = None):
        
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        if not self.variables:
            # select all numerical variables
            self.variables = list(X.select_dtypes(include=numerics).columns)            
        else:
            # variables indicated by user
            if len(X[self.variables].select_dtypes(exclude=numerics).columns) != 0:
               raise ValueError("Some of the selected variables are NOT numerical. Please cast them as numerical before fitting the imputer")
            
            self.variables = self.variables


    def transform(self, X):
        """ Replaces missing data with the calculated parameters
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe containing NO missing values for the selected
            variables
        """
        # Check is fit had been called
        check_is_fitted(self, ['imputer_dict_'])
        
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise TypeError('Number of columns in dataset is different from training set used to fit the encoder')
        
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        
        return X



class BaseCategoricalImputer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y = None):
        
        if not self.variables:
            # select all categorical variables
            self.variables = [var for var in X.columns if X[var].dtypes=='O']
        else:
            # variables indicated by user
            for var in self.variables:
                if X[var].dtypes != 'O':
                    raise TypeError("variable {} is not of type object, check that all indicated variables are of type object".format(var))
            self.variables = self.variables
        
        return self.variables


# Numerical imputers
class MeanMedianImputer(BaseNumericalImputer):
    """ Transforms features by replacing missing data in each variable by the
    mean or median value of that variable.
    
    This Imputer works only with numerical variables.
    
    The estimator first calculates the mean / median values for the
    indicated variables.
    
    The transformer fills the missing data with the estimated mean / median.
    
    Parameters
    ----------
    imputation_method : str, default=median 
        Desired method of imputation.
        Can take 'mean' or 'median'.
        
    Attributes
    ----------
    variables : list
        The list of variables to be imputed. If none, it defaults to all numerical
        type variables.
       
    imputer_dict_: dictionary
        The dictionary containg the mean / median values to use
        for each variable to replace missing data
        
    """
    
    def __init__(self, imputation_method='median', variables = None):
        
        if imputation_method not in ['median', 'mean']:
            raise ValueError("Imputation method takes only values 'median' or 'mean'")
            
        self.imputation_method = imputation_method
        self.variables = variables
        

    def fit(self, X, y=None):
        """ Learns the mean or median that should be used to replace
        mising data in each variable.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Should be the entire dataframe, not just seleted variables
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
    """ Transforms features by replacing missing data in each feature
    for a given value at the tail of the distribution.
    
    This Imputer works only with numerical variables.
    
    The estimator first calculates the values at the end of the distribution
    for the indicated features. The values at the end of the distribution can 
    be given by the Gaussian limits or the quantile limits.
    
    Gaussian limits:
        right tail: mean + 3* std
        left tail: mean - 3* std
        
    Quantile limits:
        right tail: 75th Quantile + 3* IQR
        left tail:  25th quantile - 3* IQR
        
        where IQR is the inter-quantal range.
        
    You can select to tune how far out you place your missing values by tuning
    the number by which you multiply the std or the IQR, using the parameter
    fold.
    
    The transformer fills the missing data with the estimated values.
    
    Parameters
    ----------
    distribution : str, default=gaussian 
        Desired distribution. 
        Can take 'gaussian' or 'skewed'.
        
    tail : str, default=right
        Whether the missing values will be placed at the right or left tail.
        Can take 'left' or 'right'
        
    fold: int, default=3
        How far out to place the missing data. Fold will multiply the std or the
        IQR. Recommended values, 2 or 3 for Gaussian, 1.5 or 3 for skewed.
    
    variables : list
        The list of variables to be imputed. If none, it defaults to all numerical
        type variables.
        
    Attributes
    ----------
    imputer_dict_: dictionary
        The dictionary containg the values at end of tails to use to replace
        missing data for each variable.
        
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
        self.variables = variables


    def fit(self, X, y=None, ):
        """ Learns the values that should be used to replace
        mising data in each variable.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Must be the entire dataframe, not just selected variables.
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



class AddMissingLabelCategoricalImputer(BaseCategoricalImputer):
    """ Replaces missing data in categorical variables by adding the label
    'Missing'.
    
    This Imputer works only with caterical variables.
       
    Parameters
    ----------
    variables : list
        The list of variables to be imputed. If none, it defaults to all object
        type variables.
        
    Attributes
    ----------
.  
    """
    
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        """ Checks that the selected variables are categorical.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking.

        """
        super().fit(X,y)
            
#        self.imputer_dict_ = {}
        
#        for var in self.variables:
#            if X[var].isnull().mean() >= self.tol:
#                self.imputer_dict_[var] = 'Missing'
#    
#            elif len(X[var].unique()) > 10:
#                    self.imputer_dict_[var] = 'Missing'
#            
#            else:
#                temp = X.groupby(var)[var].count() / np.float(len(X))
#                if len(temp[temp<self.tol].index) > 1:
#                    self.imputer_dict_[var] = 'Missing'
#                else:
#                    self.imputer_dict_[var] = X[var].mode()[0]
                    
        self.input_shape_ = X.shape  
        
        return self

    def transform(self, X):
        """ Replaces NAN with the new Label 'Missing'.
        
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
            raise ValueError('Number of columns in dataset is different from training set used to fit the encoder')

        X = X.copy()
        X = X[self.variables].fillna('Missing')
        
        return X
    

class FrequentCategoryImputer(BaseCategoricalImputer):
    """ Replaces missing data in categorical variables by the most frequent
    category.
    
    This Imputer works only with caterical variables.
    
    The transformer fills the missing data.
    
    Parameters
    ----------
    variables : list
        The list of variables to be imputed. If none, it defaults to all object
        type variables.
        
    Attributes
    ----------       
    imputer_dict_: dictionary
        The dictionary mapping each variable to the most frequent category which
        will be used to fill NA.    
    """
    
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        """ Learns the most frequent category for each variable.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
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
        """ Replaces missing data with the calculated parameters
        
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
            raise ValueError('Number of columns in dataset is different from training set used to fit the encoder')

        X = X.copy()
        
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
            
        return X



class RandomSampleImputer(BaseEstimator, TransformerMixin):
    """ Replaces missing data in each feature with a random sample extracted
    from the training set.
    
    This class works for both numerical and categorical variables.
    
    This class could be used at the end of each feature engineering pipeline
    to tackle coming missing values not seen during training during model
    deployment.
    
    Note: random samples will vary from execution to execution. This may affect
    the results of your work. Try and remember to set a seed before running the
    RandomSampleImputer.transform() method.
    
    This estimator stores a copy of the training set. Therefore, it can become
    quite heavy. Also, it may not be GDPR compliant if you store Personal
    information in your training set.
    
    The transformer fills the missing data with a random sample from
    the training set.
    
    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The training input samples.
        Should be the entire dataframe, not just seleted variables.
    y : None
        y is not needed in this transformer, yet the sklearn pipeline API
        requires this parameter for checking.
        
    variables : list
        The list of variables to be imputed. If none, it defaults to all
        variables.
        
    Attributes
    ----------
    X : dataframe.
        Copy of the training dataframe from which to extract the random samples.
        
    """
    
    def __init__(self, variables = None):
        
        self.variables = variables


    def fit(self, X, y=None):
        """ Makes a copy of the desired variables of the dataframe, from
        which it will randomly extract the values during transform.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Should be the entire dataframe, not just seleted variables.    
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking.

        """
           
        if not self.variables:
            self.X = X            
        else:
            self.X = X[self.variables]
            
        self.input_shape_ = X.shape         
        
        return self

    def transform(self, X):
        """ Replaces missing data with random values taken from the train set.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe containing NO missing values for all indicated variables
        """
        #global random_state
        # Check is fit had been called
        check_is_fitted(self, ['X'])
        
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the encoder')

        X = X.copy()
        for feature in self.variables:
            if X[feature].isnull().sum()>0:
                n_samples = X[feature].isnull().sum()
                
                random_sample = self.X[feature].dropna().sample(n_samples,
                                                                replace=True, 
                                                                #random_state=random_state
                                                                )
                
                random_sample.index = X[X[feature].isnull()].index
                X.loc[X[feature].isnull(), feature] = random_sample
                
        return X
    
       

class AddNaNBinaryImputer(BaseEstimator, TransformerMixin):
    """ Adds an additional column / binary variable to capture missing
    information for the selected variables.
    
    This class works for both numerical and categorical variables.
    
    Parameters
    ----------
    variables : list
        The list of variables to be imputed. If none, it defaults to all
        variables.
        
    Attributes
    ----------
     
    """
    
    def __init__(self, variables=None):

        self.variables = variables

    def fit(self, X, y=None):
        """ Learns the variables for which the additional variable 
        should be created.
        
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
        """ Adds the additional binary variables indicating the presence of 
        NA per observation.
        
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
        check_is_fitted(self, ['variables'])
        
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the encoder')
            
        X = X.copy()
        for feature in self.variables:
            X[str(feature)+'_na'] = np.where(X[feature].isnull(),1,0)
        
        return X
    
 
class ArbitraryNumberImputer(BaseNumericalImputer):
    """ Transforms features by replacing missing data in each feature
    for a given value determined by the user.
       
    Parameters
    ----------
    arbitrary_number : int or float
        the number to be used to replace missing data.
    variables : list
        The list of variables to be imputed. If none, it defaults to all numerical
        type variables.
        
    Attributes
    ----------
        
    """
    
    def __init__(self, arbitrary_number = -999, variables = None):
        
        if isinstance(arbitrary_number, int) or isinstance(arbitrary_number, float):
            self.arbitrary_number = arbitrary_number
        else:
            raise ValueError('Arbitrary number must be numeric of type int or float')
        
        self.arbitrary_number = arbitrary_number
        self.variables = variables
        

    def fit(self, X, y=None):
        """ Checks that the variables are categorical.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking.
        variables: list of columns that should be transformed

        """
        super().fit(X,y)
                  
        self.input_shape_ = X.shape
        
        return self

    def transform(self, X):
        """ Replaces missing data with the calculated parameters
        
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
            raise ValueError('Number of columns in dataset is different from training set used to fit the encoder')
        
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.arbitrary_number, inplace=True)
        
        return X