"""
Methods to fill missing values in variables
"""

# Authors: Soledad Galli <solegalli1@gmail.com>
# License: BSD 3 clause

import numpy as np

import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class MeanMedianImputer(BaseEstimator, TransformerMixin):
    """ Transforms features by replacing missing data in each feature
    for a given mean or median value.
    
    This class works only with numerical variables.
    
    The estimator first calculates the mean / median values for the
    desired features.
    
    The transformer fills the missing data with the estimated value.
    
    Parameters
    ----------
    imputation_method : str, default=median 
        Desired method of imputation.
        Can take 'mean' or 'median'.
        
    Attributes
    ----------
    variables_ : list
        The list of variables that want to be imputed
        passed to the method:`fit`
        
    imputer_dict_: dictionary
        The dictionary containg the mean / median values to use
        for each variable to replace missing data
        
    """
    
    def __init__(self, imputation_method='median'):
        
        if imputation_method not in ['median', 'mean']:
            raise ValueError("Imputation method takes only values 'median' or 'mean'")
            
        self.imputation_method = imputation_method

    def fit(self, X, y=None, variables = None):
        """ Learns the mean or median that should be used to replace
        mising data in each variable.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Should be the entire dataframe, not just seleted variables
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking.
        variables : list of variables (column names) that should be transformed
        
        Returns
        -------
        self : object
            Returns self.
        """
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        if not variables:
            # select all numerical variables
            self.variables_ = list(X.select_dtypes(include=numerics).columns)            
        else:
            # variables indicated by user
            self.variables_ = variables
            if len(X[self.variables_].select_dtypes(exclude=numerics).columns) != 0:
               raise ValueError("Some of the selected variables are NOT numerical. Please cast them as numerical before fitting the imputer")
            
        if self.imputation_method == 'mean':
            self.imputer_dict_ = X[self.variables_].mean().to_dict()
        elif self.imputation_method == 'median':
            self.imputer_dict_ = X[self.variables_].median().to_dict()
        
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
            The dataframe containing NO missing values for the selected
            variables
        """
        # Check is fit had been called
        check_is_fitted(self, ['imputer_dict_', 'variables_'])
        
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set '
                             ' used to fit the encoder')
        
        X = X.copy()
        for feature in self.variables_:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X


class RandomSampleImputer(BaseEstimator, TransformerMixin):
    """ Transforms features by replacing missing data in each feature
    with a random extracted sample from the training set.
    
    This class works for both numerical and categorical variables
    
    This class could be used at the end of each feature engineering pipeline
    to tackle coming missing values not seen during training during model
    deployment.
    
    This estimator stores a copy of the training set.
    
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
    variables : list of variables (column names) that should be transformed
        
    Attributes
    ----------
    X_ : dataframe.
        Copy of the dataframe stored, from which to extract the
        random samples
        
    """
    
    def __init__(self):
        pass

    def fit(self, X, y=None, variables = None):
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
        variables : list of variables (column names) that should be transformed
        
        Returns
        -------
        self : object
            Returns self.
        """
        if not variables:
            self.X_ = X            
        else:
            self.X_ = X[variables]
            
        self.input_shape_ = X.shape         
        return self

    def transform(self, X, random_state):
        """ Replaces missing data with the calculated parameters
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe containing NO missing values for all variables
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_'])
        
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set '
                             ' used to fit the encoder')

        X = X.copy()
        for feature in self.X_.columns:
            if X[feature].isnull().sum()>0:
                n_samples = X[feature].isnull().sum()
                random_sample = self.X_[feature].dropna().sample(n_samples,
                                                                 replace=True, 
                                                                 random_state=random_state)
                random_sample.index = X[X[feature].isnull()].index
                X.loc[X[feature].isnull(), feature] = random_sample
        return X
    
    def fit_transform(self, X, y=None, variables = None):
        warnings.warn("Fit_transform is not defined for the RandomSampleImputer; fit and transform in separate steps") 
        


class EndTailImputer(BaseEstimator, TransformerMixin):
    """ Transforms features by replacing missing data in each feature
    for a given value at the tail of the distribution
    
    This class works only with numerical variables
    
    The estimator first calculates the values at the end of the distriution
    for the desired features.
    
    The transformer fills the missing data
    
    Parameters
    ----------
    distribution : str, default=gaussian 
        Desired distribution. 
        Can take 'gaussian' or 'skewed'.
        
    tail : str, default=right
        Which tail to pick the value from.
        Can take 'left' or 'right'
        
    fold: int, default=3
        How far out to look for the value.
        Recommended values, 2 or 3 for Gaussian,
        1.5 or 3 for skewed.
        
    Attributes
    ----------
    variables_ : list
        The list of variables that want to be imputed
        passed to the method:`fit`
        
    imputer_dict_: dictionary
        The dictionary containg the values at end of tails to use
        for each variable to replace missing data
        
    """
    
    def __init__(self, distribution='gaussian', tail='right', fold=3):
        
        if distribution not in ['gaussian', 'skewed']:
            raise ValueError("distribution takes only values 'gaussian' or 'skewed'")
            
        if tail not in ['right', 'left']:
            raise ValueError("tail takes only values 'right' or 'left'")
            
        if fold <=0 :
            raise ValueError("fold takes only positive numbers")
            
        self.distribution = distribution
        self.tail = tail
        self.fold = fold

    def fit(self, X, y=None, variables = None):
        """ Learns the values that should be used to replace
        mising data in each variable.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just selected variables.
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking.
        variables: list of columns that should be transformed
        
        Returns
        -------
        self : object
            Returns self.
        """
        
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        if not variables:
            # select all numerical variables
            self.variables_ = list(X.select_dtypes(include=numerics).columns)
            
        else:
            # variables indicated by user
            self.variables_ = variables
            if len(X[self.variables_].select_dtypes(exclude=numerics).columns) != 0:
               raise ValueError("Some of the selected variables are NOT numerical. Please cast them as numerical before fitting the imputer")

        if self.tail == 'right':
            if self.distribution == 'gaussian':
                self.imputer_dict_ = (X[self.variables_].mean()+self.fold*X[self.variables_].std()).to_dict()
            elif self.distribution == 'skewed':
                IQR = X[self.variables_].quantile(0.75) - X[self.variables_].quantile(0.25)
                self.imputer_dict_ = (X[self.variables_].quantile(0.75) + (IQR * self.fold)).to_dict()
        else:
            if self.distribution == 'gaussian':
                self.imputer_dict_ = (X[self.variables_].mean()-self.fold*X[self.variables_].std()).to_dict()
            elif self.distribution == 'skewed':
                IQR = X[self.variables_].quantile(0.75) - X[self.variables_].quantile(0.25)
                self.imputer_dict_ = (X[self.variables_].quantile(0.25) - (IQR * self.fold)).to_dict()        
        
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
            The dataframe containing NO missing values for the selected
            variables
        """
        # Check is fit had been called
        check_is_fitted(self, ['imputer_dict_', 'variables_'])
        
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set '
                             ' used to fit the encoder')

        X = X.copy()
        for feature in self.variables_:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X


class na_capturer(BaseEstimator, TransformerMixin):
    """ Adds an additional column to capture missing information for the
    selected variables
    
    This class works for both numerical and categorical variables
    
    The estimator first identifies those variables for which to add the 
    binary variable to capture NA
    
    The transformer adds the additional variables to the dataframe
    
    Parameters
    ----------
    tol : float, default=0.05 
        Percentage of missing data needed to add an additional variable
        Can vary between 0 and 1
        
    Attributes
    ----------
    variables_ : list
        The list of variables that will be added        
    """
    
    def __init__(self, tol=0.05):
        if tol <0 or tol >1 :
            raise ValueError("tol takes values between 0 and 1")
        self.tol = tol

    def fit(self, X, y=None):
        """ Learns the variables for which the additional variable 
        should be created
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking.
        
        Returns
        -------
        self : object
            Returns self.
        """
        self.variables_ = [var for var in X.columns if X[var].isnull().mean()>=self.tol]
        self.input_shape_ = X.shape
        return self

    def transform(self, X):
        """ Adds additional binary variables to indicate NA to the
        dataframe
        
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
        check_is_fitted(self, ['variables_'])
        
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set '
                             ' used to fit the encoder')
        X = X.copy()
        for feature in self.variables_:
            X[feature+'_na'] = np.where(X[feature].isnull(),1,0)
        return X


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """ Replaces missing data in categorical variables by either adding the 
    label 'Missing' or by the most frequent category.
    
    It determines which method should be used automatically:
        If % NA data >= tol ==> adds "Missing"
        If % NA < tol and number of labels  > 10 ==> adds "Missing"
        If % NA < tol and number of labels < 10 and more than 1 label's frequency < tol ==> adds "Missing"
        Otherwise, replaces NA by most frequent category.
    
    The estimator first determines the percentage of null values, and labels
    and then determines which label will be used to fill NA.
    
    The transformer fills the missing data.
    
    Parameters
    ----------
    tol : float, default=0.05 
        Percentage of missing data needed to add an additional label 'Missing'
        Can vary between 0 and 1
        
    Attributes
    ----------
    variables_ : list
        The list of variables that will be transformed
        
    imputer_dict_: dictionary
        The dictionary mapping each variable to the label that will be used
        to fill NA    
    """
    
    def __init__(self, tol=0.05):
        if tol <0 or tol >1 :
            raise ValueError("tol takes values between 0 and 1")
        self.tol = tol

    def fit(self, X, y=None, variables = None):
        """ Learns whether NA should be filled with 'Missing' or the
        most frequent category
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking.
        
        Returns
        -------
        self : object
            Returns self.
        """
        if not variables:
            # select all categorical variables
            self.variables_ = [var for var in X.columns if X[var].dtypes=='O']
            
        else:
            # variables indicated by user
            self.variables_ = variables
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            if len(X[self.variables_].select_dtypes(include=numerics).columns) != 0:
                warnings.warn("Some of the selected variables are of dtype numeric; they will be handled as categorical variables.\
                               This may cause problems when transforming a variable that does not contain NA, as it will not be casted as numeric. \
                               It is recommended that variables are re-casted as object before fitting this imputer.")
            
        self.imputer_dict_ = {}
        
        for var in self.variables_:
            if X[var].isnull().mean() >= self.tol:
                self.imputer_dict_[var] = 'Missing'
    
            elif len(X[var].unique()) > 10:
                    self.imputer_dict_[var] = 'Missing'
            
            else:
                temp = X.groupby(var)[var].count() / np.float(len(X))
                if len(temp[temp<self.tol].index) > 1:
                    self.imputer_dict_[var] = 'Missing'
                else:
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
        check_is_fitted(self, ['variables_', 'imputer_dict_'])
        
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set '
                             ' used to fit the encoder')

        X = X.copy()
        for feature in self.variables_:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X
    
    
class ArbitraryImputer(BaseEstimator, TransformerMixin):
    """ Transforms features by replacing missing data in each feature
    for a given value determined by the user.
    
    THIS CLASS WORKS WITH BOTH NUMERICAL AND CATEGORICAL VARIABLES
       
    Parameters
    ----------
    imputation_dictionary : dict, default=None
        dictionary mappting each variable to the number with which the
        missing values should be replaced
        
    Attributes
    ----------
    variables_ : list
        The list of variables that want to be imputed
        passed to the method:`fit`
        
    imputer_dict_: dict, entered by the user
        Dictionary mappting each variable to the number with which the
        missing values should be replaced
        
    """
    
    def __init__(self):
        pass

    def fit(self, X, y=None, imputation_dictionary = None):
        """ Learns the mean or medians that should be use to replace
        mising data in each variable.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking.
        variables: list of columns that should be transformed
        
        Returns
        -------
        self : object
            Returns self.
        """
        if not imputation_dictionary:
            raise ValueError('Dictionary with the values to replace NA for each variable should be provided')
            
        self.imputer_dict_ = imputation_dictionary           
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
        check_is_fitted(self, ['imputer_dict_'])
        
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set '
                             ' used to fit the encoder')
        
        X = X.copy()
        for feature in self.imputer_dict_.keys():
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X