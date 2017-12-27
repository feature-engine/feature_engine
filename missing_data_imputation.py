# Authors: Soledad Galli <solegalli1@gmail.com>

# License: BSD 3 clause


import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class MeanMedianImputer(BaseEstimator, TransformerMixin):
    """ Transforms features by replacing missing data in each feature
    for a given mean or median value.
    
    This calss works only with numerical variables!!!
    
    This estimator first calculates the mean / median values for the
    desired features.
    
    The transformer fills the missing data
    
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
        self.imputation_method = imputation_method

    def fit(self, X, y=None, variables = None):
        """ Learns the mean or medians that should be use to replace
        mising data in each variable.
        
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
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
            # ADD AM ERROR HANDLER FOR NON NUMERICAL VARIABLES
            
            
        if self.imputation_method == 'mean':
            self.imputer_dict_ = X[self.variables_].mean().to_dict()
        elif self.imputation_method == 'median':
            self.imputer_dict_ = X[self.variables_].median().to_dict()
            
        return self

    def transform(self, X):
        """ Replaces missing data with the calculated parameters
        
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        X_transformed : array of int of shape = [n_samples, n_features]
            The dataframe containing no missing values for the selected
            variables
        """
        # Check is fit had been called
        check_is_fitted(self, ['imputer_dict_', 'variables_'])
        
        X = X.copy()
        for feature in self.variables_:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X


class RandomSampleImputer(BaseEstimator, TransformerMixin):
    """ Transforms features by replacing missing data in each feature
    with a random extracted sample from the training set.
    
    This class works for both numerical and categorical variables
    
    This estimator stores a copy of the training set
    
    The transformer fills the missing data with a random sample from
    the training set
    
    Parameters
    ----------
        
    Attributes
    ----------
    X_ : dataframe.
        Copy of the dataframe stored, from which to extract the
        random samples
        
    """
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """ Learns the mean or medians that should be use to replace
        mising data in each variable.
        
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        variables: list of columns that should be transformed
        
        Returns
        -------
        self : object
            Returns self.
        """
        self.X_ = X
        return self

    def transform(self, X, random_state):
        """ Replaces missing data with the calculated parameters
        
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        X_transformed : array of int of shape = [n_samples, n_features]
            The dataframe containing no missing values for the selected
            variables
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_'])

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


class EndTailImputer(BaseEstimator, TransformerMixin):
    """ Transforms features by replacing missing data in each feature
    for a given value at the tail of the distribution
    
    This class works only with numerical variables
    
    This estimator first calculates the values at the end of the distriution
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
    variables : list
        The list of variables that want to be imputed
        passed to the method:`fit`
        
    imputer_dict_: dictionary
        The dictionary containg the values at end of tails to use
        for each variable to replace missing data
        
    """
    
    def __init__(self, distribution='gaussian', tail='right', fold=3):
        self.distribution = distribution
        self.tail = tail
        self.fold = fold

    def fit(self, X, y=None, variables = None):
        """ Learns the values that should be use to replace
        mising data in each variable.
        
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
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
            # ADD AM ERROR HANDLER FOR NON NUMERICAL VARIABLES

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
                
            
        return self

    def transform(self, X):
        """ Replaces missing data with the calculated parameters
        
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        X_transformed : array of int of shape = [n_samples, n_features]
            The dataframe containing no missing values for the selected
            variables
        """
        # Check is fit had been called
        check_is_fitted(self, ['imputer_dict_', 'variables_'])

        X = X.copy()
        for feature in self.variables_:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X


class na_capturer(BaseEstimator, TransformerMixin):
    """ Transforms features by replacing missing data in each feature
    for a given value at the tail of the distribution
    
    This class works for both numerical and categorical variables
    
    This estimator first calculates the values at the end of the distriution
    for the desired features.
    
    The transformer fills the missing data
    
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
        self.tol = tol

    def fit(self, X, y=None):
        """ Learns the variables for which the additional variable 
        should be created
        
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        
        Returns
        -------
        self : object
            Returns self.
        """
        self.variables_ = [var for var in X.columns if X[var].isnull().mean()>self.tol]           
        return self

    def transform(self, X):
        """ Replaces missing data with the calculated parameters
        
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        X_transformed : array of int of shape = [n_samples, n_features]
            The dataframe containing no missing values for the selected
            variables
        """
        # Check is fit had been called
        check_is_fitted(self, ['variables_'])
        X = X.copy()
        for feature in self.variables_:
            X[feature+'_na'] = np.where(X[feature].isnull(),1,0)
        return X


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """ Transforms features by replacing missing data in each feature
    for a given value at the tail of the distribution
    
    This estimator first calculates the values at the end of the distriution
    for the desired features.
    
    The transformer fills the missing data
    
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
        self.tol = tol

    def fit(self, X, y=None):
        """ Learns the variables for which the additional variable 
        should be created
        
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        
        Returns
        -------
        self : object
            Returns self.
        """
        self.variables_ = [var for var in X.columns if X[var].dtypes == 'O']
        self.imputer_dict_ = {var:('Missing' if X[var].isnull().mean()>self.tol else X[var].mode()[0]) for var in self.variables_}
        return self

    def transform(self, X):
        """ Replaces missing data with the calculated parameters
        
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        X_transformed : array of int of shape = [n_samples, n_features]
            The dataframe containing no missing values for the selected
            variables
        """
        # Check is fit had been called
        check_is_fitted(self, ['variables_', 'imputer_dict_'])

        X = X.copy()
        for feature in self.variables_:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X
    
class ArbitraryImputer(BaseEstimator, TransformerMixin):
    """ Transforms features by replacing missing data in each feature
    for a given value determined by the user.
    
    This class works with both numerical and categorical variables
    
    This estimator first calculates the mean / median values for the
    desired features.
    
    The transformer fills the missing data
    
    Parameters
    ----------
    imputation_dictionary : dict, default=None
        Dictionary mappting each variable to the number with which the
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
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        variables: list of columns that should be transformed
        
        Returns
        -------
        self : object
            Returns self.
        """
        if not imputation_dictionary:
            # select all numerical variables
            print('PLease provide a dictionary with the values to replace NA for each variable')
            
        else:
            # variables indicated by user
            self.imputer_dict_ = imputation_dictionary           
        return self

    def transform(self, X):
        """ Replaces missing data with the calculated parameters
        
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        X_transformed : array of int of shape = [n_samples, n_features]
            The dataframe containing no missing values for the selected
            variables
        """
        # Check is fit had been called
        check_is_fitted(self, ['imputer_dict_'])
        
        X = X.copy()
        for feature in self.imputer_dict_.keys():
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X