# Authors: Soledad Galli <solegalli1@gmail.com>
# License: BSD 3 clause

import pandas as pd
import numpy as np
#import warnings

#from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from feature_engine.base_transformers import BaseCategoricalEncoder, _define_variables



class CountFrequencyCategoricalEncoder(BaseCategoricalEncoder):
    """ Replaces categories by the count of observations for that category or by
    the percentage of observations for that category.
    
    For example in the variable colour, if 10 observations are blue, blue will
    be replaced by 10. Alternatively, if 10% of the observations are blue, blue
    will be replaced by 0.1.
    
    The Encoder will encode only categorical variables (type 'object'). A list 
    of variables can be passed as an argument. If no variables are passed as 
    argument, the encoder will only encode categorical variables (object type)
    and ignore the rest.
    
    The encoder first maps the categories to the numbers for each variable (fit).
    The transformer replaces the categories by numbers (transform).
    
    Parameters
    ----------
    encoding_method : str, default='count'
        Desired method of encoding.
        'count': number of observations per category
        'frequency' : percentage of observations per category
    
    variables : list
        The list of categorical variables that will be encoded. If none, it 
        defaults to all object type variables.
        
    Attributes
    ----------
    encoder_dict_: dictionary
        The dictionary containing the count / frequency: category pairs used
        to replace categories for every variable    
    """
    
    def __init__(self, encoding_method = 'count', variables = None):
        
        if encoding_method not in ['count', 'frequency']:
            raise ValueError("encoding_method takes only values 'count' and 'frequency'")
            
        self.encoding_method = encoding_method       
        self.variables = _define_variables(variables)
           

    def fit(self, X, y = None):
        """ Learns the numbers that should be used to replace the categories in
        each variable.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Should be the entire dataframe, not just seleted variables.
        y : None
            y is not needed in this encoder, yet the sklearn pipeline API
            requires this parameter for checking. You can either leave it as None
            or pass y.
        """
        # brings the variables from the BaseEncoder
        super().fit(X, y)
        
        self.encoder_dict_ = {}
        
        for var in self.variables:
            if self.encoding_method == 'count':
                self.encoder_dict_[var] = X[var].value_counts().to_dict()
                
            elif self.encoding_method == 'frequency':
                n_obs = np.float(len(X))
                self.encoder_dict_[var] = (X[var].value_counts() / n_obs).to_dict()   
        
        if len(self.encoder_dict_)==0:
            raise ValueError('Encoder could not be fitted. Check that correct parameters and dataframe were passed during training')
            
        self.input_shape_ = X.shape
        
        return self



class OrdinalCategoricalEncoder(BaseCategoricalEncoder):
    """ Replaces categories by ordinal numbers (0, 1, 2, 3, etc). The numbers
    can be ordered based on the mean of the target per category, or assigned 
    arbitrarily.
    
    For the ordered ordinal encoding for example, in the variable colour, if the
    mean of the target for blue, red and grey is 0.5, 0.8 and 0.1 respectively,
    blue is replaced by 1, red by 2 and grey by 0.
    
    For the arbitrary ordinal encoding the numbers will be assigned arbitrarily
    to the categories.
    
    The Encoder will encode only categorical variables (type 'object'). A list 
    of variables can be passed as an argument. If no variables are passed as 
    argument, the encoder will only encode categorical variables (object type)
    and ignore the rest.
    
    The encoder first maps the categories to the numbers for each variable (fit).
    The transformer replaces the categories by numbers (transform).
    
    Parameters
    ----------
    encoding_method : str, default='ordered' 
        Desired method of encoding.
        'ordered': the categories are numbered in ascending order according to
        the target mean per category.
        'arbitrary' : categories are numbered arbitrarily.
        
    variables : list
        The list of categorical variables that will be encoded. If none, it 
        defaults to all object type variables.
        
    Attributes
    ----------
    encoder_dict_: dictionary
        The dictionary containing the ordinal number: category pairs used
        to replace categories for every variable
        
    """    
    def __init__(self, encoding_method  = 'ordered', variables = None):
        
        if encoding_method not in ['ordered', 'arbitrary']:
            raise ValueError("encoding_method takes only values 'ordered' and 'arbitrary'")
            
        self.encoding_method = encoding_method
        self.variables = _define_variables(variables)
           

    def fit(self, X, y=None):
        """ Learns the numbers that should be used to replace the labels in each
        variable.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Should be the entire dataframe, not just seleted variables.
        y : Target. Can be None if selecting encoding_method = 'arbitrary'
       
        """
        # brings the variables from the BaseEncoder
        super().fit(X, y)
        
        if self.encoding_method == 'ordered':
            if y is None:
                raise ValueError('Please provide a target (y) for this encoding method')
                            
            temp = pd.concat([X, y], axis=1)
            temp.columns = list(X.columns)+['target']

        
        self.encoder_dict_ = {}
        
        for var in self.variables:
            
            if self.encoding_method == 'ordered':
                t = temp.groupby([var])['target'].mean().sort_values(ascending=True).index
                
            elif self.encoding_method == 'arbitrary':
                t = X[var].unique()
                
            self.encoder_dict_[var] = {k:i for i, k in enumerate(t, 0)}
            
        if len(self.encoder_dict_)==0:
            raise ValueError('Encoder could not be fitted. Check that correct parameters and dataframe were passed during training')
            
        self.input_shape_ = X.shape
        
        return self



class MeanCategoricalEncoder(BaseCategoricalEncoder):
    """ Replaces categories by the mean of the target. 
    
    For example in the variable colour, if the mean of the target for blue, red
    and grey is 0.5, 0.8 and 0.1 respectively, blue is replaced by 0.5, red by 0.8
    and grey by 0.1.
    
    The Encoder will encode only categorical variables (type 'object'). A list 
    of variables can be passed as an argument. If no variables are passed as 
    argument, the encoder will only encode categorical variables (object type)
    and ignore the rest.
    
    The encoder first maps the categories to the numbers for each variable (fit).
    The transformer replaces the categories by numbers (transform).
    
    Parameters
    ----------  
    variables : list
        The list of categorical variables that will be encoded. If none, it 
        defaults to all object type variables.
        
    Attributes
    ----------
    encoder_dict_: dictionary
        The dictionary containing the ordinal number: category pairs used
        to replace categories for every variable
        
    """    
    def __init__(self, variables = None):
        
        self.variables = _define_variables(variables)
           

    def fit(self, X, y):
        """ Learns the numbers that should be used to replace
        the labels in each variable.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Should be the entire dataframe, not just seleted variables.
        y : Target
       
        """
        # brings the variables from the BaseEncoder
        super().fit(X, y)

        if y is None:
            raise ValueError('Please provide a target (y) for this encoding method')
            
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns)+['target']
        
        self.encoder_dict_ = {}
        
        for var in self.variables:
            self.encoder_dict_[var] = temp.groupby(var)['target'].mean().to_dict()

        if len(self.encoder_dict_)==0:
            raise ValueError('Encoder could not be fitted. Check that correct parameters and dataframe were passed during training')
            
        self.input_shape_ = X.shape
            
        return self
    

    
class WoERatioCategoricalEncoder(BaseCategoricalEncoder):
    """ Replaces categories by the weight of evidence or by the ratio between
    the probability of the target = 1 and the probability of the  target = 0.
    
    The weight of evidence is given by: np.log( p(1) / p(0) )
    
    The target probability ratio is given by: p(1) / p(0)
    
    Note: This categorical encoder is exclusive for binary classification.
    
    For example in the variable colour, if the mean of the target = 1 for blue
    is 0.8 and the mean of the target = 0  is 0.2, blue will be replaced by:
    np.log(0.8/0.2) = 1.386 if woe is selected. Alternatively, blue will be 
    replaced by 0.8 / 0.2 = 4.
    
    Note as well that the division by 0 is not defined, as well as the log(0).
    Thus, if p(0) = 0 for the ratio encoder, or either p(0) = 0 or p(1) = 0 for
    woe, in any of the variables, the encoder will return an error.
       
    The Encoder will encode only categorical variables (type 'object'). A list 
    of variables can be passed as an argument. If no variables are passed as 
    argument, the encoder will only encode categorical variables (object type)
    and ignore the rest.
    
    The encoder first maps the categories to the numbers for each variable (fit).
    The transformer replaces the categories by numbers (transform).
    
    Parameters
    ----------
    encoding_method : str, default=woe
        Desired method of encoding.
        'woe': weight of evidence
        'ratio' : probability ratio
    variables : list
        The list of categorical variables that will be encoded. If none, it 
        defaults to all object type variables.
        
    Attributes
    ----------
    encoder_dict_: dictionary
        The dictionary containing the ordinal number: category pairs used
        to replace categories for every variable
        
    """    
    def __init__(self, encoding_method = 'woe', variables = None):

        if encoding_method not in ['woe', 'ratio']:
            raise ValueError("encoding_method takes only values 'woe' and 'ratio'")
            
        self.encoding_method = encoding_method
        self.variables = _define_variables(variables)
           

    def fit(self, X, y):
        """ Learns the numbers that should be used to replace the labels in each
        variable.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Should be the entire dataframe, not just seleted variables.
        y : Target
       
        """
        
        # brings the variables from the BaseEncoder
        super().fit(X, y)

        if y is None:
            raise ValueError('Please provide a target (y) for this encoding method')
        
        # check that y is binary
        if len( [x for x in y.unique() if x not in [0,1] ] ) > 0:
            raise ValueError("This encoder is only designed for binary classification, values of y can be only 0 or 1")
        
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns)+['target']
        
        self.encoder_dict_ = {}  
        for var in self.variables:
            t = temp.groupby(var)['target'].mean()
            t = pd.concat([t, 1-t], axis=1)
            t.columns = ['p1', 'p0']
            
            if self.encoding_method == 'woe':
                if not t.loc[t['p0'] == 0, :].empty or not t.loc[t['p1'] == 0, :].empty:
                    raise ValueError("p(0) or p(1) for a category in variable {} is zero, log of zero is not defined".format(var))
                else:
                    self.encoder_dict_[var] = (np.log(t.p1/t.p0)).to_dict()
                
            elif self.encoding_method == 'ratio':
                if not t.loc[t['p0'] == 0, :].empty:
                    raise ValueError("p(0) or p(1) for a category in variable {} is zero, division by is not defined".format(var))
                else:
                    self.encoder_dict_[var] = (t.p1/t.p0).to_dict()  

        if len(self.encoder_dict_)==0:
            raise ValueError('Encoder could not be fitted. Check that correct parameters and dataframe were passed during training')
            
        self.input_shape_ = X.shape
               
        return self



class OneHotCategoricalEncoder(BaseCategoricalEncoder):
    """ One hot encoding consists of replacing the categorical variable by a
    combination of boolean variables which take value 0 or 1, to indicate if
    a certain category is present for that observation.
    
    Each one of the boolean variables are also known as dummy variables or binary
    variables. For example, from the categorical variable "Gender" with categories
    'female' and 'male', we can generate the boolean variable "female", which 
    takes 1 if the person is female or 0 otherwise. We can also generate the 
    variable male, which takes 1 if the person is "male" and 0 otherwise.
    
    The encoder has the option to generate one dummy variable per category present
    in a variable, or to create dummy variables only for the top n categories
    that are present in the majority of the observations.
    
    If dummy variables are created for all the categories of a variable, you have
    the option to drop one category not to create information redundancy.
    
    The Encoder will encode only categorical variables (type 'object'). A list 
    of variables can be passed as an argument. If no variables are passed as 
    argument, the encoder will only encode categorical variables (object type)
    and ignore the rest.
    
    The encoder first finds the categories to be encoded for each variable (fit).
    The transformer creates one dummy variable per category for each variable
    (transform).
    
    Parameters
    ----------
    top_categories: int, default = None
        If None is selected, a dummy variable will be created for each category
        per variable. Alternatively, the encoder will find the most frequent
        categories: top_categories indicates the number of most frequent labels
        to encode. Dummy variables will be created only for those categories and
        the rest will be ignored. Note that this is equivalent to grouping all the
        remaining categories in one group.
    variables : list
        The list of categorical variables that will be encoded. If none, it 
        defaults to all object type variables.
    drop_last: boolean, default = False
        Only used if top_categories = None. It indicates whether to create dummy
        variables for all the available categories, or if set to true, it will
        ignore the last variable of the list.
        
    Attributes
    ----------
    encoder_dict_: dictionary
        The dictionary containg the frequent categories (that will be kept)
        for each variable. Categories not present in this list will be replaced
        by 'Rare'. 
        
    """  
    def __init__(self, top_categories = None, variables = None, drop_last = False):

        if top_categories:
            if not isinstance(top_categories, int):
                raise ValueError("top_categories takes only integer numbers, 1, 2, 3, etc.")            
        self.top_categories = top_categories
        
        if drop_last not in [True, False]:
            raise ValueError("drop_last takes only True or False")
            
        self.drop_last = drop_last
        self.variables = _define_variables(variables)

    
    def fit(self, X, y=None):
        """ Learns the numbers that should be used to replace the labels in each
        variable.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Should be the entire dataframe, not just seleted variables.
        y : Target
       
        """
        
        # brings the variables from the BaseEncoder
        super().fit(X, y)
        
        self.encoder_dict_ = {}
        
        for var in self.variables:
            if not self.top_categories:
                if self.drop_last:
                    category_ls = [x for x in X[var].unique() ]
                    self.encoder_dict_[var] = category_ls[:-1]
                else:
                    self.encoder_dict_[var] = X[var].unique()
                
            else:
                self.encoder_dict_[var] = [x for x in X[var].value_counts().sort_values(ascending=False).head(self.top_categories).index]

        if len(self.encoder_dict_)==0:
            raise ValueError('Encoder could not be fitted. Check that correct parameters and dataframe were passed during training')
            
        self.input_shape_ = X.shape
               
        return self


    def transform(self, X):
        """ Creates the dummy / boolean variables.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        X_transformed : pandas dataframe. The shape of the dataframe will
        be different from the original as it includes the dummy variables.
        """
        
        # Check is fit had been called
        check_is_fitted(self, ['encoder_dict_'])
            
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the encoder')
        
        X = X.copy()
        for feature in self.variables:
            for category in self.encoder_dict_[feature]:
                X[str(feature) + '_' + str(category)] = np.where(X[feature] == category, 1, 0)
            
        # drop the original non-encoded variables.
        X.drop(labels=self.variables, axis=1, inplace=True)
        
        return X
    
    

class RareLabelCategoricalEncoder(BaseCategoricalEncoder):
    """ Groups rare / infrequent categories under the new category "Rare"
    
    For example in the variable colour, if the percentage of observations
    for the categories magenta, cyan and burgundy are < 5 %, all those
    categories will be replaced by the new label "Rare". The encoder then 
    creates one new category that groups all the above mentioned
    categories.
       
    The Encoder will encode only categorical variables (type 'object'). A list 
    of variables can be passed as an argument. If no variables are passed as 
    argument, the encoder will only encode categorical variables (object type)
    and ignore the rest.
    
    The encoder first finds the frequent labels for each variable (fit).
    The transformer groups the infrequent labels under the new label 'Rare'
    (transform).
    
    Parameters
    ----------
    tol: float, default = 0.05
        the minimum frequency a label should have to be considered frequent
        and not be removed.
    n_categories: int, default = 10
        the minimum number of categories a variable should have in order for 
        the encoder to work  to find frequent labels. If the variable contains 
        less categories, all of them will be considered frequent.
    variables : list
        The list of categorical variables that will be encoded. If none, it 
        defaults to all object type variables.
        
    Attributes
    ----------
    encoder_dict_: dictionary
        The dictionary containg the frequent categories (that will be kept)
        for each variable. Categories not present in this list will be replaced
        by 'Rare'. 
        
    """
  
    def __init__(self, tol = 0.05, n_categories = 10, variables = None):
        
        if tol <0 or tol >1 :
            raise ValueError("tol takes values between 0 and 1")
            
        if n_categories < 0 or not isinstance(n_categories, int):
            raise ValueError("n_categories takes only positive integer numbers")
            
        self.tol = tol
        self.n_categories = n_categories
        self.variables = _define_variables(variables)
        

    def fit(self, X, y = None):
        """ Learns the frequent categories for each variable.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Should be the entire dataframe, not just seleted variables
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter. You can leave y as None, or pass it as an
            argument.
        """
        # brings the variables from the BaseEncoder
        super().fit(X, y)
        
        self.encoder_dict_ = {}
        
        for var in self.variables:
            if len(X[var].unique()) > self.n_categories:
                # if the variable has more than the indicated number of categories
                # the encoder will learn the most frequent categories
                t = pd.Series(X[var].value_counts() / np.float(len(X)))
                # non-rare labels:
                self.encoder_dict_[var] = t[t>=self.tol].index
            else:
                # if the total number of categories is smaller than the indicated
                # the encoder will consider all categories as frequent.
                self.encoder_dict_[var]=  X[var].unique()
        
        self.input_shape_ = X.shape
                   
        return self
    

    def transform(self, X):
        """ Groups rare labels under separate group 'Rare'.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe where rare categories have been grouped.
        """
        
        # Check is fit had been called
        check_is_fitted(self, ['encoder_dict_'])
            
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the encoder')
        
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(self.encoder_dict_[feature]), X[feature], 'Rare')
            
        return X