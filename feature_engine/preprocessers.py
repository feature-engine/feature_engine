# Authors: João Nogueira <joaonogueira@fisica.ufc.br>
# License: BSD 3 clause

from sklearn.utils.validation import check_is_fitted
from feature_engine.base_transformers import BaseCategoricalEncoder, _define_variables


class MinMaxScaler(BaseCategoricalEncoder):
    """ Scales the numerical variables within the range 0-1.
    
    The scaler only works with numerical variables.
    
    Parameters
    ----------
    variables: list
        The list of numerical variables that will be transformed. If none, it 
        defaults to all numerical type variables.
        
    Attributes
    ----------
    """
    
    def __init__(self, variables = None):
        
        self.variables = variables


    def fit(self, X, y = None):
        """ Find the parameters to perform the min-max-scaler for each
        variable.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Should be the entire dataframe, not just seleted variables.
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking. You can either leave it as None
            or pass y.
        """
        
        #super().fit(X, y)

        self.encoder_dict_ = {}

        if self.variables != None:
            for var in self.variables:
                self.encoder_dict_[var] = {'min': X[var].min(), 'max': X[var].max()}
        else:
            for var in X.columns:
                self.encoder_dict_[var] = {'min': X[var].min(), 'max': X[var].max()}
            
        self.input_shape_ = X.shape
        
        return self
        
        
    def transform(self, X):
        """ Scales the variables. 
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        X_transformed : pandas dataframe. The shape of the dataframe will
        be the same of the original.
        """

        # Check is fit had been called
        check_is_fitted(self, ['encoder_dict_'])
            
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the encoder')
        
        X = X.copy()
        if self.variables != None:
            for feature in self.variables:
                featureRange = self.encoder_dict_[feature]['max'] - self.encoder_dict_[feature]['min'] 
                X[feature] = (X[feature] - self.encoder_dict_[feature]['min']) / featureRange
        else:
            for feature in X.columns:
                featureRange = self.encoder_dict_[feature]['max'] - self.encoder_dict_[feature]['min'] 
                X[feature] = (X[feature] - self.encoder_dict_[feature]['min']) / featureRange

        return X
            
        
        
        
        
        
        
        
        
        
        
        
