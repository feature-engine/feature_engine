from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from feature_engine.variable_manipulation import _find_all_variables
from feature_engine.variable_manipulation import _find_categorical_variables
from feature_engine.variable_manipulation import _find_numerical_variables
from feature_engine.variable_manipulation import _define_variables
from feature_engine.dataframe_checks import _is_dataframe
from feature_engine import categorical_encoders as ce
from feature_engine import discretisers as dsc

import pandas as pd 

class TargetMeanEncoderFeatureSelector(BaseEstimator, TransformerMixin):
    """
    TargetMeanEncoderFeatureSelector
    ------------------
        Description
        -----------
        Calculates the feature importance.

        For each categorical variable:
            1) Separate into train and test
            2) Determine the mean value of the target within each label of the categorical variable using the train set
            3) Use that mean target value per label as the prediction (using the test set) and calculate the roc-auc.

        For each numerical variable:
            1) Separate into train and test
            2) Divide the variable into 100 quantiles
            3) Calculate the mean target within each quantile using the training set 
            4) Use that mean target value / bin as the prediction (using the test set) and calculate the roc-auc
        
        Implementation
        --------------

            Public methods
            --------------
                `fit(self, X, y)`
                `transform(self)`
                `fit_transform(self, X, y)`
        
    Parameters
    ----------

    variables: list, default=None
        The list of variables to evaluate. If None, the transformer will evaluate all 
        variables in the dataset associated with the variables_type.

    target: string, default=None
        The target variable to evaluate. If None, the transformer will return the 
        relevant message for the missing value.

    variables_type: boolean, default=True
        If variables_type = True: the transformer will evaluate the variables as the 
            categorical variables. 
        If variables_type = False: the transformer will evaluate the variables as the
            numerical variables.

    scoring: string, default='roc_auc_score'
        This indicates the metrics score to perform the feature selection.
        The current support includes 'roc_auc_score' and 'r2_score'.


    """

    def __init__(self, variables=None, scoring='roc_auc_score'):
        self.variables = _define_variables(variables)
        self.scoring = scoring
        self.X_test_enc = None
        self.y_test = None

    def fit(self, X, y, quantiles=5, test_size=0.3, random_state=0):
        """
        Performs
        --------

            1. Handling missing values and check invalid input variables.
            2. Split data into train and test.
            3. Performs the target mean encoding.
            4. Update the results X_train_enc and X_test_enc.
        
        Returns
        -------
            None

        Parameters
        ----------

            X: pandas dataframe of shape = [n_samples, n_features]
                The input dataframe.

            y: None
                y is not needed for this transformer. You can pass y or None.
            
            test_size: float, default=0.3
                The test size setting of the data in the train_test_split method.

            random_state: int, default=0
                The random state setting in the train_test_split method.

            quantiles: int, default=5
                The amount of quantiles setting in the qcut of the fit method in case of the numerical types.
                The range of the quantiles values is in between 3 and 100. The transformer will return a message
                if the input quantiles value is out of this range.

        Attributes
        ----------

            X_test_enc: pandas dataframe of shape = [n_samples, n_features]
                The output of target mean encoding of the test set.
        
        """

        # check the target
        if y is None:
            raise ValueError('Please provide a target y for this encoding method.')

        # check input dataframe
        X = _is_dataframe(X)

        # find all variables or check those entered are in the dataframe
        self.variables = _find_all_variables(X, self.variables)
        
        # find categorical variables or check that those vars entered by the user
        # are of type object
        cat_variables = _find_categorical_variables(X[self.variables] if self.variables else X)
        num_variables = _find_numerical_variables(X[self.variables] if self.variables else X)

        data = pd.concat([X, y], axis=1) # column appending.
        data.columns = list(X.columns) + ['target']

        # split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
                                                data[cat_variables + num_variables + ['target']],
                                                data['target'],
                                                test_size = test_size,
                                                random_state = random_state)

        ## filling missing values.
        for col in cat_variables:
            X_train[col].fillna('M', inplace=True)
            X_test[col].fillna('M', inplace=True)

        for col in num_variables:
            X_train[col].fillna(0, inplace=True)
            X_test[col].fillna(0, inplace=True)

        # categorical variables
        if cat_variables: 
            # create the mean encoder instance 
            encoder = ce.MeanCategoricalEncoder()
            # fit
            encoder.fit(X_train[cat_variables], y_train)
            # transform
            X_train_cat_enc, X_test_cat_enc = encoder.transform(X_train[cat_variables]), encoder.transform(X_test[cat_variables])

            # filling missing values on the results
            X_train_cat_enc, X_test_cat_enc = X_train_cat_enc.fillna(0), X_test_cat_enc.fillna(0)

        # numerical variables
        self.numerical_binned_ = dict()
        if num_variables: 
            # Equal-frequency discretizer
            disc = dsc.EqualFrequencyDiscretiser(q=quantiles, variables=num_variables, return_object=True)
            # fit train data
            disc.fit(X_train[num_variables])
            # transform the data
            X_train_t = disc.transform(X_train[num_variables])
            X_test_t = disc.transform(X_test[num_variables])

            # create the mean encoder instance 
            encoder = ce.MeanCategoricalEncoder()
            # mean encoding
            encoder.fit(X_train_t[num_variables], y_train)
            X_train_num_enc, X_test_num_enc = encoder.transform(X_train_t[num_variables]), encoder.transform(X_test_t[num_variables])
            X_train_num_enc.columns = [i+'_binned' for i in X_train_num_enc.columns]
            X_test_num_enc.columns = [i+'_binned' for i in X_test_num_enc.columns]

            self.numerical_binned_ = disc.binner_dict_
            # filling missing values on the results
            X_train_num_enc, X_test_num_enc = X_train_num_enc.fillna(0), X_test_num_enc.fillna(0)


        # results
        if num_variables and cat_variables:
            X_test_enc = pd.concat([X_test_cat_enc, X_test_num_enc], axis=1)
        elif num_variables and not cat_variables:
            X_test_enc = X_test_num_enc
        elif cat_variables and not num_variables:
            X_test_enc = X_test_cat_enc
        else:
            raise ValueError("No variables found!")
        
        self.output_shape_ = X_test_enc.shape

        self.X_test_enc = X_test_enc
        self.y_test = y_test

        return self

    def transform(self):
        """
        Performs
        --------
            Calculates the metrics score from encodings.
            
        Returns
        -------
        output: Pandas DataFrame, default=None
            The sorted list of features importance in the descending order. 

        Parameters
        ----------
            None


        Attributes
        ----------
            scoring: string, default='roc_auc_score'
                The metrics score to evaluate the feature selection.

        """
        check_is_fitted(self)
        
        results = []
        if self.scoring == 'roc_auc_score':
            for col in self.X_test_enc.columns:
                results.append(roc_auc_score(self.y_test, self.X_test_enc[col]))
        elif self.scoring == 'r2_score':
            for col in self.X_test_enc.columns:
                results.append(r2_score(self.y_test, self.X_test_enc[col]))  
        else: # default
            for col in self.X_test_enc.columns:
                results.append(roc_auc_score(self.y_test, self.X_test_enc[col]))

        # convert to dataframe
        output_ = pd.Series(results)
        output_.index = self.X_test_enc.columns
        output = pd.DataFrame(output_).sort_values(by=0, ascending=False)
        output.columns = ['Importance']
        output.index.name = f'{self.scoring}'
        output = output.reset_index()

        return output

    def fit_transform(self, X, y, quantiles=5, test_size=0.3, random_state=0):
        """
        Performs
        --------
            fit and transform.
            
        Returns
        -------
        output: Pandas DataFrame, default=None
            The sorted list of features importance in the descending order. 

        Parameters
        ----------
            X: pandas dataframe of shape = [n_samples, n_features]
            The input dataframe.

            y: None
                y is not needed for this transformer. You can pass y or None.


        Attributes
        ----------
            None

        """
        self.fit(X, y, quantiles=quantiles, test_size=test_size, random_state=random_state)
        return self.transform(self.X_test_enc, self.y_test)
