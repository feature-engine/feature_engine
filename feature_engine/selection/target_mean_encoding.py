from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score

from feature_engine.variable_manipulation import _find_all_variables, _define_variables
from feature_engine.dataframe_checks import _is_dataframe

import pandas as pd 

class TargetMeanEncoding(BaseEstimator, TransformerMixin):
    """
    TargetMeanEncoding
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

            Private methods
            ---------------
                `__target_mean_encoding(self, df_train_, df_test_, var_cols=None)`
                `__validate_variables(self, X)`
                    
            Public methods
            --------------
                `fit(self, X, y=None)`
                `transform(self)`
                `fit_transform(self, X, y=None)`
        
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

    metrics_score: string, default='roc_auc_score'
        This indicates the metrics score to perform the feature selection.
        The current support includes 'roc_auc_score' and 'r2_score'.

    test_size: float, default=0.3
        The test size setting of the data in the train_test_split method.

    random_state: int, default=0
        The random state setting in the train_test_split method.

    quantiles: int, default=5
        The amount of quantiles setting in the qcut of the fit method in case of the numerical types.
        The range of the quantiles values is in between 3 and 100. The transformer will return a message
        if the input quantiles value is out of this range.

    """

    def __init__(self, variables=None, target=None, variables_type=True, metrics_score='roc_auc_score', test_size=0.3, random_state=0, quantiles=5):
        # inputs
        self.variables = _define_variables(variables)
        self.vars_type = variables_type
        self.metrics_score = metrics_score
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.quantiles = quantiles

        # transformed variables
        self.target_enc_dict = {}
        self.transformed_cols = []
        self.binned_vars = []
        
        # dataframes
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.X_train_enc, self.X_test_enc = None, None
        
        # output
        self.output = None


    def __target_mean_encoding(self, df_train_, df_test_, var_cols=None):
        """
        Performs
        --------
            For each variable:
                + Performs the target mean encoding on the train set. 
                + Applies the result to both train and test sets.
            Drop the target variable from the datasets.
            Return the updated target mean encoding datasets.

        Returns
        -------
        df_train: pandas dataframe of shape = [n_samples, n_features]
            The output train set dataframe.

        df_test: pandas dataframe of shape = [n_samples, n_features]
            The output test set dataframe.

        Parameters
        ----------
        df_train_: pandas dataframe of shape = [n_samples, n_features]
            The input train set dataframe.

        df_test_: pandas dataframe of shape = [n_samples, n_features]
            The input test set dataframe.

        var_cols: list, default=None
            The list of variables that is needed to be transformed 
            the target mean encoding.


        Attributes
        ----------
        __target_mean_enc: dict, default=None
            The output of group of variables by target mean of the train set.

        target_enc_dict: dict, default=None
            The dictionary that stores the result of target encoding of 
            each variable in the train set.

        """
        df_train, df_test = df_train_.copy(), df_test_.copy()
        for col in var_cols:
            # get the target encoding on train set.
            __target_mean_enc = df_train_.groupby([col])[self.target].mean().to_dict()
            # update this target encoding result to both train and test sets.
            df_train[col] = df_train_[col].map(__target_mean_enc)
            df_test[col] = df_test_[col].map(__target_mean_enc)
            self.target_enc_dict[col] = __target_mean_enc

        # drop target from the updated datasets
        df_train.drop([self.target], axis=1, inplace=True)
        df_test.drop([self.target], axis=1, inplace=True)

        return df_train, df_test 

    def __validate_variables(self, X):
        """
        Performs
        --------
            Check the target is valid. The transformer will return the appropriate message 
                if the target is not found in the data or None.

            Check if the list of variables is not found. The transformer will use all the 
                columns in variables_type (categorical/numerical) if the variables list is empty.

            Filling missing values in columns. The transformer will fill the missing values
                in the columns.

            Check valid variables in data, and either categorical or numerical types.
                The transformer will return the appropriate message if the variables 
                are misclassified as categorical or numerical types.

        Returns
        -------
            None

        Parameters
        ----------
            X: pandas dataframe of shape = [n_samples, n_features]
            The input dataframe.

        Attributes
        ----------
            None

        """
        # check quantiles value
        if self.quantiles > 100 or self.quantiles < 3:
            raise ValueError(
                "Invalid quantiles. The quantiles value should be in between 3 and 100."
                " If there is no pre-definition on its value, the method will use the value of 5 by default."
            )   

        # check if the target is valid 
        if self.target not in X.columns:
            raise ValueError(
                "Invalid target variable!"
            )

        # extract categorical and numerical variables from data columns.
        numerical_cols = list(set(X.select_dtypes("number", "bool_").columns))
        categorical_cols = list(set(X.select_dtypes("object").columns))

        # find all variables or check those entered are in the dataframe
        if self.variables:
            self.variables = _find_all_variables(X, self.variables)
        else: # None/Empty in variables list.
            if self.vars_type:
                self.variables = categorical_cols
            else:
                self.variables = numerical_cols

        # filling missing values
        if self.vars_type: ## categorical
            for col in self.variables + [self.target]:
                X[col] = X[col].fillna('Missing')
        else: ## numerical
            for col in self.variables + [self.target]:
                X[col] = X[col].fillna(0)

        # check if one or more variables of list is/are not in the correct vars_type.
        if self.vars_type: # categorical
            _missclass_var_list = list(set(self.variables) - set(categorical_cols))
            # check if a numerical variable is found
            if len(_missclass_var_list) > 0:
                raise ValueError(
                f"Invalid categorical variables! {_missclass_var_list} is/are the numerical type."
            )

        else: # numerical
            _missclass_var_list = list(set(self.variables) - set(numerical_cols))
            # check if a categorical variable is found
            if len(_missclass_var_list) > 0:
                raise ValueError(
                f"Invalid numerical variables! {_missclass_var_list} is/are the categorical type."
            )
                
        return self

    def fit(self, X, y=None):
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


        Attributes
        ----------

        X_train_enc: pandas dataframe of shape = [n_samples, n_features]
            The output of target mean encoding of the train set.

        X_test_enc: pandas dataframe of shape = [n_samples, n_features]
            The output of target mean encoding of the test set.
        
        """
        # check input dataframe
        X = _is_dataframe(X)

        # check invalidate variables
        self.__validate_variables(X)

        # split data
        X_train, X_test, y_train, y_test = train_test_split(
                                                X[self.variables + [self.target]],
                                                X[self.target],
                                                test_size = self.test_size,
                                                random_state = self.random_state)

        # additional step for numerical variables, divide the variables into a number of quantiles/bins.
        if not self.vars_type: ## numerical type
            for feature in self.variables:
                # use pandas.qcut on the train set.
                X_train[f'{feature}_binned'], intervals = pd.qcut(X_train[feature],
                                                                    q = self.quantiles,
                                                                    labels = False,
                                                                    retbins = True, # return intervals
                                                                    precision = 3,
                                                                    duplicates = 'drop')

                # use pandas.cut on the test set with the intervals result from qcut of the train set.
                X_test[f'{feature}_binned'] = pd.cut(x = X_test[feature],
                                                        bins = intervals,
                                                        labels = False)

        # calculate the mean target encoding
        if self.vars_type: # categorical
            X_train_enc, X_test_enc = self.__target_mean_encoding(X_train[self.variables + [self.target]], 
                                                                  X_test[self.variables + [self.target]], 
                                                                  var_cols=self.variables)
        else: # numerical
            self.binned_vars = [i for i in X_train.columns if '_binned' in i]
            X_train_enc, X_test_enc = self.__target_mean_encoding(X_train[self.binned_vars + [self.target]], 
                                                                  X_test[self.binned_vars + [self.target]], 
                                                                  var_cols=self.binned_vars)
        # filling missing values on the results
        X_train_enc, X_test_enc = X_train_enc.fillna(0), X_test_enc.fillna(0)

        # store results
        self.transformed_cols = self.variables if self.vars_type else self.binned_vars 
        self.X_train, self.X_test = X_train, X_test
        self.X_train_enc, self.X_test_enc = X_train_enc, X_test_enc
        self.y_train, self.y_test = y_train, y_test

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
            metrics_score: string, default='roc_auc_score'
                The metrics score to evaluate the feature selection.

            transformed_cols: list, default=None
                The list of variables to evaluate the feature selection.

        """
        check_is_fitted(self)
        results = []
        if self.metrics_score == 'roc_auc_score':
            for col in self.transformed_cols:
                results.append(roc_auc_score(self.y_test, self.X_test_enc[col]))
        elif self.metrics_score == 'r2_score':
            for col in self.transformed_cols:
                results.append(r2_score(self.y_test, self.X_test_enc[col]))  
        else: # default
            for col in self.transformed_cols:
                results.append(roc_auc_score(self.y_test, self.X_test_enc[col]))

        # convert to dataframe
        output_ = pd.Series(results)
        output_.index = self.transformed_cols
        output = pd.DataFrame(output_).sort_values(by=0, ascending=False)
        output.columns = ['Importance']
        output.index.name = f'{self.metrics_score}'
        output = output.reset_index()

        self.output = output 
        return self.output

    def fit_transform(self, X, y=None):
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
        self.fit(X)
        return self.transform()
