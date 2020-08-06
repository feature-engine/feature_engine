from feature_engine.base_transformers import BaseNumericalTransformer


class AdditionTransformer(BaseNumericalTransformer):
    """
    The AdditionTransfomer() applies basic mathematical operations using aggregation of features.

    Parameters
    ----------

    variables: list, default=None
        The list of numerical variables to be transformed. If None, the transformer
        will find and select all numerical variables.

    math_operations: list, default=None
        The list of basic math operations to be used in transformation.

        Each operation should be a string and must be one of elements
        from list: ['sum', 'prod', 'mean', 'std', 'max', 'min']

        Each operation will result in operation column in result dataset.

        If None, the transformer will calculate each of permitted
        operations: ['sum', 'prod', 'mean', 'std', 'max', 'min']
    """

    def __init__(self, variables=None, math_operations=None):
        self.variables = variables
        self.math_operations = math_operations
        self.math_operations_permitted = ['sum', 'prod', 'mean', 'std', 'max', 'min']

        if self.math_operations is None:
            self.operations_ = self.math_operations_permitted

        elif isinstance(self.math_operations, list):
            if any(elem_par not in self.math_operations_permitted for elem_par in self.math_operations):
                raise KeyError("At least one of math_operations is not found in permitted operations set. "
                               "Choose one of ['sum', 'prod', 'mean', 'std', 'max', 'min']")
            self.operations_ = self.math_operations
        else:
            raise KeyError("math_operations parameter must be a list or None")

    def fit(self, X, y=None):
        """
        Fits source dataset, verifies if variables parameter contains more than one variable.

        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the variables to transform.

        y : None
            y is not needed in this transformer. You can pass y or None.
        """
        X = super().fit(X, y)
        self.input_shape_ = X.shape

        if len(self.variables) <= 1:
            raise KeyError("AdditionTransformer requires two or more features to make proper transformations.")

        return self

    def transform(self, X):
        """
        Transforms source dataset.

        Adds column for each operation with calculation based on variables and operation.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------

        X_transformed : pandas dataframe of shape = [n_samples, n_features + n_operations]
            The dataframe with operations results added.
        """
        X = super().transform(X)

        for operation in self.operations_:
            variables_set_name = f"{operation}({','.join(self.variables)})"
            X[variables_set_name] = X[self.variables].agg(operation, axis=1)

        return X
