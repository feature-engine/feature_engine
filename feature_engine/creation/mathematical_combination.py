from feature_engine.base_transformers import BaseNumericalTransformer


class MathematicalCombination(BaseNumericalTransformer):
    """
    MathematicalCombination() applies basic mathematical operations across features,
    returning 1 or more additional features as a result.

    For example, if we have the variables number_payments_first_quarter, number_payments_second_quarter,
    number_payments_third_quarter and number_payments_fourth_quarter, we can use MathematicalCombination()
    to calculate the total number of payments and mean number of payments as follows:

    .. code-block:: python

        transformer = MathematicalCombination(
            variables=[
                'number_payments_first_quarter',
                'number_payments_second_quarter',
                'number_payments_third_quarter',
                'number_payments_fourth_quarter'
            ],
            math_operations=[
                'sum',
                'mean'
            ],
            new_variables_name=[
                'total_number_payments',
                'mean_number_payments'
            ]
        )

        transformer.fit_transform(X)

    The transformed X will contain the additional features total_number_payments and mean_number_payments,
    plus the original set of variables.

    Parameters
    ----------

    variables: list, default=None
        The list of numerical variables to be combined. If None, the transformer
        will find and select all numerical variables.

    math_operations: list, default=None
        The list of basic math operations to be used in transformation.

        If none, all of ['sum', 'prod', 'mean', 'std', 'max', 'min'] will be performed
        over the variables. Alternatively, user can enter the list of operations to
        carry out.

        Each operation should be a string and must be one of the elements
        from the list: ['sum', 'prod', 'mean', 'std', 'max', 'min']

        Each operation will result in a new variable that will be added to the transformed dataset.

    new_variables_names: list, default=None
        Names of the newly created variables. The user can enter a name or a list
        of names for the newly created features (recommended). User must enter
        one name for each mathematical transformation indicated in the math_operations
        attribute. That is, if you want to perform mean and sum of features, you
        should enter 2 new variable names. If you perform only mean of features,
        enter 1 variable name. Alternatively, if you chose to perform all
        mathematical transformations, enter 6 new variable names.

        The name of the variables indicated by the user should coincide with the order
        in which the mathematical operations are initialised in the transformer.
        That is, if you set math_operations = ['mean', 'prod'], the first new variable name
        will be assigned to the mean of the variables and the second variable name
        to the product of the variables.

        If new_variable_names=None, the transformer will assign an arbitrary name
        to the newly created features starting by the name of the mathematical operation,
        followed by the variables combined separated by -.

    """

    def __init__(self, variables=None, math_operations=None, new_variables_names=None):

        if math_operations is None:
            math_operations = ["sum", "prod", "mean", "std", "max", "min"]

        self.variables = variables
        self.new_variables_names = new_variables_names
        self._math_operations_permitted = ["sum", "prod", "mean", "std", "max", "min"]

        if not isinstance(math_operations, list):
            raise KeyError("math_operations parameter must be a list or None")

        if any(
            operation not in self._math_operations_permitted
            for operation in math_operations
        ):
            raise KeyError(
                "At least one of math_operations is not found in permitted operations set. "
                "Choose one of ['sum', 'prod', 'mean', 'std', 'max', 'min']"
            )
        else:
            self.math_operations = math_operations

        if self.variables and len(self.variables) <= 1:
            raise KeyError(
                "MathematicalCombination requires two or more features to make proper transformations."
            )

        if self.new_variables_names and len(self.new_variables_names) != len(
            self.math_operations
        ):
            raise KeyError(
                "Number of items in New_variables_names must be equal to number of items in math_operations."
            )

    def fit(self, X, y=None):
        """
        Performs dataframe checks. Selects variables to transform if None were indicated by the user.
        Creates dictionary of column to transformation mappings

        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the variables to transform.

        y : None
            y is not needed in this transformer. You can pass y or None.
        """
        X = super().fit(X, y)

        self.input_shape_ = X.shape

        if self.new_variables_names:
            self.combination_dict_ = dict(
                zip(self.new_variables_names, self.math_operations)
            )
        else:
            self.combination_dict_ = {
                f"{operation}({'-'.join(self.variables)})": operation
                for operation in self.math_operations
            }

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

        for new_variable_name, operation in self.combination_dict_.items():
            X[new_variable_name] = X[self.variables].agg(operation, axis=1)

        return X
