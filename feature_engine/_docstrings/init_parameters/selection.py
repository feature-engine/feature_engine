_confirm_variables_docstring = """confirm_variables: bool, default=False
            If set to True, variables that are not present in the input dataframe will
            be removed from the list of variables. Only used when passing a variable
            list to the parameter `variables`. See parameter variables for more details.
        """.rstrip()

_estimator_docstring = """estimator: object
            A Scikit-learn estimator for regression or classification.
            The estimator must have either a `feature_importances` or a `coef_`
            attribute after fitting.
    """.rstrip()

_order_by_docstring = """order_by: str, default=None
            How to sort the variables in the dataframe before feature selection. This
            helps to obtain consistent results.\n
            - None - preserves original variable order in the dataframe.\n
            - 'nan' - sorts columns by number of missing values (ascending).\n
            - 'unique' - sorts columns by number of unique values (descending).\n
            - 'alphabetic' - sorts columns alphabetically.\n
    """.rstrip()
