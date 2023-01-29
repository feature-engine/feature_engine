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
