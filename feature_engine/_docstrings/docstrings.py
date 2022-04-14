

# input parameters
_variables_numerical_docstring = """variables: list, default=None
        The list of numerical variables to transform. If None, the transformer will
        automatically find and select all numerical variables.
    """.rstrip()

_drop_original_docstring = """drop_original: bool, default=False
        If True, the original variables to transform will be dropped from the dataframe.
    """.rstrip()

_missing_values_docstring = """missing_values: string, default='raise'
        Indicates if missing values should be ignored or raised. If 'raise' the
        transformer will return an error if the the datasets to `fit` or `transform`
        contain missing values. If 'ignore', missing data will be ignored when learning
        parameters or performing the transformation.
        """

# Attributes
_variables_attribute_docstring = """variables_:
        The group of variables that will be transformed.
        """.rstrip()

_feature_names_in_docstring = """feature_names_in_:
        List with the names of features seen during `fit`.
        """.rstrip()

_n_features_in_docstring = """n_features_in_:
        The number of features in the train set used in fit.
        """.rstrip()

# Methods
_fit_not_learn_docstring = """fit:
        This transformer does not learn parameters.
        """.rstrip()

_fit_transform_docstring = """fit_transform:
        Fit to data, then transform it.

    get_feature_names_out:
        Get output feature names for transformation.

    get_params:
        Get parameters for this estimator.

    set_params:
        Set the parameters of this estimator.
        """.rstrip()

_inverse_transform_docstring = """inverse_transform:
        Convert the data back to the original representation.
        """.rstrip()
