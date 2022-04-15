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

_input_features_docstring = """input_features: str, list, default=None
        Input features. If `None`, then the names of all the variables in the
        transformed dataset (original + new variables) is returned. If list
        with feature names, the features in the list will be returned."
    """.rstrip()