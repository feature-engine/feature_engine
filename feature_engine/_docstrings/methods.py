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

# used in categorical encoders
_transform_encoders_docstring = """transform:
        Encode the categories to numbers.
    """.rstrip()

# used in creation module
_transform_creation_docstring = """transform:
        Create new features.
    """.rstrip()
