"""Docstrings for the methods. They are meant to be used in the init docstrings of
the transformers."""

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

# used in discretisers module
_fit_discretiser_docstring = """fit:
        Find the interval limits.
    """.rstrip()

_transform_discretiser_docstring = """transform:
        Sort continuous variable values into the intervals.
    """.rstrip()

# used in imputation module
_transform_imputers_docstring = """transform:
        Impute missing data.
    """.rstrip()
