"""Docstrings for the parameters corresponding to the  __init__"""

# ============ input variables
_variables_numerical_docstring = """variables: list, default=None
        The list of numerical variables to transform. If None, the transformer will
        automatically find and select all numerical variables.
    """.rstrip()

_variables_categorical_docstring = """variables: list, default=None
        The list of categorical variables that will be encoded. If None, the
        encoder will find and transform all variables of type object or categorical by
        default. You can also make the transformer accept numerical variables, see the
        parameter `ignore_format`.
    """.rstrip()
# ============ input variables

# ============ shared
_drop_original_docstring = """drop_original: bool, default=False
        If True, the original variables to transform will be dropped from the dataframe.
    """.rstrip()

_missing_values_docstring = """missing_values: string, default='raise'
        Indicates if missing values should be ignored or raised. If 'raise' the
        transformer will return an error if the the datasets to `fit` or `transform`
        contain missing values. If 'ignore', missing data will be ignored when learning
        parameters or performing the transformation.
        """.rstrip()
# ============ shared

# ============= used in categorical encoders
_ignore_format_docstring = """ignore_format: bool, default=False
        This transformer operates only on variables of type object or categorical. To
        override this behaviour and allow the transformer to transform numerical
        variables as well, set to `True`.\n
        If `ignore_format` is False, the encoder will automatically select variables of
        type object or categorical, or check that the variables entered by the user are
        of type object or categorical. If True, the encoder will select all variables
        or accept all variables entered by the user, including those cast as numeric.
    """.rstrip()

_unseen_docstring = """unseen: string, default='ignore'
        Indicates what to do when categories not present in the train set are
        encountered during transform. If 'raise', then unseen categories will raise an
        error. If 'ignore', then unseen categories will be set as NaN and a warning will
        be raised instead.
    """.rstrip()
# ============= used in categorical encoders
