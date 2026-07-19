"""Docstrings for the parameters corresponding to the  __init__"""

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

_drop_original_docstring = """drop_original: bool, default=False
        If True, the original variables to transform will be dropped from the dataframe.
    """.rstrip()

_missing_values_docstring = """missing_values: string, default='raise'
        Indicates if missing values should be ignored or raised. If `'raise'` the
        transformer will return an error if the the datasets to `fit` or `transform`
        contain missing values. If `'ignore'`, missing data will be ignored when
        learning parameters or performing the transformation.
        """.rstrip()

_return_empty_docstring = """return_empty : bool, default=False
        Whether to return an empty list when no variables of the required type are
        found. If False, the transformer raises an error.

        .. versionadded:: 2.0
           `return_empty` currently defaults to False. The default will change to
           True in version 2.1. To keep the current behaviour and silence the
           warning, explicitly set `return_empty=False` instead of relying on the
           default.
        """.rstrip()
