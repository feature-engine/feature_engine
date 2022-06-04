_variables_docstring = """variables: list, default=None
        The list of categorical variables that will be encoded. If None, the
        encoder will find and transform all variables of type object or categorical by
        default. You can also make the transformer accept numerical variables, see the
        next parameter.
    """.rstrip()

_ignore_format_docstring = """ignore_format: bool, default=False
        Whether the format in which the categorical variables are cast should be
        ignored. If False, the encoder will automatically select variables of type
        object or categorical, or check that the variables entered by the user are of
        type object or categorical. If True, the encoder will select all variables or
        accept all variables entered by the user, including those cast as numeric.
    """.rstrip()

_errors_docstring = """errors: string, default='ignore'
        Indicates what to do, when categories not present in the train set are
        encountered during transform. If 'raise', then unseen categories will raise an
        error. If 'ignore', then unseen categories will be set as NaN and a warning will
        be raised instead. If 'encode', then unseen categories will be encoded according
        to the default strategy from the transformer, provided that it supports it.
    """.rstrip()

_errors_docstring_with_encode = _errors_docstring + """
        If 'encode', then unseen categories will be encoded according to the default
        strategy from the transformer (see the 'Notes' section for details).
    """.rstrip()

_transform_docstring = """transform:
        Encode the categories to numbers.
    """.rstrip()
