_ignore_format_docstring = """ignore_format: bool, default=False
        This transformer operates only on variables of type object or categorical. To
        override this behaviour and allow the transformer to transform numerical
        variables as well, set to `True`.\n
        If `ignore_format` is `False`, the encoder will automatically select variables
        of type object or categorical, or check that the variables entered by the user
        are of type object or categorical. If `True`, the encoder will select all
        variables or accept all variables entered by the user, including those cast as
        numeric.\n
        In short, set to `True` when you want to encode numerical variables.
    """.rstrip()

_unseen_docstring = """unseen: string, default='ignore'
        Indicates what to do when categories not present in the train set are
        encountered during transform. If `'raise'`, then unseen categories will raise
        an error. If `'ignore'`, then unseen categories will be encoded as NaN and a
        warning will be raised instead.
    """.rstrip()
