_drop_original_docstring = """drop_original: bool, default=False
        If True, the original variables will be dropped from the dataframe after
        creating the features.
    """.rstrip()

_missing_values_docstring = """missing_values: string, default='raise'
        Indicates if missing values should be ignored or raised. If 'raise' the
        transformer will return an error if the the datasets to `fit` or `transform`
        contain missing values. If 'ignore', missing data will be ignored when creating
        the features.
        """
_transform_docstring = """transform:
        Create and add the new features.
    """.rstrip()
