"""Utilities for docstring in Feature-engine.

Taken from the project imbalanced-learn:

https://github.com/scikit-learn-contrib/imbalanced-learn/blob/
imblearn/utils/_docstring.py#L7
"""


class Substitution:
    """Decorate a function's or a class' docstring to perform string
    substitution on it.
    This decorator should be robust even if obj.__doc__ is None
    (for example, if -OO was passed to the interpreter).
    """

    def __init__(self, *args, **kwargs):
        if args and kwargs:
            raise AssertionError("Only positional or keyword args are allowed")

        self.params = args or kwargs

    def __call__(self, obj):
        obj.__doc__ = obj.__doc__.format(**self.params)
        return obj


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

_n_features_in_docstring = """n_features_in_:
        The number of features in the train set used in fit.
        """.rstrip()

# Methods
_fit_not_learn_docstring = """fit:
        This transformer does not learn parameters.
        """.rstrip()

_fit_transform_docstring = """fit_transform:
        Fit to data, then transform it.
        """.rstrip()

_inverse_transform_docstring = """inverse_transform:
        Convert the data back to the original representation.
        """.rstrip()
