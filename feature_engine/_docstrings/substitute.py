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
