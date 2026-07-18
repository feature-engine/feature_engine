"""Utilities for docstring in Feature-engine.

Adapted from the project imbalanced-learn:

https://github.com/scikit-learn-contrib/imbalanced-learn/blob/
imblearn/utils/_docstring.py#L7
"""


class Substitution:
    """Decorate a function's or a class' docstring to perform string
    substitution on it.

    Only the placeholders whose names are passed as keyword arguments are
    replaced, so literal braces in the docstring (e.g., in code examples)
    are left untouched.

    This decorator should be robust even if obj.__doc__ is None
    (for example, if -OO was passed to the interpreter).
    """

    def __init__(self, **kwargs):
        self.params = kwargs

    def __call__(self, obj):
        doc = obj.__doc__
        if doc:
            for key, value in self.params.items():
                doc = doc.replace("{" + key + "}", value)
            obj.__doc__ = doc
        return obj
