_return_object_docstring = """return_object: bool, default=False
        Whether the the discrete variable should be returned as type numeric or type
        object. If you would like to encode the discrete variables with Feature-engine's
        categorical encoders, use True. Alternatively, keep the default to False.
    """.rstrip()

_return_boundaries_docstring = """return_boundaries: bool, default=False
        Whether the output should be the interval boundaries. If True, it returns
        the interval boundaries. If False, it returns integers.
    """.rstrip()

_precision_docstring = """precision: int, default=3
        The precision at which to store and display the bins labels.
    """.rstrip()
