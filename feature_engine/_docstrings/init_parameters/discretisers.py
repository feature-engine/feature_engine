_return_object_docstring = """return_object: bool, default=False
        Whether the the discrete variable should be returned as numeric or as
        object. If you would like to proceed with the engineering of the variable as if
        it was categorical, use True. Alternatively, keep the default to False.
    """.rstrip()

_return_boundaries_docstring = """return_boundaries: bool, default=False
        Whether the output should be the interval boundaries. If True, it returns
        the interval boundaries. If False, it returns integers.
    """.rstrip()

_binner_dict_docstring = """binner_dict_:
         Dictionary with the interval limits per variable.
     """.rstrip()