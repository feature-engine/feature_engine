def _define_numerical_dict(the_dict):
    # Check that the entered dictionary is indeed a dictionary of integers and floats
    # Can take None as value
    if not the_dict:
        the_dict = the_dict
    elif isinstance(the_dict, dict):
        if not all([isinstance(x, (float, int)) for x in the_dict.values()]):
            raise ValueError("All values in the dictionary must be integer or float")
    else:
        raise TypeError("The parameter can only take a dictionary or None")
    return the_dict
