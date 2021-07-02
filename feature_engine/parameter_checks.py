from typing import Optional


def _define_numerical_dict(dict_: Optional[dict]) -> Optional[dict]:
    """
    Checks if all values in dictionary are integers and floats. Can take None as
    argument.

    Parameters
    ----------
    the_dict : dict
        The dictionary that will be checked

    Raises
    ------
    ValueError
        If any of the values in the dictionary are not int or float
    TypeError
        When argument type is not a dictionary.

    Returns
    -------
    None or the input dictionary
    """

    if not dict_:
        dict_ = dict_

    elif isinstance(dict_, dict):
        if not all([isinstance(x, (float, int)) for x in dict_.values()]):
            raise ValueError("All values in the dictionary must be integer or float")

    else:
        raise TypeError("The parameter can only take a dictionary or None")

    return dict_
