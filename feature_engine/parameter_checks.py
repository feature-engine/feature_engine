from typing import Optional


def _define_numerical_dict(dict_: Optional[dict]) -> Optional[dict]:
    """
    Takes a dictionary and checks if all values in dictionary are integers and floats.
    Can take None as argument.

    Args:
        the_dict: Dict to perform check against

    Raises:
        ValueError: If all values of dict are not int or float
        TypeError: When argument type is not dict

    Returns:
        None or the dict

    """

    if not dict_:
        dict_ = dict_

    elif isinstance(dict_, dict):
        if not all([isinstance(x, (float, int)) for x in dict_.values()]):
            raise ValueError("All values in the dictionary must be integer or float")

    else:
        raise TypeError("The parameter can only take a dictionary or None")

    return dict_
