from typing import Optional


def _check_numerical_dict(dict_: Optional[dict]) -> Optional[dict]:
    """
    Checks that all values in the dictionary are integers and floats. It can take also
    take None as value.

    Parameters
    ----------
    dict_ : dict
        The dictionary that will be checked.

    Raises
    ------
    ValueError
        If any of the values in the dictionary are not int or float.
    TypeError
        When input type is not a dictionary.
    """

    if isinstance(dict_, dict):
        if not all([isinstance(x, (float, int)) for x in dict_.values()]):
            raise ValueError(
                "All values in the dictionary must be integer or float. "
                f"Got {dict_} instead."
            )

    elif dict_ is not None:
        raise TypeError(
            f"The parameter can only take a dictionary or None. Got {dict_} instead."
        )
    return None
