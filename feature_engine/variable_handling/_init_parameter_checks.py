from typing import Any

from feature_engine.variable_handling.variable_type_selection import Variables


# set return value typehint to Any to avoid issues with the base transformer fit method
def _check_init_parameter_variables(variables: Variables) -> Any:
    """
    Checks that the input is of the correct type. Allowed  values are None, int, str or
    list of strings and ints.

    Parameters
    ----------
    variables : string, int, list of strings, list of integers. Default=None

    Returns
    -------
    variables: same as input
    """

    msg = "variables should be a string, an int or a list of strings or integers."
    msg_dupes = "the list contains duplicated variable names"

    if variables:
        if isinstance(variables, list):
            if not all(isinstance(i, (str, int)) for i in variables):
                raise ValueError(msg)
            if len(variables) != len(set(variables)):
                raise ValueError(msg_dupes)
        else:
            if not isinstance(variables, (str, int)):
                raise ValueError(msg)

    return variables
