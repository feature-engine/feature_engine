from typing import Any, List, Union

Variables = Union[None, int, str, List[Union[str, int]]]


def _check_variables_input_value(variables: Variables) -> Any:
    """
    Checks that the input value for the `variables` parameter located in the init of
    all Feature-engine transformers is of the correct type.
    Allowed  values are None, int, str or list of strings and integers.

    Parameters
    ----------
    variables : string, int, list of strings, list of integers. Default=None

    Returns
    -------
    variables: same as input
    """

    msg = (
        "`variables` should contain a string, an integer or a list of strings or "
        f"integers. Got {variables} instead."
    )
    msg_dupes = "The list entered in `variables` contains duplicated variable names."
    msg_empty = "The list of `variables` is empty."

    if variables is not None:
        if isinstance(variables, list):
            if not all(isinstance(i, (str, int)) for i in variables):
                raise ValueError(msg)
            if len(variables) == 0:
                raise ValueError(msg_empty)
            if len(variables) != len(set(variables)):
                raise ValueError(msg_dupes)
        else:
            if not isinstance(variables, (str, int)):
                raise ValueError(msg)
    return variables
