from typing import List, Optional, Union


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


def check_input_features(
        input_features: Union[List, str], all_variables: List
) -> Union[List, str]:
    """
    Checks if input_features is None; otherwise, check whether input_variables
    is a list of features that exist within transformed dataset.

    Parameters
    ----------
    input_features : Union[List, str]
        The features to check whether they

    all_variables: List
        All the variables associated with the transformed dataset.

    Raises
    ------
    ValueError
        If input_features is not a list or all the features are not included in the transformed
        variables.

    Returns
    -------
    res: bool
        Returns True if all the input_features are included in variables; otherwise,
        returns False.

    """
    if input_features is None:
        # return all feature names
        feature_names = all_variables

    else:
        # Return features requested by user.
        if not isinstance(input_features, list):
            raise ValueError(
                f"input_features must be a list. Got {input_features} instead."
            )
        if any([
            feature for feature in input_features if feature not in all_variables
        ]):
            raise ValueError(
                "Some features in input_features were not transformed by this "
                "transformer. Pass either None, or a list with the features "
                "that were transformed by this transformer."
            )
        feature_names = input_features

    return feature_names


def _check_input_features_in_variables(
        input_features: List, all_variables: List
) -> bool:
    """
    Checks if all input_variables are included in the variables list.

    Parameters
    ----------
    input_features : Union[List, str]
        The features to assess whether they exist.

    all_variables: List
        All the variables associated with the transformed dataset.

    Returns
    -------
    res: bool
        Returns True if all the input_features are included in variables; otherwise,
        returns False.
    """
    res = any([feature for feature in input_features if feature not in all_variables])
    return res
