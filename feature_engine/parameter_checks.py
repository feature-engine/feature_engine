from typing import Any, List, Optional


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


def _parse_features_to_extract(feats_requested, feats_supported: List[str]) -> Any:
    """
    Parses argument features_to_extract of ExtractDateTransformer
    (potentially shares utility with other similar feature
    extractor transformers)

    Parameters
    ----------
    feats_requested
        argument passed to the transformer as features_to_extract

    feats_supported: list[str]
        list of supported date features that the transformer is able
        to extract from the existing datetime variables

    Raises
    ------
    TypeError
        If requested features is not a string or list
        If supported features is not a list of strings

    ValueError
        If one or more requested features are not supported
        by the transformer

    """
    if not isinstance(feats_requested, (str, list)):
        raise TypeError("feats_requested must be either a str or a list of str")

    if not isinstance(feats_supported, list) or any(
        not isinstance(feat, str) for feat in feats_supported
    ):
        raise TypeError("feats_supported must be a list of str")

    if feats_requested == "all":
        return feats_supported

    if isinstance(feats_requested, str):
        feats_requested = [feats_requested]

    if any(feature not in feats_supported for feature in feats_requested):
        raise ValueError(
            "At least one of the requested feature is not supported. "
            "Supported features are {}.".format(", ".join(feats_supported))
        )

    return feats_requested
