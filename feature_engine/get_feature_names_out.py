"""Method functionality shared by all transformers that do not add features to the data,
instead modify features in place."""

from typing import List, Optional, Union


def _get_feature_names_out(
        features_in: List[Union[str, int]],
        transformed_features: List[Union[str, int]],
        input_features: Optional[List] = None,
) -> List[Union[str, int]]:
    """
    If input_features is None, returns the names of all variables in the transformed
    dataframe. If input_features is a list, returns the feature names in the list. This
    parameter exists mostly for compatibility with the Scikit-learn Pipeline.

    Parameters
    ----------
    features_in: List
        The variables associated with the transformed dataset.

    transformed_features: list
        The variables modified by this transformer.

    input_features : List, default=None
        The features which name should be returned.

    Raises
    ------
    ValueError
        If input_features is not a list or any of the features in input_features are
        not transformed by this transformer.

    Returns
    -------
    feature_names: list
        The name of the features.
    """
    if input_features is None:
        # return all feature names
        feature_names = features_in

    else:
        # Return features requested by user.
        if not isinstance(input_features, list):
            raise ValueError(
                f"input_features must be a list. Got {input_features} instead."
            )
        if any([
            feature for feature in input_features if feature not in transformed_features
        ]):
            raise ValueError(
                "Some features in input_features were not transformed by this "
                "transformer. Pass either None, or a list with the features "
                "that were transformed by this transformer."
            )
        feature_names = input_features

    return feature_names
