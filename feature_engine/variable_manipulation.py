# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause
# functions shared across transformers


def _define_variables(variables):
    # Check that variable names are passed in a list.
    # Can take None as value
    if not variables or isinstance(variables, list):
        variables = variables
    else:
        variables = [variables]
    return variables


def _define_numerical_dict(the_dict):
    # Check that the entered dictionary is indeed a dictionary of integers and floats
    # Can take None as value
    if not the_dict:
        the_dict = the_dict
    elif isinstance(the_dict, dict):
        if not all([isinstance(x, (float, int)) for x in the_dict.values()]):
            raise ValueError('All values in the dictionary must be integer or float')
    else:
        raise TypeError('The parameter can only take a dictionary or None')
    return the_dict


def _find_numerical_variables(X, variables=None):
    # Find numerical variables in a data set or check that
    # the variables entered by the user are numerical.
    if not variables:
        variables = list(X.select_dtypes(include='number').columns)
        if len(variables) == 0:
            raise ValueError('No numerical variables in this dataframe. Please check variable format with dtypes')
    else:
        if len(X[variables].select_dtypes(exclude='number').columns) != 0:
            raise TypeError("Some of the variables are not numerical. Please cast them as numerical "
                            "before calling this transformer")
    return variables


def _find_categorical_variables(X, variables=None):
    # Find categorical variables in a data set or check that
    # the variables entered by user are categorical.
    if not variables:
        variables = list(X.select_dtypes(include='O').columns)
        if len(variables) == 0:
            raise ValueError('No categorical variables in this dataframe. Please check variable format with dtypes')
    else:
        # variables indicated by user
        if len(X[variables].select_dtypes(exclude='O').columns) != 0:
            raise TypeError("Some of the variables are not categorical. Please cast them as object "
                            "before calling this transformer")
    return variables


def _find_all_variables(X, variables=None):
    # Find all variables in a data set
    if not variables:
        variables = list(X.columns)
    else:
        # variables indicated by user
        if len(set(variables).difference(X.columns)) != 0:
            raise TypeError("Some variables are not present in the dataset. Please check your variable"
                            " list")
    return variables
