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


def _define_imputer_dict(imputer_dict):
    # Check that imputer dict is passed as a dictionary
    # Can take None as value
    if not imputer_dict:
        imputer_dict = imputer_dict
    elif isinstance(imputer_dict, dict):
        if not all([isinstance(x, float) or isinstance(x, int) for x in imputer_dict.values()]):
            raise ValueError('All values in imputer_dict have to be integer or float type')
    else:
        raise TypeError('Imputer_dict parameters allow passing dictionaries only or none value')
    return imputer_dict


def _find_numerical_variables(X, variables=None):
    # Find numerical variables in a data set or check that
    # the variables entered by the user are numerical.
    if not variables:
        variables = list(X.select_dtypes(include='number').columns)
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
    else:
        # variables indicated by user
        if len(X[variables].select_dtypes(exclude='O').columns) != 0:
            raise TypeError("Some of the variables are not categorical. Please cast them as object "
                            "before calling this transformer")
    return variables
