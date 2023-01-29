_capping_method_docstring = """capping_method: str, default='gaussian'
        Desired outlier detection method. Can be 'gaussian', 'iqr', 'mad',
        'quantiles'. \n
        The transformer will find the maximum and / or minimum values beyond which a
        data point will be considered an outlier using:
        **'gaussian'**: the Gaussian approximation.
        **'iqr'**: the IQR proximity rule.
        **'quantiles'**: the percentiles.
        **'mad'**: the Gaussian approximation but using robust statistics.
    """.rstrip()

_tail_docstring = """tail: str, default='right'
        Whether to look for outliers on the right, left or both tails of the
        distribution. Can take 'left', 'right' or 'both'.
    """.rstrip()

_fold_docstring = """fold: int or float, default=0.05 if `quantile`, or 3 otherwise.
        The factor used to multiply the std, MAD or IQR to calculate
        the maximum or minimum allowed values.
        Recommended values are 2 or 3 for the gaussian approximation,
        1.5 or 3 for the IQR proximity rule and 3 or 3.5 for MAD rule. \n
        If `capping_method='quantile'`, then `'fold'` indicates the percentile. So if
        `fold=0.05`, the limits will be the 95th and 5th percentiles. \n
        **Note**: Outliers will be removed up to a maximum of the 20th percentiles on
        both sides. Thus, when `capping_method='quantile'`, then `'fold'` takes values
        between 0 and 0.20.
    """.rstrip()
