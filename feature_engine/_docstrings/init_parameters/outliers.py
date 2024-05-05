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

_fold_docstring = """fold: int, float or 'auto', default='auto'.
        The factor used to multiply the std, MAD or IQR to calculate
        the maximum or minimum allowed values.
        When 'auto', `fold` is set based on the `capping_method`: \n
         - If `capping_method='quantile'` then `'fold'` = 0.05; \n
         - If `capping_method='gaussian'` then `'fold'` = 3.0; \n
         - If `capping_method='mad'` then `'fold'` = 3.29; \n
         - If `capping_method='iqr'` then `'fold'` = 1.5. \n
        Recommended values are 2, 2.5 or 3 for the gaussian approximation,
        1.5 or 3 for the IQR proximity rule and 3 or 3.5 for MAD rule. \n
        If `capping_method='quantile'`, then `'fold'` indicates the percentile. So if
        `fold=0.05`, the limits will be the 95th and 5th percentiles. \n
        **Note**: When `capping_method='quantile'`, the maximum `fold` allowed is 0.2,
        which will find boundaries at the 20th and 80th percentile.
    """.rstrip()
