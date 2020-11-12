RecursiveFeatureElimination
=======================


RecursiveFeatureElimination() selects features following a recursive process:

    1) Rank the features according to their importance derived from the estimator.

    2) Remove one feature -the least important- and fit the estimator again
    utilising the remaining features.

    3) Calculate the performance of the estimator.

    4) If the estimator performance drops beyond the indicated threshold, then
    that feature is important and should be kept.
    Otherwise, that feature is removed.

    5) Repeat steps 2-4 until all features have been evaluated.
    
