"""
CustomFunctions is a wrapper, which allows MathFunctions to detect if a
custom function should be processed via pandas.agg() or numpy.apply_over_axes()
"""


class CustomFunctions:


    def __init__(self, scope_target):
        self.scope_target = scope_target
