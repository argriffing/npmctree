"""
Utility functions that do not use external packages.

"""
from __future__ import division, print_function, absolute_import

import operator

import numpy as np


def weighted_choice(n, p=None):
    #TODO older versions of numpy do not have the p keyword
    return np.random.choice(range(n), p=p)


def isboolobj(x):
    return x.dtype == np.bool


def make_fvec1d(n):
    return np.zeros(n, dtype=bool)


def make_fvec2d(n):
    return np.zeros((n, n), dtype=bool)


def make_distn1d(n):
    return np.zeros(n, dtype=float)


def make_distn2d(n):
    return np.zeros((n, n), dtype=float)


def _xdivy(x, y):
    if x:
        return x / y
    else:
        return x
xdivy = np.vectorize(_xdivy)


def normalized(x):
    return xdivy(x, x.sum())


def ddec(**kwargs):
    """
    A decorator that puts some named string substitutions into a docstring.

    """
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(**kwargs)
        return obj
    return dec

