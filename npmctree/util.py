"""
Utility functions that do not use external packages.

"""
from __future__ import division, print_function, absolute_import

import operator

import numpy as np


def weighted_choice(n, p=None):
    # Older versions of numpy do not have the p keyword
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


def xmap_to_lmap(all_nodes, nstates, xmap, validate=True):
    """
    Convert from a more restricted to a less restricted observation format.

    Use of this function is a likely indicator of inefficiency
    in the calling function.
    For greater efficiency the caller should use the xmap format directly.

    """
    all_nodes = set(all_nodes)
    observed_nodes = set(xmap)
    hidden_nodes = all_nodes - observed_nodes
    if validate:
        extra_nodes = observed_nodes - all_nodes
        if extra_nodes:
            raise ValueError('extra nodes: %s' % extra_nodes)
        for state in xmap.values():
            if int(state) != state:
                raise ValueError('expected integer state')
            if not (0 <= state < nstates):
                raise ValueError('state is out of bounds')
    node_to_lmap = {}
    for node, state in xmap.items():
        lmap = np.zeros(nstates, dtype=float)
        lmap[state] = 1
        node_to_lmap[node] = lmap
    for node in hidden_nodes:
        node_to_lmap[node] = np.ones(nstates, dtype=float)
    return node_to_lmap
