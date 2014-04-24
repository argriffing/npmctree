"""
Brute force likelihood calculations.

This module is only for testing.

"""
from __future__ import division, print_function, absolute_import

import numpy as np

import npmctree
from npmctree.util import ddec, make_distn1d, make_distn2d, normalized
from npmctree.history import (
        get_history_feas, get_history_lhood, gen_plausible_histories)
from ._generic_lmap_lhood import params, validated_params

__all__ = [
        'get_lhood_brute',
        'get_node_to_distn1d_brute',
        'get_edge_to_distn2d_brute',
        ]


@ddec(params=params)
def get_lhood_brute(*args):
    """
    Get the likelihood of this combination of parameters.

    Use brute force enumeration over all possible states.

    Parameters
    ----------
    {params}

    """
    args = validated_params(*args)
    T, edge_to_P, root, root_prior_distn1d, node_to_data_lmap = args

    lk_total = None
    for node_to_state in gen_plausible_histories(node_to_data_lmap):
        lk = get_history_lhood(T, edge_to_P, root,
                root_prior_distn1d, node_to_state)
        if lk is not None:
            probs = [node_to_data_lmap[v][s] for v, s in node_to_state.items()]
            lk *= np.prod(probs)
            if lk_total is None:
                lk_total = lk
            else:
                lk_total += lk
    return lk_total


@ddec(params=params)
def get_node_to_distn1d_brute(*args):
    """
    Get the map from node to state distribution.

    Use brute force enumeration over all possible states.

    Parameters
    ----------
    {params}

    """
    args = validated_params(*args)
    T, edge_to_P, root, root_prior_distn1d, node_to_data_lmap = args

    n = root_prior_distn1d.shape[0]
    nodes = set(node_to_data_lmap)
    v_to_d = dict((v, make_distn1d(n)) for v in nodes)
    for node_to_state in gen_plausible_histories(node_to_data_lmap):
        lk = get_history_lhood(T, edge_to_P, root,
                root_prior_distn1d, node_to_state)
        if lk is not None:
            probs = [node_to_data_lmap[v][s] for v, s in node_to_state.items()]
            lk *= np.prod(probs)
            for node, state in node_to_state.items():
                v_to_d[node][state] += lk
    return dict((v, normalized(d)) for v, d in v_to_d.items())


@ddec(params=params)
def get_edge_to_distn2d_brute(*args):
    """

    Parameters
    ----------
    {params}

    """
    args = validated_params(*args)
    T, edge_to_P, root, root_prior_distn1d, node_to_data_lmap = args

    n = root_prior_distn1d.shape[0]
    edge_to_d = dict((edge, make_distn2d(n)) for edge in T.edges())
    for node_to_state in gen_plausible_histories(node_to_data_lmap):
        lk = get_history_lhood(T, edge_to_P, root,
                root_prior_distn1d, node_to_state)
        if lk is not None:
            probs = [node_to_data_lmap[v][s] for v, s in node_to_state.items()]
            lk *= np.prod(probs)
            for tree_edge in T.edges():
                va, vb = tree_edge
                sa = node_to_state[va]
                sb = node_to_state[vb]
                edge_to_d[tree_edge][sa, sb] += lk
    return dict((v, normalized(d)) for v, d in edge_to_d.items())


# function suite for testing
fnsuite = (get_lhood_brute, get_node_to_distn1d_brute, get_edge_to_distn2d_brute)

