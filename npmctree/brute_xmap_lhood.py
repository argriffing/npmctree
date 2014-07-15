"""
Brute force likelihood calculations.

This module is only for testing.

"""
from __future__ import division, print_function, absolute_import

import numpy as np

from .util import ddec, make_distn1d, make_distn2d, normalized
from .history import (
        get_history_feas, get_history_lhood,
        gen_plausible_histories_from_xmap)
from ._generic_xmap_lhood import params, validated_params

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
    T, edge_to_P, root, root_prior_distn1d, xmap = args

    all_nodes = set(T)
    nstates = root_prior_distn1d.shape[0]

    lk_total = None
    for h in gen_plausible_histories_from_xmap(all_nodes, nstates, xmap):
        lk = get_history_lhood(T, edge_to_P, root, root_prior_distn1d, h)
        if lk is not None:
            if lk_total is None:
                lk_total = lk
            else:
                lk_total += lk
    return lk_total


@ddec(params=params)
def get_node_to_distn1d_brute(*args):
    """
    Get the map from node to state distribution.

    Use brute force enumeration over all possible state assignments.

    Parameters
    ----------
    {params}

    """
    args = validated_params(*args)
    T, edge_to_P, root, root_prior_distn1d, xmap = args

    all_nodes = set(T)
    nstates = root_prior_distn1d.shape[0]

    nodes = set(node_to_data_fvec1d)
    v_to_d = dict((v, make_distn1d(nstates)) for v in nodes)
    for h in gen_plausible_histories_from_xmap(all_nodes, nstates, xmap):
        lk = get_history_lhood(T, edge_to_P, root, root_prior_distn1d, xmap)
        if lk is not None:
            for node, state in h.items():
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
    T, edge_to_P, root, root_prior_distn1d, xmap = args

    all_nodes = set(T)
    nstates = root_prior_distn1d.shape[0]

    edge_to_d = dict((edge, make_distn2d(nstates)) for edge in T.edges())
    for h in gen_plausible_histories(all_nodes, nstates, xmap):
        lk = get_history_lhood(T, edge_to_P, root, root_prior_distn1d, h)
        if lk is not None:
            for tree_edge in T.edges():
                va, vb = tree_edge
                sa = node_to_state[va]
                sb = node_to_state[vb]
                edge_to_d[tree_edge][sa, sb] += lk
    for tree_edge in T.edges():
        edge_to_d[tree_edge] = normalized(edge_to_d[tree_edge])
    return edge_to_d


# function suite for testing
fnsuite = (get_lhood_brute, get_node_to_distn1d_brute, get_edge_to_distn2d_brute)

