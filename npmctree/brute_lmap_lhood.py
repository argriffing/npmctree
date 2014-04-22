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

__all__ = [
        'get_lhood_brute',
        'get_node_to_distn1d_brute',
        'get_edge_to_distn2d_brute',
        ]


params = """\
    T : directed networkx tree graph
        Edge and node annotations are ignored.
    edge_to_adjacency : dict
        A map from directed edges of the tree graph
        to networkx graphs representing state transition feasibility.
    root : hashable
        This is the root node.
        Following networkx convention, this may be anything hashable.
    root_prior_distn1d : dict
        Prior state distribution at the root.
    node_to_data_lmap : dict
        For each node, a map from state to observation likelihood.
"""


@ddec(params=params)
def get_lhood_brute(T, edge_to_P, root, root_prior_distn1d, node_to_data_lmap):
    """
    Get the likelihood of this combination of parameters.

    Use brute force enumeration over all possible states.

    Parameters
    ----------
    {params}

    """
    lk_total = None
    for node_to_state in gen_plausible_histories(node_to_data_lmap):
        lk = get_history_lhood(T, edge_to_P, root,
                root_prior_distn1d, node_to_state)
        if lk is not None:
            probs = [node_to_data_lmap[v][s] for v, s in node_to_state.items()]
            lk *= prod(probs)
            if lk_total is None:
                lk_total = lk
            else:
                lk_total += lk
    return lk_total


@ddec(params=params)
def get_node_to_distn1d_brute(T, edge_to_P, root,
        root_prior_distn1d, node_to_data_lmap):
    """
    Get the map from node to state distribution.

    Use brute force enumeration over all possible states.

    Parameters
    ----------
    {params}

    """
    n = root_prior_distn1d.shape[0]
    nodes = set(node_to_data_lmap)
    v_to_d = dict((v, make_distn1d(n)) for v in nodes)
    for node_to_state in gen_plausible_histories(node_to_data_lmap):
        lk = get_history_lhood(T, edge_to_P, root,
                root_prior_distn1d, node_to_state)
        if lk is not None:
            probs = [node_to_data_lmap[v][s] for v, s in node_to_state.items()]
            lk *= prod(probs)
            for node, state in node_to_state.items():
                v_to_d[node][state] += lk
    return dict((v, normalized(d)) for v, d in v_to_d.items())


@ddec(params=params)
def get_edge_to_distn2d_brute(T, edge_to_P, root,
        root_prior_distn1d, node_to_data_lmap):
    """

    Parameters
    ----------
    {params}

    """
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

