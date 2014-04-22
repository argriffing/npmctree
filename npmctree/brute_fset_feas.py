"""
Brute force feasibility calculations.

This module is only for testing.

"""
from __future__ import division, print_function, absolute_import

import numpy as np

import npmctree
from npmctree.util import ddec
from npmctree.history import get_history_feas, gen_plausible_histories

__all__ = [
        'get_feas_brute',
        'get_node_to_fvec1d_brute',
        'get_edge_to_fvec2d_brute',
        ]


params = """\
    T : directed networkx tree graph
        Edge and node annotations are ignored.
    edge_to_A : dict
        A map from directed edges of the tree graph
        to 2d bool ndarrays representing state transition feasibility.
    root : hashable
        This is the root node.
        Following networkx convention, this may be anything hashable.
    root_prior_fvec1d : 1d ndarray
        The set of feasible prior root states.
        This may be interpreted as the support of the prior state
        distribution at the root.
    node_to_data_fvec1d : dict
        Map from node to set of feasible states.
        The feasibility could be interpreted as due to restrictions
        caused by observed data.
"""


@ddec(params=params)
def get_feas_brute(T, edge_to_A, root, root_prior_fvec1d, node_to_data_fvec1d):
    """
    Get the feasibility of this combination of parameters.

    Use brute force enumeration over all possible states.

    Parameters
    ----------
    {params}

    Returns
    -------
    feas : bool
        True if the data is structurally supported by the model,
        otherwise False.

    """
    for node_to_state in gen_plausible_histories(node_to_data_fvec1d):
        if get_history_feas(T, edge_to_A, root,
                root_prior_fvec1d, node_to_state):
            return True
    return False


@ddec(params=params)
def get_node_to_fvec1d_brute(T, edge_to_A, root,
        root_prior_fvec1d, node_to_data_fvec1d):
    """
    Get the map from node to state feasibility.

    Use brute force enumeration over all histories.

    Parameters
    ----------
    {params}

    Returns
    -------
    node_to_posterior_fvec1d : dict
        Map from node to fvec1d of posterior feasible states.

    """
    n = root_prior_fvec1d.shape[0]
    nodes = set(node_to_data_fvec1d)
    v_to_feas = dict((v, np.zeros(n, dtype=bool)) for v in nodes)
    for node_to_state in gen_plausible_histories(node_to_data_fvec1d):
        if get_history_feas(T, edge_to_A, root,
                root_prior_fvec1d, node_to_state):
            for node, state in node_to_state.items():
                v_to_feas[node][state] = True
    return v_to_feas


@ddec(params=params)
def get_edge_to_fvec2d_brute(T, edge_to_A, root,
        root_prior_fvec1d, node_to_data_fvec1d):
    """
    Use brute force enumeration over all histories.

    Parameters
    ----------
    {params}

    Returns
    -------
    edge_to_fvec2d : map from directed edge to 2d boolean ndarray
        For each directed edge in the rooted tree report the
        2d bool ndarray among states, for which presence/absence of an edge
        defines the posterior feasibility of the corresponding state transition
        along the edge.

    """
    n = root_prior_fvec1d.shape[0]
    edge_to_d = dict((edge, np.zeros((n, n), dtype=bool)) for edge in T.edges())
    for node_to_state in gen_plausible_histories(node_to_data_fvec1d):
        if get_history_feas(T, edge_to_A, root,
                root_prior_fvec1d, node_to_state):
            for tree_edge in T.edges():
                va, vb = tree_edge
                sa = node_to_state[va]
                sb = node_to_state[vb]
                edge_to_d[tree_edge][sa, sb] = True
    return edge_to_d


# function suite for testing
fnsuite = (get_feas_brute, get_node_to_fvec1d_brute, get_edge_to_fvec2d_brute)

