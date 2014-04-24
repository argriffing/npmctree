"""
Brute force feasibility calculations.

This module is only for testing.

"""
from __future__ import division, print_function, absolute_import

import numpy as np

from .util import ddec, make_fvec1d, make_fvec2d
from .history import get_history_feas, gen_plausible_histories
from ._generic_fset_feas import params, validated_params

__all__ = [
        'get_feas_brute',
        'get_node_to_fvec1d_brute',
        'get_edge_to_fvec2d_brute',
        ]


@ddec(params=params)
def get_feas_brute(*args):
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
    args = validated_params(*args)
    T, edge_to_A, root, root_prior_fvec1d, node_to_data_fvec1d = args

    for node_to_state in gen_plausible_histories(node_to_data_fvec1d):
        if get_history_feas(T, edge_to_A, root,
                root_prior_fvec1d, node_to_state):
            return True
    return False


@ddec(params=params)
def get_node_to_fvec1d_brute(*args):
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
    args = validated_params(*args)
    T, edge_to_A, root, root_prior_fvec1d, node_to_data_fvec1d = args

    n = root_prior_fvec1d.shape[0]
    nodes = set(node_to_data_fvec1d)
    v_to_feas = dict((v, make_fvec1d(n)) for v in nodes)
    for node_to_state in gen_plausible_histories(node_to_data_fvec1d):
        if get_history_feas(T, edge_to_A, root,
                root_prior_fvec1d, node_to_state):
            for node, state in node_to_state.items():
                v_to_feas[node][state] = True
    return v_to_feas


@ddec(params=params)
def get_edge_to_fvec2d_brute(*args):
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
    args = validated_params(*args)
    T, edge_to_A, root, root_prior_fvec1d, node_to_data_fvec1d = args

    n = root_prior_fvec1d.shape[0]
    edge_to_d = dict((edge, make_fvec2d(n)) for edge in T.edges())
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

