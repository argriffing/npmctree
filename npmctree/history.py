"""
Functions related to histories on trees.

Every node in a Markov chain tree history has a known state.
These functions are mostly for testing.

"""
from __future__ import division, print_function, absolute_import

import itertools

import numpy as np
import networkx as nx

__all__ = [
        'get_history_lhood',
        'get_history_feas',
        'gen_plausible_histories',
        'gen_plausible_histories_from_xmap',
        ]


def get_history_feas(T, edge_to_A, root, root_prior_fvec1d, node_to_state):
    """
    Compute the feasibility of a single specific history.

    """
    root_state = node_to_state[root]
    if not root_prior_fvec1d[root_state]:
        return False
    for edge in nx.bfs_edges(T, root):
        va, vb = edge
        A = edge_to_A[edge]
        sa = node_to_state[va]
        sb = node_to_state[vb]
        if not A[sa, sb]:
            return False
    return True


def get_history_lhood(T, edge_to_P, root, root_prior_distn1d, node_to_state):
    """
    Compute the probability of a single specific history.

    """
    root_state = node_to_state[root]
    if not root_prior_distn1d[root_state]:
        return None
    lk = root_prior_distn1d[root_state]
    for edge in nx.bfs_edges(T, root):
        va, vb = edge
        P = edge_to_P[edge]
        sa = node_to_state[va]
        sb = node_to_state[vb]
        if not P[sa, sb]:
            return None
        lk *= P[sa, sb]
    return lk


def gen_plausible_histories(node_to_data_fvec1d):
    """
    Yield histories compatible with directly observed data.

    Each history is a map from node to state.
    Some of these histories may have zero probability when the
    shape of the tree and the structure of the transition matrices
    is taken into account.

    """
    # for each node define the set of states that are data-feasible
    node_set_pairs = []
    for node, fvec1d in node_to_data_fvec1d.items():
        pair = (node, set(i for i, x in enumerate(fvec1d) if x))
        node_set_pairs.append(pair)

    # yield combinatorial node_to_state assignments
    nodes, sets = zip(*node_set_pairs)
    for assignment in itertools.product(*sets):
        node_to_state = dict(zip(nodes, assignment))
        yield node_to_state


def gen_plausible_histories_from_xmap(all_nodes, nstates, xmap):
    """
    Yield histories compatible with directly observed data.

    Each history is a map from node to state.
    Some of these histories may have zero probability when the
    shape of the tree and the structure of the transition matrices
    is taken into account.

    Parameters
    ----------
    all_nodes : collection of nodes
        The collection of all nodes in the tree.
        Nodes present in this collection but missing from the xmap
        are assumed to have completely unknown state.
    nstates : integer
        Size of the state space.
        In this npmctree package the state space is considered
        fixed across all nodes and branches in the tree.
    xmap : dict
        Map from some nodes to their known states.
        Nodes either have completely known or completely unknown state.
        Nodes with completely known state are in the xmap,
        and nodes with completely unknown state are missing from the xmap.

    """
    hidden_nodes = list(set(all_nodes) - set(xmap))
    nhidden = len(hidden_nodes)
    for assignment in itertools.product(range(nstates), repeat=nhidden):
        h = dict(zip(hidden_nodes, assignment))
        h.update(xmap)
        yield h
