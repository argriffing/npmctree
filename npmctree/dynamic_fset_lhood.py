"""
Markov chain algorithms to compute likelihoods and distributions on trees.

The NetworkX digraph representing a sparse transition probability matrix
will be represented by the notation 'P'.
State distributions are represented by dense ndarrays.
Joint endpoint state distributions are represented by networkx graphs.

"""
from __future__ import division, print_function, absolute_import

import itertools
from collections import defaultdict

import numpy as np
import networkx as nx

import npmctree
from npmctree.util import ddec, make_distn1d, make_distn2d, normalized
from ._generic_fset_lhood import params, validated_params

__all__ = [
        'get_lhood',
        'get_node_to_distn1d',
        'get_edge_to_distn2d',
        'get_unconditional_joint_distn',
        ]


def get_unconditional_joint_distn(
        T, edge_to_P, root, root_prior_distn1d, selected_nodes):
    """
    Get the unconditional joint state distribution for a given set of nodes.

    Parameters
    ----------
    T : directed networkx tree graph
        Edge and node annotations are ignored.
    edge_to_P : dict
        A map from directed edges of the tree graph
        to 2d float ndarrays representing state transition probabilities.
    root : hashable
        This is the root node.
        Following networkx convention, this may be anything hashable.
    root_prior_distn1d : dict
        Prior state distribution at the root.
    selected_nodes : a collection consisting of a subset of nodes in T
        A collection of nodes to include in the distribution.

    Returns
    -------
    node_to_data_fvec1d_prob_pairs : sequence
        A sequence of (node_to_data_fvec1d, probability) pairs.
        The first element of each sequence maps each node
        to a 1d feasibility vector.

    """
    selected_nodes = list(selected_nodes)
    unselected_nodes = set(T) - set(selected_nodes)
    nselected = len(selected_nodes)
    nstates = root_prior_distn1d.shape[0]
    states = range(nstates)

    # Compute the state distribution at the nodes,
    # under the given parameter values.
    pairs = []
    for assignment in itertools.product(states, repeat=nselected):

        # Get the map from leaf to state.
        leaf_to_state = dict(zip(selected_nodes, assignment))

        # Define the data associated with this assignment.
        # All leaf states are fully observed.
        # All internal states are completely unobserved.
        node_to_data_fvec1d = {}
        for node in selected_nodes:
            state = leaf_to_state[node]
            fvec1d = np.zeros(nstates, dtype=bool)
            fvec1d[state] = True
            node_to_data_fvec1d[node] = fvec1d
        for node in unselected_nodes:
            fvec1d = np.ones(nstates, dtype=bool)
            node_to_data_fvec1d[node] = fvec1d

        # Compute the likelihood for this data.
        lhood = get_lhood(
                T, edge_to_P, root, root_prior_distn1d, node_to_data_fvec1d)
        pairs.append((node_to_data_fvec1d, lhood))

    # Check that the likelihoods sum to 1.
    lhood_sum = sum(p for d, p in pairs)
    if not np.allclose(lhood_sum, 1):
        raise Exception('probabilities of a finite distribution '
                'sum to %s' % lhood_sum)

    # Return the observations weighted by probability.
    return pairs


@ddec(params=params)
def get_lhood(*args):
    """
    Get the likelihood of this combination of parameters.

    Parameters
    ----------
    {params}

    Returns
    -------
    lhood : float or None
        If the data is structurally supported by the model then
        return the likelihood, otherwise None.

    """
    args = validated_params(*args)
    T, edge_to_P, root, root_prior_distn1d, node_to_data_fvec1d = args

    root_lhoods = _get_root_lhoods(*args)
    if root_lhoods.any():
        return root_lhoods.sum()
    else:
        return None


@ddec(params=params)
def get_node_to_distn1d(*args):
    """
    Get the map from node to state distribution.

    Parameters
    ----------
    {params}

    """
    args = validated_params(*args)
    T, edge_to_P, root, root_prior_distn1d, node_to_data_fvec1d = args

    v_to_subtree_partial_likelihoods = _backward(*args)
    v_to_posterior_distn1d = _forward(T, edge_to_P, root,
            v_to_subtree_partial_likelihoods)
    return v_to_posterior_distn1d


@ddec(params=params)
def get_edge_to_distn2d(*args):
    """
    Get the map from edge to joint state distribution at endpoint nodes.

    Parameters
    ----------
    {params}

    """
    args = validated_params(*args)
    T, edge_to_P, root, root_prior_distn1d, node_to_data_fvec1d = args

    v_to_subtree_partial_likelihoods = _backward(*args)
    edge_to_J = _forward_edges(T, edge_to_P, root,
            v_to_subtree_partial_likelihoods)
    return edge_to_J


@ddec(params=params)
def _get_root_lhoods(T, edge_to_P, root,
        root_prior_distn1d, node_to_data_fvec1d):
    """
    Get the posterior likelihoods at the root, conditional on root state.

    These are also known as partial likelihoods.

    Parameters
    ----------
    {params}

    """
    v_to_subtree_partial_likelihoods = _backward(T, edge_to_P, root,
            root_prior_distn1d, node_to_data_fvec1d)
    return v_to_subtree_partial_likelihoods[root]


@ddec(params=params)
def _backward(T, edge_to_P, root, root_prior_distn1d, node_to_data_fvec1d):
    """
    Determine the subtree feasible state set of each node.

    This is the backward pass of a backward-forward algorithm.

    Parameters
    ----------
    {params}

    """
    n = root_prior_distn1d.shape[0]
    v_to_subtree_partial_likelihoods = {}
    for va in nx.topological_sort(T, [root], reverse=True):
        fvec1d_data = node_to_data_fvec1d[va]
        if T[va]:
            vbs = T[va]
        else:
            vbs = set()
        if vbs:
            partial_likelihoods = make_distn1d(n)
            for s, value in enumerate(fvec1d_data):
                if value:
                    prob = _get_partial_likelihood(edge_to_P,
                            v_to_subtree_partial_likelihoods, va, vbs, s)
                    if prob is not None:
                        partial_likelihoods[s] = prob
        else:
            partial_likelihoods = np.array(fvec1d_data, dtype=float)
        if va == root:
            partial_likelihoods *= root_prior_distn1d
        v_to_subtree_partial_likelihoods[va] = partial_likelihoods
    return v_to_subtree_partial_likelihoods


def _forward(T, edge_to_P, root, v_to_subtree_partial_likelihoods):
    """
    Forward pass.

    Return a map from node to posterior state distribution.

    """
    root_partial_likelihoods = v_to_subtree_partial_likelihoods[root]
    n = root_partial_likelihoods.shape[0]
    v_to_posterior_distn1d = {root : normalized(root_partial_likelihoods)}
    for edge in nx.bfs_edges(T, root):
        va, vb = edge
        P = edge_to_P[edge]

        # For each parent state, compute the distribution over child states.
        distn1d = make_distn1d(n)
        va_distn1d = v_to_posterior_distn1d[va]
        vb_partial_likelihoods = v_to_subtree_partial_likelihoods[vb]
        for sa, pa in enumerate(va_distn1d):
            distn1d += pa * normalized(P[sa] * vb_partial_likelihoods)
        v_to_posterior_distn1d[vb] = distn1d

    return v_to_posterior_distn1d


def _forward_edges(T, edge_to_P, root,
        v_to_subtree_partial_likelihoods):
    """
    Forward pass.

    Return a map from edge to joint state distribution.
    Also calculate the posterior state distributions at nodes,
    but do not return them.

    """
    root_partial_likelihoods = v_to_subtree_partial_likelihoods[root]
    n = root_partial_likelihoods.shape[0]
    v_to_posterior_distn1d = {root : normalized(root_partial_likelihoods)}
    edge_to_J = dict((edge, make_distn2d(n)) for edge in T.edges())
    for edge in nx.bfs_edges(T, root):
        va, vb = edge
        P = edge_to_P[edge]
        J = edge_to_J[edge]

        # For each parent state, compute the distribution over child states.
        distn1d = make_distn1d(n)
        va_distn1d = v_to_posterior_distn1d[va]
        vb_partial_likelihoods = v_to_subtree_partial_likelihoods[vb]
        for sa, pa in enumerate(va_distn1d):

            # Construct conditional transition probabilities.
            sb_distn = normalized(P[sa] * vb_partial_likelihoods)

            # Define the row of the joint distribution.
            J[sa] = pa * sb_distn

            # Add to the marginal distribution over states at the vb node.
            distn1d += J[sa]

        v_to_posterior_distn1d[vb] = distn1d

    return edge_to_J


def _get_partial_likelihood(edge_to_P,
        v_to_subtree_partial_likelihoods, va, vbs, s):
    """
    edge_to_P : dict
        A map from directed edges of the tree graph
        to 2d float ndarrays representing state transition probability.
    v_to_subtree_partial_likelihoods : map a node to dict of partial likelihoods
    va : node under consideration
    vbs : child nodes of va
    s : state under consideration
    """
    probs = []
    for vb in vbs:
        P = edge_to_P[va, vb]
        p = np.dot(P[s], v_to_subtree_partial_likelihoods[vb])
        probs.append(p)
    if np.any(probs):
        return np.prod(probs)
    else:
        return None


# function suite for testing
fnsuite = (get_lhood, get_node_to_distn1d, get_edge_to_distn2d)

