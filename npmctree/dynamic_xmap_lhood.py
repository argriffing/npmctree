"""
Markov chain algorithms to compute likelihoods and distributions on trees.

The tree is represented by a sparse NetworkX DiGraph.
Transition matrices and finite 1d and 2d distributions
are represented by dense numpy ndarrays.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx

from .util import ddec, make_distn1d
from .dynamic_fset_lhood import (
        _get_partial_likelihood, _forward_edges, _forward)
from ._generic_xmap_lhood import params, validated_params
from .cyfels import iid_likelihoods

__all__ = [
        'get_lhood',
        'get_iid_lhoods',
        'get_node_to_distn1d',
        'get_edge_to_distn2d',
        ]


def get_iid_lhoods(T, edge_to_P, root, root_prior_distn1d, xmaps):
    """
    Get the likelihood of this combination of parameters.

    Parameters
    ----------
    T : directed networkx tree graph
        Edge and node annotations are ignored.
    edge_to_P : dict of 2d float ndarrays
        A map from directed edges of the tree graph
        to 2d float ndarrays representing state transition probabilities.
    root : hashable
        This is the root node.
        Following networkx convention, this may be anything hashable.
    root_prior_distn1d : 1d ndarray
        Prior state distribution at the root.
    xmaps : sequence of dicts
        This is the observed data. Nodes not included in the dict
        are assumed to have completely unobserved state.
        Partial observations are not supported in this xmap-based module.

    Returns
    -------
    lhoods : 1d float array
        Likelihood for each iid site.

    """
    nsites = len(xmaps)
    nstates = root_prior_distn1d.shape[0]

    # Define a toposort node ordering and a corresponding csr matrix.
    nodes = nx.topological_sort(T, [root])
    node_to_idx = dict((na, i) for i, na in enumerate(nodes))
    m = nx.to_scipy_sparse_matrix(T, nodes)

    # Stack the transition matrices into a single array.
    nnodes = len(nodes)
    trans = np.empty((nnodes-1, nstates, nstates), dtype=float)
    for (na, nb), P in edge_to_P.items():
        edge_idx = node_to_idx[nb] - 1
        trans[edge_idx, :, :] = P

    # Stack the data into a single array.
    data = np.empty((nsites, nnodes, nstates), dtype=float)
    for i, xmap in enumerate(xmaps):
        for j, na in enumerate(nodes):
            state = xmap.get(na, None)
            if state is None:
                data[i, j, :] = 1
            else:
                data[i, j, :] = 0
                data[i, j, state] = 1

    # Compute the likelihoods.
    lhoods = np.empty(nsites, dtype=float)
    validation = 0
    iid_likelihoods(m.indices, m.indptr,
        trans, data, root_prior_distn1d, lhoods, validation)

    # Return the dense array that contains the likelihood at each iid site.
    return lhoods


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
    T, edge_to_P, root, root_prior_distn1d, xmap = args

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
    T, edge_to_P, root, root_prior_distn1d, xmap = args

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
    T, edge_to_P, root, root_prior_distn1d, xmap = args

    v_to_subtree_partial_likelihoods = _backward(*args)
    edge_to_J = _forward_edges(T, edge_to_P, root,
            v_to_subtree_partial_likelihoods)
    return edge_to_J


@ddec(params=params)
def _get_root_lhoods(T, edge_to_P, root, root_prior_distn1d, xmap):
    """
    Get the posterior likelihoods at the root, conditional on root state.

    These are also known as partial likelihoods.

    Parameters
    ----------
    {params}

    """
    v_to_subtree_partial_likelihoods = _backward(T, edge_to_P, root,
            root_prior_distn1d, xmap)
    return v_to_subtree_partial_likelihoods[root]


@ddec(params=params)
def _backward(T, edge_to_P, root, root_prior_distn1d, xmap):
    """
    This is the backward pass of a backward-forward algorithm.

    Parameters
    ----------
    {params}

    """
    n = root_prior_distn1d.shape[0]
    v_to_subtree_partial_likelihoods = {}
    for va in nx.topological_sort(T, [root], reverse=True):
        state = xmap.get(va, None)
        if state is None:
            lmap_data = np.ones(n, dtype=float)
        else:
            lmap_data = np.zeros(n, dtype=float)
            lmap_data[state] = 1
        if T[va]:
            vbs = T[va]
        else:
            vbs = set()
        if vbs:
            partial_likelihoods = make_distn1d(n)
            for s, lk_obs in enumerate(lmap_data):
                if lk_obs:
                    prob = _get_partial_likelihood(edge_to_P,
                            v_to_subtree_partial_likelihoods, va, vbs, s)
                    if prob is not None:
                        partial_likelihoods[s] = prob * lk_obs
        else:
            partial_likelihoods = lmap_data.copy()
        if va == root:
            partial_likelihoods *= root_prior_distn1d
        v_to_subtree_partial_likelihoods[va] = partial_likelihoods
    return v_to_subtree_partial_likelihoods


# function suite for testing
fnsuite = (get_lhood, get_node_to_distn1d, get_edge_to_distn2d)

