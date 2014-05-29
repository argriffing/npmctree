"""
Markov chain algorithms to compute likelihoods and distributions on trees.

The NetworkX digraph representing a sparse transition probability matrix
will be represented by the notation 'P'.
State distributions are represented by sparse dicts.
Joint endpoint state distributions are represented by networkx graphs.

"""
from __future__ import division, print_function, absolute_import

import itertools
from collections import defaultdict

import numpy as np
import networkx as nx

from .util import ddec, make_distn1d
from .dynamic_fset_lhood import (
        _get_partial_likelihood, _forward_edges, _forward)
from ._generic_lmap_lhood import params, validated_params
from .cyfels import esd_site_first_pass

__all__ = [
        'get_lhood',
        'get_node_to_distn1d',
        'get_edge_to_distn2d',
        ]

#TODO everything except _backward is copypasted


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
    T, edge_to_P, root, root_prior_distn1d, node_to_data_lmap = args

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
    T, edge_to_P, root, root_prior_distn1d, node_to_data_lmap = args

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
    T, edge_to_P, root, root_prior_distn1d, node_to_data_lmap = args

    v_to_subtree_partial_likelihoods = _backward(*args)
    edge_to_J = _forward_edges(T, edge_to_P, root,
            v_to_subtree_partial_likelihoods)
    return edge_to_J


@ddec(params=params)
def _get_root_lhoods(T, edge_to_P, root,
        root_prior_distn1d, node_to_data_lmap):
    """
    Get the posterior likelihoods at the root, conditional on root state.

    These are also known as partial likelihoods.

    Parameters
    ----------
    {params}

    """
    v_to_subtree_partial_likelihoods = _backward(T, edge_to_P, root,
            root_prior_distn1d, node_to_data_lmap)
    return v_to_subtree_partial_likelihoods[root]


def _wrapped_first_pass(*args):
    return esd_site_first_pass(*args)


@ddec(params=params)
def _backward(T, edge_to_P, root, root_prior_distn1d, node_to_data_lmap):
    """
    This is the first pass of a forward-backward algorithm.

    Parameters
    ----------
    {params}

    """
    # Define a toposort node ordering and a corresponding csr matrix.
    nodes = nx.topological_sort(T, [root])
    node_to_idx = dict((na, i) for i, na in enumerate(nodes))
    m = nx.to_scipy_sparse_matrix(T, nodes)

    # Stack the transition matrices into a single array.
    nnodes = len(nodes)
    nstates = root_prior_distn1d.shape[0]
    trans = np.empty((nnodes-1, nstates, nstates), dtype=float)
    for (na, nb), P in edge_to_P.items():
        edge_idx = node_to_idx[nb] - 1
        trans[edge_idx, :, :] = P

    # Stack the data into a single array.
    data = np.empty((nnodes, nstates), dtype=float)
    for i, na in enumerate(nodes):
        data[i, :] = node_to_data_lmap[na]

    # Compute the partial likelihoods.
    lhood = np.empty((nnodes, nstates), dtype=float)
    validation = 0
    _wrapped_first_pass(m.indices, m.indptr, trans, data, lhood, validation)
    lhood[0, :] *= root_prior_distn1d

    # Convert the output into a dictionary.
    return dict((na, lhood[i, :]) for i, na in enumerate(nodes))


# function suite for testing
fnsuite = (get_lhood, get_node_to_distn1d, get_edge_to_distn2d)

