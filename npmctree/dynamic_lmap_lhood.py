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

import npmctree
from npmctree.util import ddec, make_distn1d
from npmctree.dynamic_fset_lhood import (
        _get_partial_likelihood, _forward_edges, _forward)

__all__ = [
        'get_lhood',
        'get_node_to_distn1d',
        'get_edge_to_distn2d',
        ]


params = """\
    T : directed networkx tree graph
        Edge and node annotations are ignored.
    edge_to_P : dict
        A map from directed edges of the tree graph
        to networkx graphs representing state transition probability.
    root : hashable
        This is the root node.
        Following networkx convention, this may be anything hashable.
    root_prior_distn : dict
        Prior state distribution at the root.
    node_to_data_lmap : dict
        For each node, a map from state to observation likelihood.
"""


@ddec(params=params)
def get_lhood(T, edge_to_P, root, root_prior_distn1d, node_to_data_lmap):
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
    root_lhoods = _get_root_lhoods(T, edge_to_P, root,
            root_prior_distn1d, node_to_data_lmap)
    if root_lhoods.any():
        return root_lhoods.sum()
    else:
        return None


@ddec(params=params)
def get_node_to_distn1d(T, edge_to_P, root,
        root_prior_distn1d, node_to_data_lmap):
    """
    Get the map from node to state distribution.

    Parameters
    ----------
    {params}

    """
    v_to_subtree_partial_likelihoods = _backward(T, edge_to_P, root,
            root_prior_distn1d, node_to_data_lmap)
    v_to_posterior_distn1d = _forward(T, edge_to_P, root,
            v_to_subtree_partial_likelihoods)
    return v_to_posterior_distn1d


@ddec(params=params)
def get_edge_to_distn2d(T, edge_to_P, root,
        root_prior_distn1d, node_to_data_lmap):
    """
    Get the map from edge to joint state distribution at endpoint nodes.

    Parameters
    ----------
    {params}

    """
    v_to_subtree_partial_likelihoods = _backward(T, edge_to_P, root,
            root_prior_distn1d, node_to_data_lmap)
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


@ddec(params=params)
def _backward(T, edge_to_P, root, root_prior_distn1d, node_to_data_lmap):
    """
    This is the backward pass of a backward-forward algorithm.

    Parameters
    ----------
    {params}

    """
    n = root_prior_distn1d.shape[0]
    v_to_subtree_partial_likelihoods = {}
    for va in nx.topological_sort(T, [root], reverse=True):
        lmap_data = node_to_data_lmap[va]
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
            partial_likelihoods = lmap_data
        if va == root:
            partial_likelihoods *= root_prior_distn1d
        v_to_subtree_partial_likelihoods[va] = partial_likelihoods
    return v_to_subtree_partial_likelihoods


# function suite for testing
fnsuite = (get_lhood, get_node_to_distn1d, get_edge_to_distn2d)

