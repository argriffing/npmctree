"""
Joint state sampling algorithm for a Markov chain on a NetworkX tree graph.

"""
from __future__ import division, print_function, absolute_import

import random

import numpy as np
import networkx as nx

from npmctree import dynamic_fset_lhood, dynamic_lmap_lhood
from .util import normalized

__all__ = [
        'sample_history',
        'sample_histories',
        ]


def sample_history(T, edge_to_P, root,
        root_prior_distn1d, node_to_data_lmap):
    """
    Jointly sample states on a tree.
    This is called a history.

    """
    v_to_subtree_partial_likelihoods = dynamic_lmap_lhood._backward(
            T, edge_to_P, root, root_prior_distn1d, node_to_data_lmap)
    node_to_state = _sample_states_preprocessed(T, edge_to_P, root,
            v_to_subtree_partial_likelihoods)
    return node_to_state


def sample_histories(T, edge_to_P, root,
        root_prior_distn1d, node_to_data_lmap, nhistories):
    """
    Sample multiple history.
    Each history is a joint sample of states on the tree.

    """
    v_to_subtree_partial_likelihoods = dynamic_lmap_lhood._backward(
            T, edge_to_P, root, root_prior_distn1d, node_to_data_lmap)
    for i in range(nhistories):
        node_to_state = _sample_states_preprocessed(T, edge_to_P, root,
                v_to_subtree_partial_likelihoods)
        yield node_to_state


def _sample_states_preprocessed(T, edge_to_P, root,
        v_to_subtree_partial_likelihoods):
    """
    Jointly sample states on a tree.

    This variant requires subtree partial likelihoods.

    """
    root_partial_likelihoods = v_to_subtree_partial_likelihoods[root]
    n = root_partial_likelihoods.shape[0]
    if not root_partial_likelihoods.any():
        return None
    distn1d = normalized(root_partial_likelihoods)
    root_state = np.random.choice(range(n), p=distn1d)
    v_to_sampled_state = {root : root_state}
    for edge in nx.bfs_edges(T, root):
        va, vb = edge
        P = edge_to_P[edge]

        # For the relevant parent state,
        # compute an unnormalized distribution over child states.
        sa = v_to_sampled_state[va]

        # Construct conditional transition probabilities.
        sb_weights = P[sa] * v_to_subtree_partial_likelihoods[vb]

        # Sample the state.
        distn1d = normalized(sb_weights)
        v_to_sampled_state[vb] = np.random.choice(range(n), p=distn1d)

    return v_to_sampled_state

