"""
Markov chain algorithms to compute likelihoods and distributions on trees.

The NetworkX digraph representing a sparse transition probability matrix
will be represented by the notation 'P'.
State distributions are represented by sparse dicts.
Joint endpoint state distributions are represented by networkx graphs.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx

from .cyfels import iid_likelihoods

#TODO move this function to the dynamic_lmap_lhood.py module.

__all__ = [
        'get_iid_lhoods',
        #'get_lhood',
        #'get_node_to_distn1d',
        #'get_edge_to_distn2d',
        ]


def get_iid_lhoods(T, edge_to_P, root, root_prior_distn1d, node_to_data_lmaps):
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
    node_to_data_lmaps : sequence of dicts of 1d float ndarrays
        Observed data.
        For each iid site, a dict mapping each node to a 1d array
        giving the observation likelihood for each state.
        This parameter is similar to the sample_histories output.

    Returns
    -------
    lhoods : 1d float array
        Likelihood for each iid site.

    """
    pass

