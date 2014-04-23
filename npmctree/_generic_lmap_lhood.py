"""
"""
from __future__ import division, print_function, absolute_import


params = """\
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
    node_to_data_lmap : dict of 1d float ndarrays
        For each node, a dense map from state to observation likelihood.
"""


@ddec(params=params)
def validated_params(T, edge_to_A, root,
        root_prior_distn1d, node_to_data_lmap):
    """
    This function is not so interesting.
    """
    return T, edge_to_A, root, root_prior_distn1d, node_to_data_lmap

