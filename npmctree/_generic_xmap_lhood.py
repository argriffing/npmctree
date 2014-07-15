"""
"""
from __future__ import division, print_function, absolute_import

from .util import ddec


params = """\
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
    node_to_state : dict
        Sparse map from node to state.
        Nodes with unobserved states are not included in this map.
"""


@ddec(params=params)
def validated_params(T, edge_to_P, root, root_prior_distn1d, node_to_state):
    """
    """
    nstates = root_prior_distn1d.shape[0]
    all_nodes = set(T)
    for node, state in node_to_state.items():
        if node not in T:
            raise Exception('node %s was not found in the tree')
        if int(state) != state:
            raise Exception('expected integer state')
        if not (0 <= state < nstates):
            raise Exception('the state is out of bounds')

    return T, edge_to_P, root, root_prior_distn1d, node_to_state
