"""
"""
from __future__ import division, print_function, absolute_import

from warnings import warn

from .util import isboolobj


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
    node_to_data_fvec1d : dict
        Map from node to fvec1d of feasible states.
        The feasibility could be interpreted as due to restrictions
        caused by observed data.
"""


@ddec(params=params)
def validated_params(T, edge_to_A, root,
        root_prior_distn1d, node_to_data_fvec1d):
    """
    """
    if not all(isboolobj(d) for d in node_to_data_fvec1d.values()):
        warn('converting data arrays to bool')
        node_to_data_fvec1d = dict(
                (v, d.astype(bool)) for v, d in node_to_data_fvec1d.items())

    return T, edge_to_A, root, root_prior_distn1d, node_to_data_fvec1d
