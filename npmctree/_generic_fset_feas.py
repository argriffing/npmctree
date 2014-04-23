"""
"""

from warnings import warn

from .util import isboolobj


params = """\
    T : directed networkx tree graph
        Edge and node annotations are ignored.
    edge_to_A : dict
        A map from directed edges of the tree graph
        to 2d bool ndarrays representing state transition feasibility.
    root : hashable
        This is the root node.
        Following networkx convention, this may be anything hashable.
    root_prior_fvec1d : 1d ndarray
        The set of feasible prior root states.
        This may be interpreted as the support of the prior state
        distribution at the root.
    node_to_data_fvec1d : dict
        Map from node to set of feasible states.
        The feasibility could be interpreted as due to restrictions
        caused by observed data.
"""


@ddec(params=params)
def _validated_params(T, edge_to_A, root,
        root_prior_fvec1d, node_to_data_fvec1d):
    """
    """
    if not all(isboolobj(A) for A in edge_to_A.values()):
        warn('converting adjacency matrices to bool')
        edge_to_A = dict((e, A.astype(bool)) for e, A in edge_to_A.items())

    if not isboolobj(root_prior_fvec1d):
        warn('converting root prior feasibility to bool')
        root_prior_fvec1d = root_prior_fvec1d.astype(bool)

    if not all(isboolobj(d) for d in node_to_data_fvec1d.values()):
        warn('converting data arrays to bool')
        node_to_data_fvec1d = dict(
                (v, d.astype(bool)) for v, d in node_to_data_fvec1d.items())

    return T, edge_to_A, root, root_prior_fvec1d, node_to_data_fvec1d
