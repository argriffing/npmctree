"""
Speed up likelihood calculations.

This is a newer and more focused version of the pyfelscore package
without the application-specific functions.
This Cython module should be built automatically
by the setup.py infrastructure, so the end user does not need to invoke
any special command or know anything about Cython.
But because the intermediate .c file will not be included in the repo,
the end user will need to have Cython.

"""

#TODO what are the commands to check for full vs. partial Cythonization?

#TODO check if this line can be removed
from cython.view cimport array as cvarray

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport log, exp, sqrt

#TODO check if this line can be removed
cnp.import_array()


# Use fused types to support both 32 bit and 64 bit sparse matrix indices.
# The following mailing list question has some doubt about how well this works.
# https://mail.python.org/pipermail/cython-devel/2014-March/004002.html
ctypedef fused idx_t:
    cnp.int32_t
    cnp.int64_t


__all__ = ['esd_site_first_pass']


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def esd_site_first_pass(
        idx_t[:] csr_indices,
        idx_t[:] csr_indptr,
        cnp.float_t[:, :, :] trans, # (nnodes-1, nstates, nstates)
        cnp.float_t[:, :] data, # (nnodes, nstates)
        cnp.float_t[:, :] lhood, # (nnodes, nstates)
        ):
    """
    Compute partial likelihoods for single site.

    The esd abbreviation refers to 'edge-specific dense' transition matrices.
    Nodes of the tree are indexed according to a topological sort,
    starting at the root.
    Edges of the tree are indexed such that if x and y are node indices,
    then the directed edge (x, y) has index y-1.
    Note that the number of edges in a tree is one fewer
    than the number of nodes in the tree.

    Note that csr_indices and csr_indptr can be computed by using
    networkx to construct a csr sparse matrix by calling
    nx.to_scipy_sparse_matrix function and passing a node ordering
    constructed using nx.topological_sort.
    Because we want to accept csr_indices and csr_indptr arrays
    from scipy.sparse.csr_matrix objects, we must allow both 32 bit and 64 bit
    integer types.

    Parameters
    ----------
    csr_indices : ndarray view
        Part of the Compressed Sparse Row format of the tree structure.
    csr_indptr : ndarray view
        Part of the Compressed Sparse Row format of the tree structure.
    trans : ndarray view
        For each edge, a dense transition matrix.
    data : ndarray view
        For each node, the emission likelihood for each state.
    lhood : ndarray view
        For each node, the partial likelihood for each state.
        This array is for output only.

    """
    return 0
    """
    cdef int nnodes = state_mask.shape[0]
    cdef int nstates = state_mask.shape[1]
    cdef int node_ind_start, node_ind_stop
    cdef double multiplicative_prob
    cdef double additive_prob
    cdef int na, nb
    cdef int i, j
    for i in range(nnodes):

        # Define the current node.
        na = (nnodes - 1) - i
        node_ind_start = tree_csr_indptr[na]
        node_ind_stop = tree_csr_indptr[na+1]

        # Compute the subtree probability for each possible state.
        for sa in range(nstates):
            subtree_probability[na, sa] = 0
            if not state_mask[na, sa]:
                continue
            multiplicative_prob = 1
            for j in range(node_ind_start, node_ind_stop):
                nb = tree_csr_indices[j]
                additive_prob = 0 
                for sb in range(nstates):
                    if state_mask[nb, sb]:
                        additive_prob += (
                                esd_transitions[nb, sa, sb] *
                                subtree_probability[nb, sb])
                multiplicative_prob *= additive_prob
            subtree_probability[na, sa] = multiplicative_prob

    return 0
    """

