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
from numpy.testing import assert_equal
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


__all__ = ['assert_tree_csr', 'esd_site_first_pass']


def assert_tree_csr(
        idx_t[:] csr_indices,
        idx_t[:] csr_indptr,
        cnp.int_t nnodes,
        ):
    """
    Assume the node indices are 0..(nnodes-1) and are in toposort preorder.

    """
    # Require at least one node.
    # For example networkx raises an exception if you try to build
    # a csr matrix from a graph without nodes.
    assert_array_less(0, nnodes)

    # Check the conformability of the inputs.
    # Note that the global interpreter lock (gil) should be in effect
    # for this section.
    assert_equal(csr_indices.shape, (nnodes-1,))
    assert_equal(csr_indptr.shape, (nnodes+1,))

    # Check that each indptr element is either a valid index
    # into the indices array or is equal to the length of the indices array.
    assert_array_less(-1, csr_indptr)
    assert_array_less(csr_indptr, nnodes+1)
    assert_array_less(-1, csr_indices)
    assert_array_less(csr_indices, nnodes)

    # Check preorder.
    cdef int j
    cdef idx_t indstart, indstop
    cdef idx_t na, nb
    cdef cnp.int_t visited[:] = cnp.zeros(nnodes, dtype=cnp.int_t)
    cdef cnp.int_t head[:] = cnp.zeros(nnodes, dtype=cnp.int_t)
    cdef cnp.int_t tail[:] = cnp.zeros(nnodes, dtype=cnp.int_t)
    visited[0] = 1
    for na in range(nnodes):
        head[na] = visited[na]
        indstart = csr_indptr[na]
        indstop = csr_indptr[na+1]
        for j in range(indstart, indstop):
            nb = csr_indices[j]
            tail[nb] = visited[nb]
            visited[nb] += 1

    # Check that each node had been visited exactly once.
    # Check that each head node had been visited exactly once.
    # Check that each tail node had not been visited.
    assert_equal(visited, 1)
    assert_equal(head, 1)
    assert_equal(tail, 0)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def esd_site_first_pass(
        idx_t[:] csr_indices,
        idx_t[:] csr_indptr,
        cnp.float_t[:, :, :] trans, # (nnodes-1, nstates, nstates)
        cnp.float_t[:, :] data, # (nnodes, nstates)
        cnp.float_t[:, :] lhood, # (nnodes, nstates)
        cnp.int_t check_csr=1,
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
    # Get the number of nodes and the number of states.
    cdef int nnodes = data.shape[0]
    cdef int nstates = data.shape[1]

    # Check the conformability of the inputs.
    # Note that the global interpreter lock (gil) should be in effect
    # for this section.
    assert_equal(csr_indices.shape, (nnodes-1,))
    assert_equal(csr_indptr.shape, (nnodes+1,))
    assert_equal(trans.shape, (nnodes-1, nstates, nstates))
    assert_equal(lhood.shape, (nnodes, nstates))

    # Check that each indptr element is either a valid index
    # into the indices array or is equal to the length of the indices array.

    # Avoid the interpreter lock.
    with nogil:
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

