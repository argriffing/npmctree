"""

"""
from __future__ import division, print_function, absolute_import

import networkx as nx
import numpy as np

from numpy.testing import assert_equal, assert_array_less, assert_raises

import npmctree
from npmctree.cyfels import assert_csr_tree


def _check_csr_non_path_tree(G, root):
    # For this test require at least two nodes.
    assert_array_less(1, len(G))
    nodes = nx.topological_sort(G, [root])


def _graph_to_csr(G, nodes):
    m = nx.to_scipy_sparse_matrix(G, nodes)
    return m.indices, m.indptr


def test_csr_tree_path():


    # Make a directed path which is a kind of tree.
    G = nx.DiGraph()
    root = 'x'
    G.add_edges_from([('x', 'y'), ('y', 'z'), ('z', 'w')])

    # Check a directed path which is a kind of tree.
    nodes = nx.topological_sort(G, [root])
    indices, indptr = _graph_to_csr(G, nodes)
    assert_csr_tree(indices, indptr, len(G))

    # The reversed toposort order should not work.
    nodes = nx.topological_sort(G, [root], reverse=True)
    indices, indptr = _graph_to_csr(G, nodes)
    assert_raises(AssertionError, assert_csr_tree, indices, indptr, len(G))


def test_csr_tree_not_path():

    # Make a tree that branches.
    x, y, z, w = 'xyzw'
    root = 'x'
    G = nx.DiGraph()
    G.add_edges_from([(x, y), (x, z), (x, w)])

    # The toposort order should work.
    nodes = nx.topological_sort(G, [root])
    indices, indptr = _graph_to_csr(G, nodes)
    assert_csr_tree(indices, indptr, len(G))

    # The reversed toposort order should not work.
    nodes = nx.topological_sort(G, [root], reverse=True)
    indices, indptr = _graph_to_csr(G, nodes)
    assert_raises(AssertionError, assert_csr_tree, indices, indptr, len(G))


def test_csr_tree_disconnected():
    # This disconnected graph has the correct number of edges
    # relative to the number of nodes, but it is not a tree.
    x, y, z, u, v = 'xyzuv'
    root = 'x'
    G = nx.DiGraph()
    G.add_edges_from([(x, y), (y, z), (x, z), (u, v)])
    nodes = (x, y, z, u, v)
    indices, indptr = _graph_to_csr(G, nodes)
    assert_raises(AssertionError, assert_csr_tree, indices, indptr, len(G))

