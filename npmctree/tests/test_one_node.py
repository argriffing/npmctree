"""
Test calculations on a tree with only a single node.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx
import numpy as np
from numpy.testing import assert_equal

import npmctree
from npmctree import dynamic_fset_lhood, brute_fset_lhood
from npmctree import dynamic_lmap_lhood, brute_lmap_lhood
from npmctree import dynamic_fset_feas, brute_fset_feas


def test_tree_with_one_node():

    # define the tree and the root
    T = nx.DiGraph()
    root = 'ROOT'
    T.add_node(root)

    # define the prior state distribution at the root
    n = 4
    prior_distn1d = np.ones(n, dtype=float) / n
    prior_fvec1d = np.ones(n, dtype=bool)

    # define the data
    node_to_data_fvec1d = {root : np.array([False, True, False, True])}
    node_to_data_lmap = {root : np.array([0, 1/8, 0, 3/8])}

    # define the desired posterior state distribution at the root
    desired_fvec1d = np.array([False, True, False, True])
    desired_fset_distn1d = np.array([0, 1/2, 0, 1/2])
    desired_lmap_distn1d = np.array([0, 1/4, 0, 3/4])

    # define extra parameters
    edge_to_P = {}

    # brute feasibility
    actual_fvec1d = brute_fset_feas.get_node_to_fvec1d_brute(
            T, edge_to_P, root, prior_fvec1d, node_to_data_fvec1d)[root]
    assert_equal(actual_fvec1d, desired_fvec1d)

    # dynamic feasibility
    actual_fvec1d = dynamic_fset_feas.get_node_to_fvec1d(
            T, edge_to_P, root, prior_fvec1d, node_to_data_fvec1d)[root]
    assert_equal(actual_fvec1d, desired_fvec1d)

    # brute fset distribution
    actual_fset_distn1d = brute_fset_lhood.get_node_to_distn1d_brute(
            T, edge_to_P, root, prior_distn1d, node_to_data_fvec1d)[root]
    assert_equal(actual_fset_distn1d, desired_fset_distn1d)

    # dynamic fset distribution
    actual_fset_distn1d = dynamic_fset_lhood.get_node_to_distn1d(
            T, edge_to_P, root, prior_distn1d, node_to_data_fvec1d)[root]
    assert_equal(actual_fset_distn1d, desired_fset_distn1d)

    # brute lmap distribution
    actual_lmap_distn1d = brute_lmap_lhood.get_node_to_distn1d_brute(
            T, edge_to_P, root, prior_distn1d, node_to_data_lmap)[root]
    assert_equal(actual_lmap_distn1d, desired_lmap_distn1d)

    # dynamic lmap distribution
    actual_lmap_distn1d = dynamic_lmap_lhood.get_node_to_distn1d(
            T, edge_to_P, root, prior_distn1d, node_to_data_lmap)[root]
    assert_equal(actual_lmap_distn1d, desired_lmap_distn1d)

