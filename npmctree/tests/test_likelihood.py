"""
Test Markov chain algorithms to compute likelihoods and distributions on trees.

Test the likelihood calculation for a single specific example.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx
from numpy.testing import (run_module_suite, TestCase,
        decorators, assert_, assert_allclose)

import npmctree
from npmctree import dynamic_fset_lhood, brute_fset_lhood
from npmctree import dynamic_lmap_lhood, brute_lmap_lhood
from npmctree import cy_dynamic_lmap_lhood


def test_dynamic_history_likelihood():
    # In this test the history is completely specified.

    G = nx.Graph()
    G.add_edge('a', 'b')
    G.add_edge('a', 'c')
    G.add_edge('a', 'd')

    # Define the rooted tree.
    root = 'a'
    T = nx.dfs_tree(G, root)

    # The data completely restricts the set of states.
    node_to_data_fvec1d = {
            'a' : np.array([1, 0, 0, 0], dtype=bool),
            'b' : np.array([1, 0, 0, 0], dtype=bool),
            'c' : np.array([1, 0, 0, 0], dtype=bool),
            'd' : np.array([1, 0, 0, 0], dtype=bool),
            }

    # The data completely restricts the set of states and includes likelihood.
    node_to_data_lmap = {
            'a' : np.array([0.1, 0.0, 0.0, 0.0], dtype=float),
            'b' : np.array([0.2, 0.0, 0.0, 0.0], dtype=float),
            'c' : np.array([0.3, 0.0, 0.0, 0.0], dtype=float),
            'd' : np.array([0.4, 0.0, 0.0, 0.0], dtype=float),
            }

    # The root prior distribution is informative.
    root_prior_distn1d = np.array([0.5, 0.5, 0.0, 0.0], float)

    # Define the transition matrix.
    P = np.array([
        [0.50, 0.25, 0.25, 0.00],
        [0.25, 0.50, 0.25, 0.00],
        [0.25, 0.25, 0.50, 0.00],
        [0.00, 0.00, 0.00, 1.00],
        ], float)

    # Associate each edge with the transition matrix.
    edge_to_P = dict((edge, P) for edge in T.edges())

    # The likelihood is simple in this case.
    desired_fset_likelihood = (0.5 ** 4)
    desired_lmap_likelihood = (0.5 ** 4) * (0.1 * 0.2 * 0.3 * 0.4)

    # Compare to brute fset likelihood.
    actual_likelihood = brute_fset_lhood.get_lhood_brute(T, edge_to_P, root,
            root_prior_distn1d, node_to_data_fvec1d)
    assert_allclose(actual_likelihood, desired_fset_likelihood)

    # Compare to dynamic fset likelihood.
    actual_likelihood = dynamic_fset_lhood.get_lhood(T, edge_to_P, root,
            root_prior_distn1d, node_to_data_fvec1d)
    assert_allclose(actual_likelihood, desired_fset_likelihood)

    # Compare to brute lmap likelihood.
    actual_likelihood = brute_lmap_lhood.get_lhood_brute(T, edge_to_P, root,
            root_prior_distn1d, node_to_data_lmap)
    assert_allclose(actual_likelihood, desired_lmap_likelihood)

    # Compare to dynamic lmap likelihood.
    actual_likelihood = dynamic_lmap_lhood.get_lhood(T, edge_to_P, root,
            root_prior_distn1d, node_to_data_lmap)
    assert_allclose(actual_likelihood, desired_lmap_likelihood)

    # Compare to cythonized dynamic lmap likelihood.
    actual_likelihood = cy_dynamic_lmap_lhood.get_lhood(T, edge_to_P, root,
            root_prior_distn1d, node_to_data_lmap)
    assert_allclose(actual_likelihood, desired_lmap_likelihood)

