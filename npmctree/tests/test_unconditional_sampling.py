"""
Test unconditional joint state sampling on Markov chains on trees.

The Markov transition matrices and the tree graphs are both represented by
networkx DiGraphs.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx

import numpy as np
from numpy.testing import assert_equal

import npmctree
from npmctree.util import make_distn1d, make_distn2d
from npmctree.sampling import (
        sample_unconditional_history, sample_unconditional_histories)


def get_single_node_tree():
    G = nx.DiGraph()
    root = 'root'
    G.add_node(root)
    return G, root


def get_path_tree():
    G = nx.DiGraph()
    root = 'a'
    G.add_edge('a', 'b')
    G.add_edge('b', 'c')
    return G, root


def get_three_leaf_tree():
    G = nx.DiGraph()
    root = 'A'
    G.add_edge('A', 'B')
    G.add_edge('A', 'C')
    G.add_edge('A', 'D')
    return G, root


def get_toy_tree():
    G = nx.DiGraph()
    root = 0
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(0, 3)
    G.add_edge(1, 4)
    G.add_edge(1, 5)
    return G, root


def gen_example_trees():
    yield get_single_node_tree()
    yield get_path_tree()
    yield get_three_leaf_tree()
    yield get_toy_tree()



def _smoke_sample_unconditional_history_switch(tree, root):
    # Check immediate switching sampling.
    nstates = 4
    root_state = 2
    target_state = 3
    distn = make_distn1d(nstates)
    distn[root_state] = 1
    P = make_distn2d(nstates)
    P[:, target_state] = 1
    edge_to_P = dict((e, P) for e in tree.edges())
    #
    node_to_state = sample_unconditional_history(tree, edge_to_P, root, distn)
    assert_equal(set(node_to_state), set(tree))
    for node, state in node_to_state.items():
        if node == root:
            assert_equal(state, root_state)
        else:
            assert_equal(state, target_state)


def _smoke_sample_unconditional_histories_switch(tree, root):
    # Check immediate switching sampling.
    nstates = 4
    root_state = 2
    target_state = 3
    distn = make_distn1d(nstates)
    distn[root_state] = 1
    P = make_distn2d(nstates)
    P[:, target_state] = 1
    edge_to_P = dict((e, P) for e in tree.edges())
    #
    nhistories = 3
    for node_to_state in sample_unconditional_histories(
            tree, edge_to_P, root, distn, nhistories):
        assert_equal(set(node_to_state), set(tree))
        for node, state in node_to_state.items():
            if node == root:
                assert_equal(state, root_state)
            else:
                assert_equal(state, target_state)


def _smoke_sample_unconditional_history_identity(tree, root):
    # Check identity sampling.
    nstates = 4
    dominant_state = 2
    distn = make_distn1d(nstates)
    distn[dominant_state] = 1
    P = np.identity(nstates)
    edge_to_P = dict((e, P) for e in tree.edges())
    #
    node_to_state = sample_unconditional_history(tree, edge_to_P, root, distn)
    assert_equal(set(node_to_state), set(tree))
    for node, state in node_to_state.items():
        assert_equal(state, dominant_state)


def _smoke_sample_unconditional_histories_identity(tree, root):
    # Check identity sampling.
    nstates = 4
    dominant_state = 2
    distn = make_distn1d(nstates)
    distn[dominant_state] = 1
    P = np.identity(nstates)
    edge_to_P = dict((e, P) for e in tree.edges())
    #
    nhistories = 3
    for node_to_state in sample_unconditional_histories(
            tree, edge_to_P, root, distn, nhistories):
        assert_equal(set(node_to_state), set(tree))
        for node, state in node_to_state.items():
            assert_equal(state, dominant_state)


def test_smoke_sample_unconditional_history():
    # Sample a history unconditionally.
    for tree, root in gen_example_trees():
        _smoke_sample_unconditional_history_identity(tree, root)
        _smoke_sample_unconditional_history_switch(tree, root)


def test_smoke_sample_unconditional_histories():
    # Sample a history unconditionally.
    for tree, root in gen_example_trees():
        _smoke_sample_unconditional_histories_identity(tree, root)
        _smoke_sample_unconditional_histories_switch(tree, root)

