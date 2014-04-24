"""
This is a small test based on a blinking model puzzle.

The tree is essentially a Markov chain with structure
N0 -- N1 -- N2
and with binary hidden states.
The hidden state transition matrices are each 2x2 matrices with entries 1/2.
The observation likelihoods for the hidden states are more interesting.

"""
from __future__ import division, print_function, absolute_import

import math

import networkx as nx
import numpy as np
from numpy.testing import assert_allclose

from npmctree.dynamic_lmap_lhood import (
        get_node_to_distn1d, get_edge_to_distn2d)
from npmctree.brute_lmap_lhood import (
        get_node_to_distn1d_brute, get_edge_to_distn2d_brute)


def exp_neg(x):
    return math.exp(-x)


def get_blinking_model(omega):
    T = nx.DiGraph()
    T.add_edge('N1', 'N0')
    T.add_edge('N1', 'N2')
    P = np.array([
        [1 - 1/omega, 1/omega],
        [1/omega, 1 - 1/omega],
        ], dtype=float)
    edge_to_P = dict((v, P) for v in T.edges())
    root = 'N1'
    root_prior_distn = np.array([1/2, 1/2], dtype=float)
    node_to_data_lmap = {
            'N0' : np.array([
                    exp_neg(0.25*(2/14)),
                    exp_neg(0.25*(2/14)),
                    ]),
            'N1': np.array([
                    exp_neg(0.85*(2/14) + 0.25*(2/14)),
                    exp_neg(0.85*(2/14) + 0.25*(3/14)),
                    ]),
            'N2' : np.array([
                    exp_neg(1.125*(2/14)),
                    exp_neg(1.125*(3/14)),
                    ]),
            }
    return T, edge_to_P, root, root_prior_distn, node_to_data_lmap


def test_blinking_model_chain():
    for omega in 2, 5:
        args = get_blinking_model(omega)
        T, edge_to_P, root, root_prior_distn, node_to_data_lmap = args

        # posterior transitions on edges
        post_dynamic_map = get_edge_to_distn2d(*args)
        post_brute_map = get_edge_to_distn2d_brute(*args)
        for edge in T.edges():
            va, vb = edge
            post_distn_dynamic = post_dynamic_map[edge]
            post_distn_brute = post_brute_map[edge]
            assert_allclose(post_distn_dynamic, post_distn_brute)

        # posterior distributions at nodes
        post_dynamic_map = get_node_to_distn1d(*args)
        post_brute_map = get_node_to_distn1d_brute(*args)
        for v in T:
            post_distn_dynamic = post_dynamic_map[v]
            post_distn_brute = post_brute_map[v]
            assert_allclose(post_distn_dynamic, post_distn_brute)


