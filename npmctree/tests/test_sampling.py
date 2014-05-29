"""
Test joint state sampling on Markov chains on NetworkX tree graphs.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict
import math

import networkx as nx

import numpy as np
from numpy.testing import (run_module_suite, assert_equal, assert_allclose,
        assert_array_less, assert_array_equal, decorators)

import npmctree
from npmctree.sampling import (
        sample_history, sample_histories)
from npmctree.util import make_distn2d, normalized
from npmctree.puzzles import gen_random_lmap_systems
from npmctree.history import get_history_lhood
from npmctree.dynamic_fset_feas import get_feas
from npmctree.dynamic_lmap_lhood import get_lhood, get_edge_to_distn2d


def _sampling_helper(sqrt_nsamples):

    # Define an arbitrary tree.
    # The nodes 'a', 'b' are internal nodes.
    # The nodes 'c', 'd', 'e', 'f' are tip nodes.
    G = nx.Graph()
    G.add_edge('a', 'b')
    G.add_edge('a', 'c')
    G.add_edge('a', 'd')
    G.add_edge('b', 'e')
    G.add_edge('b', 'f')

    # Define a symmetric 3-state path-like state transition matrix.
    P = np.array([
        [0.75, 0.25, 0.00],
        [0.25, 0.50, 0.25],
        [0.00, 0.25, 0.75],
        ], dtype=float)

    # Define an informative distribution at the root.
    root_prior_distn1d = np.array([0.6, 0.4, 0.0], dtype=float)

    # Define data state restrictions at nodes.
    # Use some arbitrary emission likelihoods.
    node_to_data_lmap_traditional = {
            'a' : np.array([0.8, 0.8, 0.1], dtype=float),
            'b' : np.array([0.1, 0.8, 0.8], dtype=float),
            'c' : np.array([0.1, 0.0, 0.0], dtype=float),
            'd' : np.array([0.0, 0.2, 0.0], dtype=float),
            'e' : np.array([0.0, 0.3, 0.0], dtype=float),
            'f' : np.array([0.0, 0.0, 0.4], dtype=float),
            }
    node_to_data_lmap_internal_constraint = {
            'a' : np.array([0.8, 0.8, 0.0], dtype=float),
            'b' : np.array([0.0, 0.8, 0.8], dtype=float),
            'c' : np.array([0.1, 0.0, 0.0], dtype=float),
            'd' : np.array([0.0, 0.2, 0.0], dtype=float),
            'e' : np.array([0.0, 0.3, 0.0], dtype=float),
            'f' : np.array([0.0, 0.0, 0.4], dtype=float),
            }

    # Three states.
    n = 3

    # Try a couple of state restrictions.
    for node_to_data_lmap in (
            node_to_data_lmap_traditional,
            node_to_data_lmap_internal_constraint):

        # Try a couple of roots.
        for root in ('a', 'c'):

            # Get the rooted tree.
            T = nx.dfs_tree(G, root)
            edge_to_P = dict((edge, P) for edge in T.edges())

            # Compute the exact joint distributions at edges.
            edge_to_J_exact = get_edge_to_distn2d(
                    T, edge_to_P, root, root_prior_distn1d, node_to_data_lmap)

            # Sample a bunch of joint states.
            nsamples = sqrt_nsamples * sqrt_nsamples
            edge_to_J_approx = dict(
                    (edge, make_distn2d(n)) for edge in T.edges())
            for node_to_state in sample_histories(T, edge_to_P, root,
                    root_prior_distn1d, node_to_data_lmap, nsamples):
                for tree_edge in T.edges():
                    va, vb = tree_edge
                    sa = node_to_state[va]
                    sb = node_to_state[vb]
                    edge_to_J_approx[tree_edge][sa, sb] += 1
            edge_to_nx_distn = {}
            for edge in T.edges():
                edge_to_J_approx[edge] = normalized(edge_to_J_approx[edge])

            # Compare exact vs. approx joint state distributions on edges.
            # These should be similar up to finite sampling error.
            zstats = []
            for edge in T.edges():
                J_exact = edge_to_J_exact[edge]
                J_approx = edge_to_J_approx[edge]

                # Check that for each edge
                # the set of nonzero joint state probabilities is the same.
                # Technically this may not be true because of sampling error,
                # but we will assume that it is required.
                A = J_exact
                B = J_approx
                assert_array_equal(A != 0, B != 0)

                # Compute a z statistic for the error of each edge proportion.
                for sa in range(n):
                    for sb in range(n):
                        if A[sa, sb] and B[sa, sb]:
                            p_observed = A[sa, sb]
                            p_exact = B[sa, sb]
                            num = sqrt_nsamples * (p_observed - p_exact)
                            den = math.sqrt(p_exact * (1 - p_exact))
                            z = num / den
                            zstats.append(z)

            # The z statistics should be smaller than a few standard deviations.
            assert_array_less(np.absolute(z), 4)


@decorators.slow
def test_sampling_slow():
    sqrt_nsamples = 400
    _sampling_helper(sqrt_nsamples)


def test_sampling_fast():
    sqrt_nsamples = 10
    _sampling_helper(sqrt_nsamples)


def test_puzzles():
    # Check for raised exceptions but do not check the answers.
    pzero = 0.2
    for args in gen_random_lmap_systems(pzero):
        T, e_to_P, r, r_prior, node_data = args
        n = r_prior.shape[0]
        node_to_state = sample_history(*args)

        # get feasibility
        r_prior_fvec1d = np.array(r_prior, dtype=bool)
        node_data_fvec1d = dict(
                (v, np.array(x, dtype=bool)) for v, x in node_data.items())
        e_to_A = dict(
                (v, np.array(x, dtype=bool)) for v, x in e_to_P.items())
        feas = get_feas(T, e_to_A, r, r_prior_fvec1d, node_data_fvec1d)

        # check quantities more complicated than feasibility
        if node_to_state:
            if not feas:
                raise Exception('sampled a node to state map '
                        'for an infeasible problem')
            else:
                # sampled a node to state map for a feasible problem
                data = dict()
                for v, s in node_to_state.items():
                    data[v] = np.zeros(n, dtype=float)
                    data[v][s] = 1
                lk = get_lhood(T, e_to_P, r, r_prior, data)
                hlk = get_history_lhood(T, e_to_P, r, r_prior, node_to_state)
                assert_allclose(lk, hlk)
        else:
            if feas:
                raise Exception('failed to sample a node to state map '
                        'for a feasible problem')
            else:
                # failed to sample a node to state map for infeasible problem
                pass

