"""
Test algorithms that compute NetworkX Markov tree feasibility.

"""
from __future__ import division, print_function, absolute_import

import itertools

import numpy as np
import networkx as nx
from numpy.testing import (run_module_suite, TestCase,
        decorators, assert_, assert_equal)

import npmctree
from npmctree.dynamic_fset_feas import get_feas, get_node_to_fvec1d


class Test_ShortPathFeasibility(TestCase):

    def setUp(self):

        # Define the tree.
        # It is just a path on three nodes.
        G = nx.Graph()
        G.add_path(['a', 'b', 'c'])

        # Define the state transition adjacency matrix.
        # It is a bidirectional path on three states,
        # where self-transitions are allowed.
        nstates = 3
        A = np.array([
            [1, 1, 0],
            [1, 1, 1],
            [0, 1, 1],
            ], dtype=bool)

        # Store the setup.
        self.G = G
        self.A = A
        self.nstates = nstates

    def test_unrestricted(self):
        # For each root position, each state should be allowed at each node.

        # Define the uninformative set of feasible states.
        uninformative_fvec1d = np.ones(self.nstates, dtype=bool)

        # Use an uninformative prior state distribution at the root.
        root_prior_fvec1d = uninformative_fvec1d

        # The data does not restrict the set of states.
        node_to_data_fvec1d = {
                'a' : uninformative_fvec1d,
                'b' : uninformative_fvec1d,
                'c' : uninformative_fvec1d,
                }

        # Check each possible root position.
        for root in self.G:
            T = nx.dfs_tree(self.G, root)
            edge_to_adjacency = dict((edge, self.A) for edge in T.edges())

            # Assert that the combination of input parameters is feasible.
            feas = get_feas(
                    T, edge_to_adjacency, root,
                    root_prior_fvec1d, node_to_data_fvec1d)
            assert_(feas)

            # Assert that the posterior feasibility is the same
            # as the feasibility imposed by the data.
            v_to_fvec1d = get_node_to_fvec1d(
                    T, edge_to_adjacency, root,
                    root_prior_fvec1d, node_to_data_fvec1d)
            assert_equal(v_to_fvec1d, node_to_data_fvec1d)

    def test_restricted(self):
        # The data imposes restrictions that imply further restrictions.

        # Define the uninformative set of feasible states.
        uninformative_fvec1d = np.ones(self.nstates, dtype=bool)

        # Use an uninformative prior state distribution at the root.
        root_prior_fvec1d = uninformative_fvec1d

        # Restrict the two endpoint states to 'a' and 'c' respectively.
        # Together with the tree graph and the state transition adjacency graph
        # this will imply that the middle node must have state 'b'.
        node_to_data_fvec1d = {
                'a' : np.array([0, 0, 1], dtype=bool),
                'b' : np.array([1, 1, 1], dtype=bool),
                'c' : np.array([1, 0, 0], dtype=bool),
                }

        # Regardless of the root, the details of the tree topology and the
        # state transition matrix imply the following map from nodes
        # to feasible state sets.
        node_to_implied_fvec1d = {
                'a' : np.array([0, 0, 1], dtype=bool),
                'b' : np.array([0, 1, 0], dtype=bool),
                'c' : np.array([1, 0, 0], dtype=bool),
                }

        # Check each possible root position.
        for root in self.G:
            T = nx.dfs_tree(self.G, root)
            edge_to_adjacency = dict((edge, self.A) for edge in T.edges())
            v_to_fvec1d = get_node_to_fvec1d(
                    T, edge_to_adjacency, root,
                    root_prior_fvec1d, node_to_data_fvec1d)
            assert_equal(v_to_fvec1d, node_to_implied_fvec1d)


class Test_LongPathFeasibility(TestCase):

    def setUp(self):

        # Define the tree.
        # It is a path on three nodes.
        G = nx.Graph()
        G.add_path(['a', 'b', 'c'])

        # Define the state transition adjacency matrix.
        # It is a bidirectional path on four states,
        # where self-transitions are allowed.
        # Note that the two extreme states cannot both occur on the
        # particular tree that we have chosen.
        nstates = 4
        A = np.array([
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            ], dtype=bool)

        # Store the setup.
        self.G = G
        self.A = A
        self.nstates = nstates

    def test_long_path_infeasibility(self):
        
        nodes = set(self.G)
        for a, b, c in itertools.permutations(nodes):
            for root in self.G:

                # Define the uninformative set of feasible states.
                uninformative_fvec1d = np.ones(self.nstates, dtype=bool)

                # Use an uninformative prior state distribution at the root.
                root_prior_fvec1d = uninformative_fvec1d

                # Let two of the states be endpoints of the transition path,
                # and let the other state be anything.
                # No state assignment will work for this setup.
                node_to_data_fvec1d = {
                        a : np.array([1, 0, 0, 0], dtype=bool),
                        b : np.array([1, 1, 1, 1], dtype=bool),
                        c : np.array([0, 0, 0, 1], dtype=bool),
                        }

                # This dict represents an infeasible combination
                # of prior and data and tree shape and transition matrix.
                node_to_implied_fvec1d = {
                        a : np.zeros(self.nstates, dtype=bool),
                        b : np.zeros(self.nstates, dtype=bool),
                        c : np.zeros(self.nstates, dtype=bool),
                        }

                T = nx.dfs_tree(self.G, root)
                edge_to_adjacency = dict((edge, self.A) for edge in T.edges())
                v_to_fvec1d = get_node_to_fvec1d(
                        T, edge_to_adjacency, root,
                        root_prior_fvec1d, node_to_data_fvec1d)
                assert_equal(v_to_fvec1d, node_to_implied_fvec1d)
