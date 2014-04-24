"""
Test Markov chain algorithms to compute likelihoods and distributions on trees.

This module compares dynamic programming implementations against
brute force implementations.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import (run_module_suite, assert_,
        assert_equal, assert_allclose, assert_array_less, assert_array_equal)

import npmctree
from npmctree.puzzles import (
        gen_random_fset_systems, gen_random_infeasible_fset_systems,
        gen_random_lmap_systems, gen_random_infeasible_lmap_systems)
from npmctree import brute_fset_feas, dynamic_fset_feas
from npmctree import brute_fset_lhood, dynamic_fset_lhood
from npmctree import brute_lmap_lhood, dynamic_lmap_lhood


# function suites for testing
fset_suites = (
        brute_fset_feas.fnsuite,
        brute_fset_lhood.fnsuite,
        dynamic_fset_feas.fnsuite,
        dynamic_fset_lhood.fnsuite)
lmap_suites = (
        brute_lmap_lhood.fnsuite,
        dynamic_lmap_lhood.fnsuite)
all_suites = fset_suites + lmap_suites


def test_infeasible_fset_systems():
    # Test systems that are structurally infeasible.
    for args in gen_random_infeasible_fset_systems():
        T, e_to_P, r, r_prior, node_feas = args

        for f_overall, f_node, f_edge in fset_suites:

            # overall likelihood or feasibility
            scalar_summary = f_overall(*args)
            assert_(not scalar_summary)

            # state distributions or feasible sets at nodes
            node_info = f_node(*args)
            for d in node_info.values():
                assert_(not d.any())

            # joint state distributions at edge endpoints
            edge_info = f_edge(*args)
            for edge in T.edges():
                assert_(not edge_info[edge].any())


def test_infeasible_lmap_systems():
    # Test systems that are structurally infeasible.
    for args in gen_random_infeasible_lmap_systems():
        T, e_to_P, r, r_prior, node_data = args

        for f_overall, f_node, f_edge in all_suites:

            # overall likelihood or feasibility
            scalar_summary = f_overall(*args)
            assert_(not scalar_summary)

            # state distributions or feasible sets at nodes
            node_info = f_node(*args)
            for d in node_info.values():
                assert_(not d.any())

            # joint state distributions at edge endpoints
            edge_info = f_edge(*args)
            for edge in T.edges():
                assert_(not edge_info[edge].any())


def test_complete_fset_density():
    # Test the special case of a completely dense system.
    pzero = 0
    for args in gen_random_fset_systems(pzero):
        T, e_to_P, r, r_prior, node_feas = args

        for feas_suite, lhood_suite in (
                (dynamic_fset_feas.fnsuite, dynamic_fset_lhood.fnsuite),
                (brute_fset_feas.fnsuite, brute_fset_lhood.fnsuite),
                ):
            f_feas, f_node_to_fset, f_edge_to_nxfset = feas_suite
            f_lhood, f_node_to_distn, f_edge_to_nxdistn = lhood_suite

            # Check overall likelihood and feasibility.
            assert_allclose(f_lhood(*args), 1)
            assert_equal(f_feas(*args), True)

            # Check node and edge distributions and feasibility.
            for f_node, f_edge in (
                    (f_node_to_fset, f_edge_to_nxfset),
                    (f_node_to_distn, f_edge_to_nxdistn)):

                # node info
                d = f_node(*args)
                for v in set(node_feas):
                    assert_equal(d[v].astype(bool), node_feas[v].astype(bool))

                # edge info
                d = f_edge(*args)
                for edge in T.edges():
                    observed_edges = d[edge].astype(bool)
                    desired_edges = e_to_P[edge].astype(bool)
                    assert_array_equal(observed_edges, desired_edges)


def test_complete_lmap_density():
    # Test the special case of a completely dense system.
    pzero = 0
    for args in gen_random_lmap_systems(pzero):
        T, e_to_P, r, r_prior, node_data = args

        # Define the set of nodes.
        nodes = set(node_data)

        # Convert some of the args to use feasibility data
        # instead of likelihood data.
        r_prior_0 = np.array(r_prior, dtype=bool)
        node_data_0 = dict(
                (v, np.array(x, dtype=bool)) for v, x in node_data.items())
        e_to_P_0 = dict(
                (v, np.array(x, dtype=bool)) for v, x in e_to_P.items())
        args_0 = (T, e_to_P_0, r, r_prior_0, node_data_0)

        # The intermediate complicatedness suite
        # requires feasibility data but allows continuous prior distribution
        # and transition probabilities.
        args_1 = (T, e_to_P, r, r_prior, node_data_0)

        for feas_suite, lhood_suite, lmap_suite in (
                (
                    dynamic_fset_feas.fnsuite,
                    dynamic_fset_lhood.fnsuite,
                    dynamic_lmap_lhood.fnsuite),
                (
                    brute_fset_feas.fnsuite,
                    brute_fset_lhood.fnsuite,
                    brute_lmap_lhood.fnsuite),
                ):
            f_feas, f_node_to_fset, f_edge_to_nxfset = feas_suite
            f_lhood, f_node_to_distn, f_edge_to_nxdistn = lhood_suite
            f_lmap_overall, f_lmap_node, f_lmap_edge = lmap_suite

            # Check overall likelihood and feasibility.
            assert_array_less(0, f_lmap_overall(*args))
            assert_array_less(f_lmap_overall(*args), 1)
            assert_allclose(f_lhood(*args_1), 1)
            assert_equal(f_feas(*args_0), True)

            # Check node and edge distributions and feasibility.
            for f_node, f_edge in (
                    (f_lmap_node, f_lmap_edge),
                    (f_node_to_fset, f_edge_to_nxfset),
                    (f_node_to_distn, f_edge_to_nxdistn)):

                # node info
                d = f_node(*args)
                for v in nodes:
                    assert_equal(set(d[v]), set(node_data[v]))

                # edge info
                d = f_edge(*args)
                for edge in T.edges():
                    observed_edges = set(d[edge].edges())
                    desired_edges = set(e_to_P[edge].edges())
                    assert_equal(observed_edges, desired_edges)


def test_fset_dynamic_vs_brute():
    # Compare dynamic programming vs. summing over all histories.
    pzero = 0.2
    for args in gen_random_fset_systems(pzero):
        T, e_to_P, r, r_prior, node_feas = args

        # likelihood
        dynamic = dynamic_fset_lhood.get_lhood(*args)
        brute = brute_fset_lhood.get_lhood_brute(*args)
        if dynamic is None or brute is None:
            assert_equal(dynamic, None)
            assert_equal(brute, None)
        else:
            assert_allclose(dynamic, brute)

        # feasibility
        dynamic = dynamic_fset_feas.get_feas(*args)
        brute = brute_fset_feas.get_feas_brute(*args)
        assert_equal(dynamic, brute)

        # state distributions at nodes
        dynamic = dynamic_fset_lhood.get_node_to_distn1d(*args)
        brute = brute_fset_lhood.get_node_to_distn1d_brute(*args)
        for v in set(node_feas):
            assert_allclose(dynamic[v], brute[v])

        # state feasibility at nodes
        dynamic = dynamic_fset_feas.get_node_to_fvec1d(*args)
        brute = brute_fset_feas.get_node_to_fvec1d_brute(*args)
        for v in set(node_feas):
            assert_equal(dynamic[v], brute[v])

        # joint state distributions at edge endpoints
        dynamic = dynamic_fset_lhood.get_edge_to_distn2d(*args)
        brute = brute_fset_lhood.get_edge_to_distn2d_brute(*args)
        for edge in T.edges():
            assert_allclose(dynamic[edge], brute[edge])

        # joint state feasibility at edge endpoints
        dynamic = dynamic_fset_feas.get_edge_to_fvec2d(*args)
        brute = brute_fset_feas.get_edge_to_fvec2d_brute(*args)
        for edge in T.edges():
            dynamic_edges = dynamic[edge].astype(bool)
            brute_edges = brute[edge].astype(bool)
            assert_array_equal(dynamic_edges, brute_edges)


def test_lmap_dynamic_vs_brute():
    # Compare dynamic programming vs. summing over all histories.
    pzero = 0.2
    for args in gen_random_lmap_systems(pzero):
        T, e_to_P, r, r_prior, node_lmap = args

        # likelihood
        dynamic = dynamic_lmap_lhood.get_lhood(*args)
        brute = brute_lmap_lhood.get_lhood_brute(*args)
        if dynamic is None or brute is None:
            assert_equal(dynamic, None)
            assert_equal(brute, None)
        else:
            assert_allclose(dynamic, brute)

        # state distributions at nodes
        dynamic = dynamic_lmap_lhood.get_node_to_distn1d(*args)
        brute = brute_lmap_lhood.get_node_to_distn1d_brute(*args)
        for v in set(node_lmap):
            assert_allclose(dynamic[v], brute[v])

        # joint state distributions at edge endpoints
        dynamic = dynamic_lmap_lhood.get_edge_to_distn2d(*args)
        brute = brute_lmap_lhood.get_edge_to_distn2d_brute(*args)
        for edge in T.edges():
            assert_allclose(dynamic[edge], brute[edge])

        # get simplified data without subtlety in the observations
        simple_node_data = dict(
                (v, d.astype(bool).astype(float)) for v, d in node_lmap.items())

        simple_args = (T, e_to_P, r, r_prior, simple_node_data)

        # simplified data likelihood
        dynamic_fset = dynamic_fset_lhood.get_lhood(*simple_args)
        brute_fset = brute_fset_lhood.get_lhood_brute(*simple_args)
        dynamic_lmap = dynamic_lmap_lhood.get_lhood(*simple_args)
        brute_lmap = brute_lmap_lhood.get_lhood_brute(*simple_args)
        if None in (dynamic_fset, brute_fset, dynamic_lmap, brute_lmap):
            assert_equal(dynamic_fset, None)
            assert_equal(brute_fset, None)
            assert_equal(dynamic_lmap, None)
            assert_equal(brute_lmap, None)
        else:
            assert_allclose(dynamic_fset, brute_fset)
            assert_allclose(dynamic_lmap, brute_fset)
            assert_allclose(brute_lmap, brute_fset)

        # simplified data state distributions at nodes
        dynamic_fset = dynamic_fset_lhood.get_node_to_distn1d(*simple_args)
        brute_fset = brute_fset_lhood.get_node_to_distn1d_brute(*simple_args)
        dynamic_lmap = dynamic_lmap_lhood.get_node_to_distn1d(*simple_args)
        brute_lmap = brute_lmap_lhood.get_node_to_distn1d_brute(*simple_args)
        for v in set(simple_node_data):
            assert_allclose(dynamic_fset[v], brute_fset[v])
            assert_allclose(dynamic_lmap[v], brute_fset[v])
            assert_allclose(brute_lmap[v], brute_fset[v])

        # simplified data joint state distributions at edge endpoints
        dynamic_fset = dynamic_fset_lhood.get_edge_to_distn2d(*simple_args)
        brute_fset = brute_fset_lhood.get_edge_to_distn2d_brute(*simple_args)
        dynamic_lmap = dynamic_lmap_lhood.get_edge_to_distn2d(*simple_args)
        brute_lmap = brute_lmap_lhood.get_edge_to_distn2d_brute(*simple_args)
        for edge in T.edges():
            assert_allclose(dynamic_fset[edge], brute_fset[edge])
            assert_allclose(dynamic_lmap[edge], brute_fset[edge])
            assert_allclose(brute_lmap[edge], brute_fset[edge])

