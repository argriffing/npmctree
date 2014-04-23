"""
Sample random systems for testing.

"""
from __future__ import division, print_function, absolute_import

import random

import numpy as np
import networkx as nx

from .util import normalized


def sample_P(nstates, pzero):
    """
    Return a random transition matrix.
    Some entries may be zero.

    Parameters
    ----------
    nstates : 2d float ndarray
        transition probability matrix
    pzero : float
        probability that any given transition has nonzero probability

    """
    P_weight = np.exp(np.random.randn(nstates, nstates))
    P_mask = np.random.binomial(1, 1-pzero, size=(nstates, nstates))
    P = np.array([normalized(row) for row in P_weight * P_mask])
    return P


def sample_fvec1d(nstates, pzero):
    """
    Parameters
    ----------
    nstates : int
        number of states
    pzero : float
        probability that any given state is infeasible

    """
    v = np.random.binomial(1, 1-pzero, size=nstates)
    return np.array(v, dtype=bool)


def sample_distn1d(nstates, pzero):
    """
    Return a random state distribution as a 1d float ndarray.
    Some entries may be zero.

    Parameters
    ----------
    nstates : int
        number of states
    pzero : float
        probability that any given state has nonzero probability

    """
    v = np.exp(np.random.randn(nstates))
    mask = np.random.binomial(1, 1-pzero, size=nstates)
    return normalized(v * mask)


def sample_lmap(nstates, pzero):
    """
    Parameters
    ----------
    nstates : int
        number of states
    pzero : float
        probability that any given state has nonzero probability

    """
    return sample_distn1d(nstates, pzero)


def sample_data_fvec1ds(nodes, nstates, pzero):
    """
    Return a map from node to feasible state set.

    Parameters
    ----------
    nodes : set
        nodes
    nstates : int
        number of states
    pzero : float
        probability that any given state is infeasible

    """
    return dict((v, sample_fvec1d(nstates, pzero)) for v in nodes)


def sample_data_lmaps(nodes, nstates, pzero):
    """
    Return a map from node to feasible state set.

    Parameters
    ----------
    nodes : set
        nodes
    nstates : int
        number of states
    pzero : float
        probability that any given state is infeasible

    """
    return dict((v, sample_lmap(nstates, pzero)) for v in nodes)


def _sample_single_node_system(pzero, fn_sample_data):
    T = nx.DiGraph()
    root = 'a'
    nstates = 3
    T.add_node(root)
    nodes = set(T)
    root_prior_distn1d = sample_distn1d(nstates, pzero)
    edge_to_P = {}
    node_to_data = fn_sample_data(nodes, nstates, pzero)
    return (T, edge_to_P, root, root_prior_distn1d, node_to_data)


def _sample_four_node_system(pzero_transition, pzero_other, fn_sample_data):
    nstates = 3
    G = nx.Graph()
    G.add_edge('a', 'b')
    G.add_edge('a', 'c')
    G.add_edge('a', 'd')
    nodes = set(G)
    root = random.choice(list(nodes))
    T = nx.dfs_tree(G, root)
    root_prior_distn = sample_distn1d(nstates, pzero_other)
    edge_to_P = {}
    for edge in T.edges():
        edge_to_P[edge] = sample_P(nstates, pzero_transition)
    node_to_data = fn_sample_data(nodes, nstates, pzero_other)
    return (T, edge_to_P, root, root_prior_distn, node_to_data)


def _gen_random_systems(pzero, fn_sample_data, nsystems):
    for i in range(nsystems):
        if random.choice((0, 1)):
            yield _sample_single_node_system(pzero, fn_sample_data)
        else:
            yield _sample_four_node_system(pzero, pzero, fn_sample_data)


def _gen_random_infeasible_systems(fn_sample_data, nsystems):
    pzero = 1
    for i in range(nsystems):
        k = random.randrange(3)
        if k == 0:
            yield _sample_single_node_system(pzero, fn_sample_data)
        elif k == 1:
            yield _sample_four_node_system(pzero, pzero, fn_sample_data)
        else:
            pzero_transition = 1
            pzero_other = 0.2
            yield _sample_four_node_system(
                    pzero_transition, pzero_other, fn_sample_data)


def gen_random_fset_systems(pzero, nsystems=40):
    """
    Sample whole systems for testing likelihood.
    The pzero parameter indicates sparsity.
    Yield (T, edge_to_P, root, root_prior_distn, node_to_data_fset).

    """
    for x in _gen_random_systems(pzero, sample_data_fvec1ds, nsystems):
        yield x


def gen_random_infeasible_fset_systems(nsystems=60):
    for x in _gen_random_infeasible_systems(sample_data_fvec1ds, nsystems):
        yield x


def gen_random_lmap_systems(pzero, nsystems=40):
    """
    Sample whole systems for testing likelihood.
    The pzero parameter indicates sparsity.
    Yield (T, edge_to_P, root, root_prior_distn, node_to_data_lmap).

    """
    for x in _gen_random_systems(pzero, sample_data_lmaps, nsystems):
        yield x


def gen_random_infeasible_lmap_systems(nsystems=60):
    for x in _gen_random_infeasible_systems(sample_data_lmaps, nsystems):
        yield x

