"""
A small example based on the biological phenomenon of gene conversion.

"""
from __future__ import division, print_function, absolute_import

from functools import partial
from itertools import product
import argparse

import numpy as np
import networkx as nx
import scipy.optimize
import scipy.linalg
from numpy.testing import assert_allclose

import npmctree
from npmctree.sampling import sample_histories
from npmctree.dynamic_fset_lhood import get_lhood


def get_data_constraints(nodes, pair_to_state):
    # constructed with branch length 0.1 for all branches and phi = 2.0.
    nsites = 60
    leaf_to_paralogs = {
            'N0' : (
                'TCCTATAACGAATGCCAACTCCGGGGTTCGCATAGTTTCGTCAGGACACACCTGCCGAGC',
                'GCTGATGACCACTCTTTTGTTGAGGCTGTGGCTTGGCTCCCTGGATAAATCCGACGACAA'),
            'N3' : (
                'TCCTATCCCGAGTGCCGACACAGCCGTTTGCATATTGTCGTCGGGACCCACCTGCCGAGC',
                'GCTAAGGACCAATCTCTTGTTGCGGCCGTGGCTTGGCTCCCCGGGTAAATCCGACGACCA'),
            'N4' : (
                'TGCTATCACGAGAGCCAACACCGCGGTTCGCATATTGTCGTCGGGACCCACCGGCCGAGC',
                'GCGAAGGACCACGCTCATGTTGCGGCTCTGGCTTGGCTCCCCGGATAAATCCGACGCCGA'),
            'N5' : (
                'TCCTATATCGAGTGCCAACACCGCGGTACGCATAGTTTCGTCAGGACACACCTGCCGAGC',
                'GCTAAGGACCACTCTTTTGCTGCGGCTGTGGCTTGGCTCCCCGGATAAAACCGATGACAA'),
            }
    nstates = len(pair_to_state)
    non_leaves = set(nodes) - set(leaf_to_paralogs)
    per_site_constraints = []
    npaired_yes = 0
    npaired_no = 0
    for site in range(nsites):
        node_to_data_fset = {}
        for node in nodes:
            if node in non_leaves:
                node_to_data_fset[node] = np.ones(nstates)
            else:
                p0, p1 = leaf_to_paralogs[node]
                pair = (p0[site].lower(), p1[site].lower())
                if pair[0] == pair[1]:
                    npaired_yes += 1
                else:
                    npaired_no += 1
                state = pair_to_state[pair]
                fset = np.zeros(nstates)
                fset[state] = 1
                node_to_data_fset[node] = fset
        per_site_constraints.append(node_to_data_fset)
    return per_site_constraints, npaired_yes, npaired_no


def get_tree_info():
    T = nx.DiGraph()
    edge_to_blen = {}
    root = 'N1'
    triples = (
            ('N1', 'N0', 0.4),
            ('N1', 'N2', 0.4),
            ('N1', 'N5', 0.4),
            ('N2', 'N3', 0.4),
            ('N2', 'N4', 0.4))
    for va, vb, blen in triples:
        edge = (va, vb)
        T.add_edge(*edge)
        edge_to_blen[edge] = blen
    return T, root, edge_to_blen


def get_pre_Q(nt_pairs, phi):
    """
    Parameters
    ----------
    nt_pairs : sequence
        Ordered pairs of nucleotide states.
        The order of this sequence defines the order of rows and columns of Q.
    phi : float
        Under this model nucleotide substitutions are more likely to
        consist of changes that make paralogous sequences more similar
        to each other.  This parameter is the ratio capturing this effect.

    Returns
    -------
    pre_Q : numpy ndarray
        The rate matrix without normalization and without the diagonal.

    """
    slow = 1
    fast = phi
    n = len(nt_pairs)
    pre_Q = np.zeros((n, n), dtype=float)
    for i, (s0a, s1a) in enumerate(nt_pairs):
        for j, (s0b, s1b) in enumerate(nt_pairs):
            # Diagonal entries will be set later.
            if i == j:
                continue
            # Only one change is allowed at a time.
            if s0a != s0b and s1a != s1b:
                continue
            # Determine which paralog changes.
            if s0a != s0b:
                sa = s0a
                sb = s0b
                context = s1a
            if s1a != s1b:
                sa = s1a
                sb = s1b
                context = s0a
            # Set the rate according to the kind of change.
            if context == sb:
                rate = fast
            else:
                rate = slow
            pre_Q[i, j] = rate
    return pre_Q


def get_Q_and_distn(nt_pairs, phi):

    # Define the unnormalized rate matrix with negative diagonal.
    pre_Q = get_pre_Q(nt_pairs, phi)
    unnormalized_Q = pre_Q - np.diag(pre_Q.sum(axis=1))

    # Guess the stationary distribution.
    root_distn = []
    for nt0, nt1 in nt_pairs:
        root_distn.append(phi if nt0 == nt1 else 1)
    root_distn = np.array(root_distn) / sum(root_distn)

    # Check that the stationary distribution is correct.
    equilibrium_rates = np.dot(root_distn, unnormalized_Q)
    assert_allclose(equilibrium_rates, 0, atol=1e-12)

    # Normalize the rate matrix so that branch lengths
    # have the usual interpretation.
    expected_rate = np.dot(root_distn, -np.diag(unnormalized_Q))
    Q = unnormalized_Q / expected_rate

    return Q, root_distn


def objective(T, root, edge_to_blen, nt_pairs, constraints, params):
    """
    Negative log likelihood as a function of the parameters (only phi).

    """
    # unpack the parameters
    phi = params[0]

    # Define the stationary distribution and transition probability matrices
    # using the parameters.
    Q, root_distn = get_Q_and_distn(nt_pairs, phi)
    edge_to_P = {}
    for edge in T.edges():
        blen = edge_to_blen[edge]
        P = scipy.linalg.expm(blen * Q)
        edge_to_P[edge] = P

    # sum the negative log likelihoods over each site in the alignment
    neg_ll = 0
    for node_to_data_fvec1d in constraints:
        lhood = get_lhood(T, edge_to_P, root, root_distn, node_to_data_fvec1d)
        neg_ll -= np.log(lhood)

    # return negative log likelihood
    return neg_ll


def main(args):
    """
    Sample some data and subsequently infer parameter values.

    """
    # define the state space
    nt_pairs = []
    pair_to_state = {}
    for i, pair in enumerate(product('acgt', repeat=2)):
        nt_pairs.append(pair)
        pair_to_state[pair] = i
    nstates = len(nt_pairs)

    # interpret the command line arguments
    nhistories = args.n
    phi = args.phi

    # get the transition rate matrix and stationary distribution
    Q, root_distn = get_Q_and_distn(nt_pairs, phi)

    # get the tree info
    T, root, edge_to_blen = get_tree_info()
    nodes = set(T)
    leaves = set(v for v, degree in T.degree().items() if degree == 1)
    not_leaves = nodes - leaves

    # compute probability transition matrices for each edge
    edge_to_P = {}
    for edge in T.edges():
        blen = edge_to_blen[edge]
        P = scipy.linalg.expm(blen * Q)
        edge_to_P[edge] = P

    # do not impose any data constraints
    node_to_data_lmap = {}
    for node in nodes:
        node_to_data_lmap[node] = np.ones(nstates)

    # for each node, get a sequence of nucleotide pairs
    npaired_yes = 0
    npaired_no = 0
    constraints = []
    node_to_pairs = dict((node, []) for node in nodes)
    for node_to_state in sample_histories(T, edge_to_P, root,
            root_distn, node_to_data_lmap, nhistories):

        # convert node_to_state to constraints
        d = {}
        for v in not_leaves:
            d[v] = np.ones(nstates, dtype=bool)
        for v, state in node_to_state.items():
            d[v] = np.zeros(nstates, dtype=bool)
            d[v][state] = True
        constraints.append(d)
        
        # summarize the data and build the sequences from the sites
        for node, state in node_to_state.items():
            nt_pair = nt_pairs[state]
            if nt_pair[0] == nt_pair[1]:
                npaired_yes += 1
            else:
                npaired_no += 1
            node_to_pairs[node].append(nt_pair)

    # report the paralogs for the leaf nodes
    for leaf in leaves:
        pairs = node_to_pairs[leaf]
        seqs = [''.join(x).upper() for x in zip(*pairs)]
        print('taxon', leaf)
        print('paralog 1:', seqs[0])
        print('paralog 2:', seqs[1])
        print()

    # Get the negative log likelihood function,
    # purely as a function of the phi parameter.
    f = partial(objective, T, root, edge_to_blen, nt_pairs, constraints)

    # Use the data summary to get a preliminary estimate of phi.
    # This would be the correct estimate if branch lengths were zero,
    # using a back of the envelope calculation.
    print('paralog site matches   :', npaired_yes)
    print('paralog site mismatches:', npaired_no)
    nsmall = 4
    pa = npaired_yes * (nsmall - 1)
    pb = npaired_no
    phi_hat = pa / pb
    print('preliminary estimate of phi:', phi_hat)

    # get the max likelihood estimate of phi
    result = scipy.optimize.minimize(
            f, x0=[phi_hat], method='L-BFGS-B', bounds=[(0, None)])
    print(result)



def main_inference(args):

    # define the state space
    nt_pairs = []
    pair_to_state = {}
    for i, pair in enumerate(product('acgt', repeat=2)):
        nt_pairs.append(pair)
        pair_to_state[pair] = i
    nstates = len(nt_pairs)

    # interpret the command line arguments
    nhistories = args.n
    phi = args.phi

    # get the tree info
    T, root, edge_to_blen = get_tree_info()
    nodes = set(T)
    leaves = set(v for v, degree in T.degree().items() if degree == 1)

    # get the per-site constraints
    constraint_info = get_data_constraints(nodes, pair_to_state)
    constraints, npaired_yes, npaired_no = constraint_info

    # Get the negative log likelihood function,
    # purely as a function of the phi parameter.
    f = partial(objective, T, root, edge_to_blen, nt_pairs, constraints)

    # Use the data summary to get a preliminary estimate of phi.
    # This would be the correct estimate if branch lengths were zero,
    # using a back of the envelope calculation.
    print('paralog site matches   :', npaired_yes)
    print('paralog site mismatches:', npaired_no)
    nsmall = 4
    pa = npaired_yes * (nsmall - 1)
    pb = npaired_no
    phi_hat = pa / pb
    print('preliminary estimate of phi:', phi_hat)

    # get the max likelihood estimate of phi
    result = scipy.optimize.minimize(
            f, x0=[phi_hat], method='L-BFGS-B', bounds=[(0, None)])
    print(result)


def main_sample(args):

    # define the state space
    nt_pairs = []
    pair_to_state = {}
    for i, pair in enumerate(product('acgt', repeat=2)):
        nt_pairs.append(pair)
        pair_to_state[pair] = i
    nstates = len(nt_pairs)

    # interpret the command line arguments
    nhistories = args.n
    phi = args.phi

    # get the transition rate matrix and stationary distribution
    Q, root_distn = get_Q_and_distn(nt_pairs, phi)

    # get the tree info
    T, root, edge_to_blen = get_tree_info()
    nodes = set(T)
    leaves = set(v for v, degree in T.degree().items() if degree == 1)

    # compute probability transition matrices for each edge
    edge_to_P = {}
    for edge in T.edges():
        blen = edge_to_blen[edge]
        P = scipy.linalg.expm(blen * Q)
        edge_to_P[edge] = P

    # do not impose any data constraints
    node_to_data_lmap = {}
    for node in nodes:
        node_to_data_lmap[node] = np.ones(nstates)

    # for each node, get a sequence of nucleotide pairs
    node_to_pairs = dict((node, []) for node in nodes)
    for node_to_state in sample_histories(T, edge_to_P, root,
            root_distn, node_to_data_lmap, nhistories):
        for node, state in node_to_state.items():
            nt_pair = nt_pairs[state]
            node_to_pairs[node].append(nt_pair)

    # report the paralogs for the leaf nodes
    for leaf in leaves:
        pairs = node_to_pairs[leaf]
        seqs = [''.join(x).upper() for x in zip(*pairs)]
        print('taxon', leaf)
        print('paralog 1:', seqs[0])
        print('paralog 2:', seqs[1])
        print()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=10, type=int,
            help='number of sites in the sampled alignment')
    parser.add_argument('--phi', default=2.0, type=float,
            help='strength of the gene conversion effect')
    args = parser.parse_args()
    main(args)

