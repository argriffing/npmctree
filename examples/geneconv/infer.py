"""
A small example based on the biological phenomenon of gene conversion.

"""
from __future__ import division, print_function, absolute_import

from functools import partial
import argparse

import numpy as np
import networkx as nx
import scipy.optimize
import scipy.linalg

import npmctree
from npmctree.dynamic_fset_lhood import get_lhood

from model import get_state_space, get_tree_info, get_pre_Q, get_Q_and_distn
from util import json_to_seqdata


def get_data_constraints(nodes, pair_to_state, nsites, data):
    """
    Get sequence data constraints for the likelihood calculation.

    Also return some match/mismatch counts.

    Parameters
    ----------
    nodes : set
        set of all nodes in the tree
    pair_to_state : dict
        maps nucleotide pairs to an integer state index
    nsites : integer
        number of sites in the alignment
    data : dict
        map from (node, paralog) to nucleotide sequence string

    Returns
    -------
    per_site_constraints : sequence
        for each site, a map from
    npaired_yes : integer
        number of site at which aligned paralog nucleotides match
    npaired_no : integer
        number of site at which aligned paralog nucleotides mismatch

    """
    # maybe do not hard code this...
    paralogs = ('alpha', 'beta')
    # assume we only have sequence data at the leaves...
    leaves = set(taxon for taxon, paralog in data)
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
        for node in non_leaves:
            node_to_data_fset[node] = np.ones(nstates, dtype=bool)
        for node in leaves:
            nt0 = data[node, 'alpha'][site]
            nt1 = data[node, 'beta'][site]
            pair = (nt0, nt1)
            if nt0 == nt1:
                npaired_yes += 1
            else:
                npaired_no += 1
            state = pair_to_state[pair]
            fset = np.zeros(nstates, dtype=bool)
            fset[state] = True
            node_to_data_fset[node] = fset
        per_site_constraints.append(node_to_data_fset)
    return per_site_constraints, npaired_yes, npaired_no


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

    # read the input file with the sequence alignment paralog data at leaves
    with open(args.seqdata) as fin:
        s = fin.read()
        nsites, data = json_to_seqdata(s)

    # define the state space
    nt_pairs, pair_to_state = get_state_space()
    nstates = len(nt_pairs)

    # get the tree info, using the default branch lengths
    T, root, edge_to_blen = get_tree_info()
    nodes = set(T)

    # get the per-site constraints
    constraint_info = get_data_constraints(nodes, pair_to_state, nsites, data)
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
    x0 = [phi_hat]
    print('preliminary estimate of phi:', phi_hat)
    print('negative log likelihood for preliminary estimate:', f(x0))

    # get the max likelihood estimate of phi
    result = scipy.optimize.minimize(
            f, x0=x0, method='L-BFGS-B', bounds=[(0, None)])
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqdata', required=True,
            help='paralog alignment file generated by the sampling script')
    args = parser.parse_args()
    main(args)

