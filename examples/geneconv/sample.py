"""
A small example based on the biological phenomenon of gene conversion.

This script samples an ungapped alignment of some paralogs for some taxa.

"""
from __future__ import division, print_function, absolute_import

import argparse

import numpy as np
import networkx as nx
import scipy.linalg

import npmctree
from npmctree.sampling import sample_histories

from util import seqdata_to_json
from model import get_state_space, get_tree_info, get_pre_Q, get_Q_and_distn


def main(args):

    # interpret the command line arguments
    nhistories = args.n
    phi = args.phi

    # define the state space
    nt_pairs, pair_to_state = get_state_space()
    nstates = len(nt_pairs)

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
    d = {}
    for leaf in leaves:
        pairs = node_to_pairs[leaf]
        seqs = [''.join(x).upper() for x in zip(*pairs)]
        d[leaf, 'alpha'] = seqs[0]
        d[leaf, 'beta'] = seqs[1]

    print(seqdata_to_json(nhistories, d))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=10, type=int,
            help='number of sites in the sampled alignment')
    parser.add_argument('--phi', default=2.0, type=float,
            help='strength of the gene conversion effect')
    args = parser.parse_args()
    main(args)

