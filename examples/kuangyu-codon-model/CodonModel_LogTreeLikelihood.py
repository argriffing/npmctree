import itertools

import numpy as np
import networkx as nx
from numpy.testing import assert_equal
from scipy.linalg import expm
#import scipy.optimize

import npmctree
from npmctree.dynamic_lmap_lhood import get_iid_lhoods

import sys
import re
import math

from YN98_utility import *

codon2state = {
    'AAA':0,'AAC':1,'AAG':2,'AAT':3,
    'ACA':4,'ACC':5,'ACG':6,'ACT':7,
    'AGA':8,'AGC':9,'AGG':10,'AGT':11,
    'ATA':12,'ATC':13,'ATG':14,'ATT':15,
    'CAA':16,'CAC':17,'CAG':18,'CAT':19,
    'CCA':20,'CCC':21,'CCG':22,'CCT':23,
    'CGA':24,'CGC':25,'CGG':26,'CGT':27,
    'CTA':28,'CTC':29,'CTG':30,'CTT':31,
    'GAA':32,'GAC':33,'GAG':34,'GAT':35,
    'GCA':36,'GCC':37,'GCG':38,'GCT':39,
    'GGA':40,'GGC':41,'GGG':42,'GGT':43,
    'GTA':44,'GTC':45,'GTG':46,'GTT':47,
    'TAC':48,'TAT':49,
    'TCA':50,'TCC':51,'TCG':52,'TCT':53,
    'TGC':54,'TGG':55,'TGT':56,
    'TTA':57,'TTC':58,'TTG':59, 'TTT':60
}

def ad_hoc_codonSeq_reader(fin):
	name_seq_pairs = []
	while True:
		line = fin.readline().strip()
		if not line:
			return name_seq_pairs

		assert_equal(line[0], '>')
		name = line[1:].strip()
		line = fin.readline().strip()

		if len(line) % 3 != 0:
			raise Exception('length should be multiple of 3')

		chunks, chunk_size = len(line), 3
		seq = [line[i:i+chunk_size] for i in range(0, chunks, chunk_size) ]

		#need to add check for stop codons
		#unrecognized = set(line) - set('ACGT') # need to change to stop codons
		#if unrecognized:
			#raise Exception('unrecognized nucleotides: ' + str(unrecognized))

		name_seq_pairs.append((name, seq))
	return name_seq_pairs

def get_logTreeLikelihood(T, root, edge_to_blen, rateM, root_distn, data):
	edge_to_P = {}
	for edge in T.edges():
		blen = edge_to_blen[edge]
		P = expm(blen * rateM)
		#print('row sums of P:')
		#print(P.sum(axis=1))
		edge_to_P[edge] = P

	lhoods = get_iid_lhoods(T, edge_to_P, root, root_distn, data)
	return np.log(lhoods).sum()

def getYN98LogTreeLikelihood(T, root, edge_to_blen, seqFile, k, omega, nucleoFreqs):

	# Read the data as name sequence pairs.
	with open(seqFile) as fin:
		name_seq_pairs = ad_hoc_codonSeq_reader(fin)
	name_to_seq = dict(name_seq_pairs)

	#print(name_to_seq)
	taxa = ['A', 'B']
	nsites = len(name_seq_pairs[0][1])
	#print("nSites:" + str(nsites))

	#also called: node_to_data_lmaps (e.g. get_iid_lhoods(T, edge_to_P, root, root_prior_distn1d, node_to_data_lmaps))
	constraints = []

	for site in range(nsites):
		node_to_lmap = {}
		for node in T:
			if node in taxa:
				lmap = np.zeros(len(codon2state), dtype=float)
				codon = name_to_seq[node][site]
				state = codon2state[codon]
				lmap[state] = 1.0
			else:
				lmap = np.ones(len(codon2state), dtype=float)
			node_to_lmap[node] = lmap
		constraints.append(node_to_lmap)
	
	#model specific parameters
	root_distn = np.array(getDiagM(nucleoFreqs))
	rateM = getRateM(k, omega, nucleoFreqs)

	logP = get_logTreeLikelihood(T, root, edge_to_blen, rateM, root_distn, constraints)
	return logP

def get_tree_info():
    T = nx.DiGraph()
    edge_to_blen = {}
    root = 'N0'
    triples = (
            ('N0', 'A', 5.092090428490227),
            ('N0', 'B', 5.092090428490227))
    for va, vb, blen in triples:
        edge = (va, vb)
        T.add_edge(*edge)
        edge_to_blen[edge] = blen
    return T, root, edge_to_blen

# Read the hardcoded tree information.
T, root, edge_to_blen = get_tree_info()

if __name__ == "__main__":
	# Read the hardcoded tree information.
	T, root, edge_to_blen = get_tree_info()

	seqFile = '[path to]/example1.fasta'
	k = 0.5946553895418681
	omega = 0.008271934959822749
	nucleoFreqs = [0.2617387287959177,0.2428420396566907,0.2899242743384996,0.20549495720889255]
	logP = getYN98LogTreeLikelihood(T, root, edge_to_blen, seqFile, k, omega, nucleoFreqs)
	print("LogTreeLikelihood:" + str(logP))
