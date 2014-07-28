"""
Test the cython implementation of iid likelihood calculation.

"""
from __future__ import division, print_function, absolute_import

from numpy.testing import assert_allclose

from npmctree import puzzles, dynamic_lmap_lhood


def test_get_iid_lhoods():
    pzero = 0.2
    for args in puzzles.gen_random_iid_lmap_systems(pzero):
        T, e_to_P, r, r_prior, node_lmaps = args
        cy_lhoods = dynamic_lmap_lhood.get_iid_lhoods(*args)
        py_lhoods = []
        for node_lmap in node_lmaps:
            lhood = dynamic_lmap_lhood.get_lhood(
                    T, e_to_P, r, r_prior, node_lmap)
            py_lhoods.append(0 if lhood is None else lhood)
        assert_allclose(cy_lhoods, py_lhoods)
