"""
A helper module for dealing with serialization of a gap-free paralog alignment.

This uses JSON for a few reasons.
It is easy to convert to and from a Python data representation,
it is more or less human-readable, and it is easy for Javascript
(and other languages) to read, for example if you want to make a visualization.
It should also be easy to convert to and from other file formats
and library formats.

When this module is run from the command line as a script,
it does a round-trip test.

"""
from __future__ import division, print_function, absolute_import

import unittest
import json

__all__ = ['seqdata_to_json', 'json_to_seqdata']


def seqdata_to_json(nsites, data):
    """
    Get a JSON representation of an alignment that knows about paralogs.

    Parameters
    ----------
    nsites : integer
        Number of aligned nucleotide sites per sequence.
    data : dict
        Map from (taxon, paralog) to nucleotide sequence string.

    Returns
    -------
    s : string
        JSON string

    """
    verbose_triples = []
    for (taxon, paralog), sequence in data.items():
        d = dict(taxon=taxon, paralog=paralog, sequence=sequence)
        verbose_triples.append(d)
    toplevel = dict(nsites=nsites, seqdata=verbose_triples)
    return json.dumps(toplevel, indent=4)


def json_to_seqdata(s):
    """
    Read a JSON representation of an alignment that knows about paralogs.

    Parameters
    ----------
    s : string
        A JSON string as created by seqdata_to_json.

    Returns
    -------
    nsites : integer
        Number of aligned nucleotide sites per sequence.
    data : dict
        Map from (taxon, paralog) to nucleotide sequence string.

    """
    toplevel = json.loads(s)
    data = {}
    for d in toplevel['seqdata']:
        data[d['taxon'], d['paralog']] = d['sequence']
    return toplevel['nsites'], data


class TestRoundTrip(unittest.TestCase):
    def test_roundtrip(self):
        nsites_in = 4
        data_in = {
                ('dog', 'AlphaGlobin') : 'AACC',
                ('dog', 'BetaGlobin') : 'ACTG',
                ('cat', 'AlphaGlobin') : 'AAGC',
                ('cat', 'BetaGlobin') : 'AATG'}
        s = seqdata_to_json(nsites_in, data_in)
        nsites_out, data_out = json_to_seqdata(s)
        self.assertEqual(type(s), str)
        self.assertEqual(nsites_out, nsites_in)
        self.assertEqual(data_out, data_in)


if __name__ == '__main__':
    unittest.main()
