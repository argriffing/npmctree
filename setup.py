#!/usr/bin/env python
"""Markov chain algorithms on a Python NetworkX tree with numpy arrays.

"""

DOCLINES = __doc__.split('\n')

# This setup script is written according to
# http://docs.python.org/2/distutils/setupscript.html
#
# It is meant to be installed through github using pip.
#
# More stuff added for Cython extensions.

from distutils.core import setup

from distutils.extension import Extension

from Cython.Distutils import build_ext

import numpy as np


setup(
        name='npmctree',
        version='0.1',
        description=DOCLINES[0],
        author='alex',
        url='https://github.com/argriffing/npmctree/',
        download_url='https://github.com/argriffing/npmctree/',
        packages=['npmctree'],
        test_suite='nose.collector',
        package_data={'npmctree' : ['tests/test_*.py']},
        cmdclass={'build_ext' : build_ext},
        ext_modules=[Extension('npmctree.cyfels', ['npmctree/cyfels.pyx'],
            include_dirs=[np.get_include()])],

        )

