Markov chain algorithms on a Python NetworkX tree with numpy arrays.

Required dependencies:
 * [Python 2.7+](http://www.python.org/)
 * [pip](https://pip.readthedocs.org/) (installation)
 * [git](http://git-scm.com/) (installation)
 * [NetworkX](http://networkx.lanl.gov/) (graph data types and algorithms)
   - `$ pip install --user git+https://github.com/networkx/networkx`
 * [Cython](http://cython.org) (manually write unsafe fast code for Python)
   - `$ pip install --user git+https://github.com/cython/cython`
 * [numpy](http://www.numpy.org/) (arrays)

Optional dependencies:
 * [nose](http://readthedocs.org/docs/nose/en/latest/) (testing)
   - `$ pip install --user git+https://github.com/nose-devs/nose`
 * [coverage](http://nedbatchelder.com/code/coverage/) (test coverage)
   - `$ apt-get install python-coverage`


User
----

Install:

    $ pip install --user git+https://github.com/argriffing/npmctree

Test:

    $ python -c "import npmctree; npmctree.test()"

Uninstall:

    $ pip uninstall npmctree


Developer
---------

Install:

    $ git clone git@github.com:argriffing/npmctree.git

Test:

    $ python runtests.py

Coverage:

    $ python-coverage run runtests.py
    $ python-coverage html
    $ chromium-browser htmlcov/index.html

Build docs locally:

    $ sh make-docs.sh
    $ chromium-browser /tmp/nxdocs/index.html

Subsequently update online docs:

    $ git checkout gh-pages
    $ cp /tmp/nxdocs/. ./ -R
    $ git add .
    $ git commit -am "update gh-pages"
    $ git push


Developer notes -- variable naming glossary
-------------------------------------------

 * `fvec1d` : fixed-length 1d bool ndarray
              representing a feasible set
 * `fvec2d` : fixed-size 2d bool ndarray
              representing a feasible set
 * `distn1d` : fixed-length 1d float ndarray
               representing a finite univariate distribution
 * `distn2d` : fixed-size 2d float ndarray
               representing a finite bivariate distribution

