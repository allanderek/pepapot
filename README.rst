===============================
pepapot
===============================

An attempt at writing a very simple PEPA tool in Python. It is intended to be
a compliment to pyPEPA. Here though the focus is on being as simple as
possible and hence can be used in, for example, student projects.

* Free software: BSD license

Features
--------

So far we can solve a PEPA model and produce a list of utilisation
dictionaries. There is a dictionary for each process in the system equation
which maps each local state of the process into the steady-state probability
of the process being in that local state. For aggregated processes it is
essentially a population mapping.

Getting Started
---------------

The easiest way to get started is to use virtualenv. Having installed that
then one can simply do:

    $ mkvirtualenv --distribute -p <path to python3> pepapot
    $ pip install pyparsing numpy lazy
    $ python setup.py test

I have not yet made up an executable but one can evaluate a PEPA file with:

    $ python pepapot/pepapot.py steady util <path to PEPA file>