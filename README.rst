===============================
pypepa
===============================

.. image:: https://badge.fury.io/py/pypepa.png
    :target: http://badge.fury.io/py/pypepa
    
.. image:: https://travis-ci.org/allanderek/pypepa.png?branch=master
        :target: https://travis-ci.org/allanderek/pypepa

.. image:: https://pypip.in/d/pypepa/badge.png
        :target: https://crate.io/packages/pypepa?version=latest


An attempt at writing a very simple PEPA tool in Python. It is intended to be
a compliment to pyPEPA. Here though the focus is on being as simple as
possible and hence can be used in, for example, student projects.

* Free software: BSD license
* Documentation: http://pypepa.rtfd.org.

Features
--------

So far we can solve a PEPA model and produce a list of utilisation
dictionaries. There is a dictionary for each process in the system equation
which maps each local state of the process into the steady-state probability
of the process being in that local state. For aggregated processes it is
essentially a population mapping.