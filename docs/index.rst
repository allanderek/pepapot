.. complexity documentation master file, created by
   sphinx-quickstart on Tue Jul  9 22:26:36 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pepapot's documentation!
======================================

PEPApot is a library and command-line tool for analysing performance models.
Models are written in either PEPA or Bio-PEPA format. PEPA models are
currently solved by deriving and solving the underlying CTMC. Bio-PEPA models
are solved by converting the model to a set of Ordinary Differential Equations
and evaluting those.

One advantage of such process algebra approaches to model specficiation is
that the same model can be evaluated using separate methods. It is hence
planned to implement extra methods for both PEPA and Bio-PEPA models. In
particular it is planned to implement the evaluation of both kinds of models
using stochastic simulation.

.. toctree::
   :maxdepth: 2

   readme
   installation
   usage
   contributing
   authors
   history

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
