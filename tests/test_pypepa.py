#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pypepa
----------------------------------

Tests for `pypepa` module.
"""

import unittest
import logging

import numpy

from pypepa import pypepa
Action = pypepa.Action

simple_components = """
P = (a,1.0).P1;
P1 = (b, 1.0).P;

Q = (a,1.0).Q1;
Q1 = (b, 1.0).Q;
"""

simple_choice_component = """
P = (a, 1.0).P1 + (b, 1.0).P2;
P1 = (c, 1.0).P;
P2 = (d, 1.0).P;
"""

class TestSimpleNoCoop(unittest.TestCase):
    """This tests a very simple test model. It also serves as a base class from
       which all other cases testing particular models should derive.
       A subclass should re-write the setup method populating all of the
       expected values with appropriate ones.
    """
    def setUp(self):
        self.model_source = simple_components + "\nP || Q"
        self.expected_used_process_names = set(["P", "P1", "Q", "Q1"])
        self.expected_defined_process_names = set(["P", "P1", "Q", "Q1"])

        self.expected_actions_dictionary = dict()
        self.expected_actions_dictionary["P"] = [ Action("a", 1.0, "P1") ]
        self.expected_actions_dictionary["P1" ] = [ Action ("b", 1.0, "P") ]
        self.expected_actions_dictionary["Q" ] = [ Action("a", 1.0, "Q1") ]
        self.expected_actions_dictionary["Q1" ] = [ Action("b", 1.0, "Q") ]

        self.expected_shared_actions = set()

        self.expected_initial_state = ("P", "Q")
        self.expected_state_space_size = 4

        self.expected_solution = [ (("P", "Q"), 0.25),
                                   (("P1", "Q"), 0.25),
                                   (("P", "Q1"), 0.25),
                                   (("P1", "Q1"), 0.25)
                                 ]

        self.expected_utilisations = [ dict([ ("P", 0.5),
                                              ("P1", 0.5) ]),
                                       dict([ ("Q", 0.5),
                                              ("Q1", 0.5) ])
                                     ]

    @property
    def model(self):
        if getattr(self, "_model", None) is None:
            self._model = pypepa.parse_model(self.model_source)
        return self._model

    @property
    def state_space(self):
        if getattr(self, "_state_space", None) is None:
            self._state_space = pypepa.build_state_space(self.model)
        return self._state_space

    @property
    def gen_matrix(self):
        if getattr(self, "_gen_matrix", None) is None:
            self._gen_matrix = pypepa.get_generator_matrix(self.state_space)
        return self._gen_matrix

    @property
    def steady_solution(self):
        if getattr(self, "_steady_solution", None) is None:
            self._steady_solution = pypepa.solve_generator_matrix(self.gen_matrix)
        return self._steady_solution

    @property
    def steady_utilisations(self):
        if getattr(self, "_steady_utilisations", None) is None:
            initial_state = self.model.get_initial_state()
            self._steady_utilisations = pypepa.get_utilisations(initial_state,
                                                                self.state_space,
                                                                self.steady_solution)
        return self._steady_utilisations

    def assertAlmostEqual(self, a, b):
        """A helper method to assert that two values are approximately equal.
           This is useful since floating point operations often do not end in
           exactly correct answers. There is scope here for adding in an
           absolute and relative tolerance, but for now we'll assume that we're
           interested in a fixed level of accuracy. The scipy assertall method
           has something a bit more sophisticated than this including atol and
           rtol, if this becomes necessary. The scipy method works over arrays
           and we likely wish to work over single values but we could easily
           adapt the code.
        """
        self.assertTrue((abs(a - b)) < 1e-8)

    def test_parse_model(self):
        shared_actions = self.model.system_equation.get_shared_actions()
        self.assertEqual(shared_actions, self.expected_shared_actions)

    def test_used_names(self):
        used_names = self.model.used_process_names()
        self.assertEqual(used_names, self.expected_used_process_names)

    def test_defined_names(self):
        defined_names = self.model.defined_process_names()
        self.assertEqual(defined_names, self.expected_defined_process_names)

    def test_actions(self):
        actual_actions = self.model.get_process_actions()
        self.assertEqual(actual_actions, self.expected_actions_dictionary)

    def test_initial_state(self):
        initial_state = self.model.get_initial_state()
        self.assertEqual(initial_state, self.expected_initial_state)

    def test_state_space_size(self):
        self.assertEqual(len(self.state_space), self.expected_state_space_size)

    def test_generator_matrix(self):
        for (row_number, row) in enumerate(self.gen_matrix):
            self.assertEqual(0.0, sum(row))
            self.assertTrue(row[row_number] < 0.0)

    def test_steady_state_solve(self):
        for (state, probability) in self.expected_solution:
            state_number, transitions = self.state_space[state]
            # You can search a little for testing floating point numbers
            # for equality and conceivably we could do so with respect to an
            # absolute tolerance and a relative tolerance, for now I believe
            # this is sufficient.
            expected_probability = self.steady_solution[state_number]
            self.assertAlmostEqual(probability, expected_probability)

    def test_utilisations(self):
        for actual_utils, expected_utils in zip(self.steady_utilisations,
                                                self.expected_utilisations):
            for process, utilisation in expected_utils.items():
                self.assertAlmostEqual(actual_utils[process], utilisation)

class TestSimpleSingleCoop(TestSimpleNoCoop):
    """This model has most of the same results as the model without any
       cooperation so we inherit from that and then change the model rather
       than from the base test model class.
    """
    def setUp(self):
        # This model has most of the same results as the model without
        # any cooperation so we inhebase test case
        # So I'll just call super here rather than copy and paste, but in
        # general it shouldn't be necessary to call super here.
        super(TestSimpleSingleCoop, self).setUp()
        self.model_source = simple_components + "\nP <a> Q"
        self.expected_shared_actions = set(["a"])
        self.expected_solution = [ (("P", "Q"), 0.4),
                                   (("P1", "Q"), 0.2),
                                   (("P", "Q1"), 0.2),
                                   (("P1", "Q1"), 0.2)
                                 ]
        self.expected_utilisations = [ dict([ ("P", 0.6),
                                              ("P1", 0.4) ]),
                                       dict([ ("Q", 0.6),
                                              ("Q1", 0.4) ])
                                     ]

class TestSimpleDoubleCoop(TestSimpleNoCoop):
    """Similar to the above case we're only using super here because we can
       and so many of th expected results are the same.
    """
    def setUp(self):
        super(TestSimpleDoubleCoop, self).setUp()
        self.model_source = simple_components + "\nP <a, b> Q"
        self.expected_shared_actions = set(["a", "b"])
        self.expected_state_space_size = 2
        self.expected_solution = [ (("P", "Q"), 0.5),
                                   (("P1", "Q1"), 0.5)
                                 ]
        self.expected_utilisations = [ dict([ ("P", 0.5),
                                              ("P1", 0.5) ]),
                                       dict([ ("Q", 0.5),
                                              ("Q1", 0.5) ])
                                     ]

class TestSimpleAlias(TestSimpleNoCoop):
    """Similar to the above case we're only using super here because we can
       and so many of the expected results are the same.
    """
    def setUp(self):
        super(TestSimpleAlias, self).setUp()
        self.model_source = "A = P;\n" + simple_components + "\nP || Q"
        self.expected_defined_process_names.add("A")
        self.expected_actions_dictionary["A"] = self.expected_actions_dictionary["P"]

    # Note, if you expect everything to fail, you can decorate the class with
    # unittest.expectedFailure, however I prefer this as if you decorate the
    # entire class, then it is essentially the same as skipping the tests, that
    # is no report is given saying how many expected failures there were.
    @unittest.expectedFailure
    def test_actions(self):
        super(TestSimpleAlias, self).test_actions()

    @unittest.expectedFailure
    def test_state_space_size(self):
        super(TestSimpleAlias, self).test_state_space_size()

    @unittest.expectedFailure
    def test_generator_matrix(self):
        super(TestSimpleAlias, self).test_generator_matrix()

    @unittest.expectedFailure
    def test_steady_state_solve(self):
        super(TestSimpleAlias, self).test_steady_state_solve()

    @unittest.expectedFailure
    def test_utilisations(self):
        super(TestSimpleAlias, self).test_utilisations()

class TestSimpleChoice(TestSimpleNoCoop):
    def setUp(self):
        self.model_source = simple_choice_component + "\nP"

        self.expected_used_process_names = set(["P", "P1", "P2"])
        self.expected_defined_process_names = self.expected_used_process_names

        self.expected_actions_dictionary = dict()
        self.expected_actions_dictionary["P"] = [ Action("a", 1.0, "P1"),
                                                  Action("b", 1.0, "P2") ]
        self.expected_actions_dictionary["P1" ] = [ Action("c", 1.0, "P") ]
        self.expected_actions_dictionary["P2" ] = [ Action("d", 1.0, "P") ]

        self.expected_shared_actions = set()

        self.expected_initial_state = "P"
        self.expected_state_space_size = 3

        self.expected_solution = [ ("P", 1.0 / 3.0),
                                   ("P1", 1.0 / 3.0),
                                   ("P2", 1.0 / 3.0)
                                 ]

        self.expected_utilisations = [ dict(self.expected_solution) ]

class TestChoiceAlias(TestSimpleNoCoop):
    def setUp(self):
        self.model_source = """P = P1 + P2;
                               P1 = (a, 1.0).P3;
                               P2 = (b, 1.0).P3;
                               P3 = (c, 1.0).P;
                               P
                            """

        self.expected_used_process_names = set(["P", "P1", "P2", "P3"])
        self.expected_defined_process_names = self.expected_used_process_names

        self.expected_shared_actions = set()

        self.expected_actions_dictionary = dict()
        self.expected_actions_dictionary["P"] = [ Action("a", 1.0, "P3"),
                                                  Action("b", 1.0, "P3") ]
        self.expected_actions_dictionary["P1" ] = [ Action("a", 1.0, "P3") ]
        self.expected_actions_dictionary["P2" ] = [ Action("b", 1.0, "P3") ]
        self.expected_actions_dictionary["P3" ] = [ Action("c", 1.0, "P") ]

        self.expected_initial_state = "P"
        self.expected_state_space_size = 4

    @unittest.expectedFailure
    def test_actions(self):
        super(TestChoiceAlias, self).test_actions()

    @unittest.expectedFailure
    def test_state_space_size(self):
        super(TestChoiceAlias, self).test_state_space_size()

    @unittest.expectedFailure
    def test_generator_matrix(self):
        super(TestChoiceAlias, self).test_generator_matrix()

    @unittest.expectedFailure
    def test_steady_state_solve(self):
        super(TestChoiceAlias, self).test_steady_state_solve()

    @unittest.expectedFailure
    def test_utilisations(self):
        super(TestChoiceAlias, self).test_utilisations()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
