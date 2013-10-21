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
        # It would be nice to actually test the parsing but that doesn't
        # seem to be possible.
        self.model = pypepa.parse_model(self.model_source)
        self.expected_used_process_names = set(["P", "P1", "Q", "Q1"])
        self.expected_defined_process_names = set(["P", "P1", "Q", "Q1"])

        self.expected_actions_dictionary = dict()
        self.expected_actions_dictionary["P"] = [ Action("a", 1.0, "P1") ]
        self.expected_actions_dictionary["P1" ] = [ Action ("b", 1.0, "P") ]
        self.expected_actions_dictionary["Q" ] = [ Action("a", 1.0, "Q1") ]
        self.expected_actions_dictionary["Q1" ] = [ Action("b", 1.0, "Q") ]

        self.expected_initial_state = ("P", "Q")
        self.expected_state_space_size = 4

        self.expected_solution = [ (("P", "Q"), 0.25),
                                   (("P1", "Q"), 0.25),
                                   (("P", "Q1"), 0.25),
                                   (("P1", "Q1"), 0.25)
                                 ]


    def test_used_names(self):
        used_names = self.model.used_process_names()
        self.assertEqual(used_names, self.expected_used_process_names)

    def test_defined_names(self):
        defined_names = self.model.defined_process_names()
        self.assertEqual(defined_names, self.expected_defined_process_names)

    # TODO: Somehow test the correct parsing of coopertions (system equation)

    def test_actions(self):
        actual_actions = self.model.get_process_actions()
        self.assertEqual(actual_actions, self.expected_actions_dictionary)

    def test_initial_state(self):
        initial_state = self.model.get_initial_state()
        self.assertEqual(initial_state, self.expected_initial_state)

    def test_state_space_size(self):
        state_space = pypepa.build_state_space(self.model)
        self.assertEqual(len(state_space), self.expected_state_space_size)

    def test_generator_matrix(self):
        state_space = pypepa.build_state_space(self.model)
        gen_matrix  = pypepa.get_generator_matrix(state_space)
        for (row_number, row) in enumerate(gen_matrix):
            self.assertEqual(0.0, sum(row))
            self.assertTrue(row[row_number] < 0.0)

    def test_steady_state_solve(self):
        state_space = pypepa.build_state_space(self.model)
        gen_matrix = pypepa.get_generator_matrix(state_space)
        solution = pypepa.solve_generator_matrix(gen_matrix)
        for (state, probability) in self.expected_solution:
            state_number, transitions = state_space[state]
            # You can search a little for testing floating point numbers
            # for equality and conceivably we could do so with respect to an
            # absolute tolerance and a relative tolerance, for now I believe
            # this is sufficient.
            difference = abs(probability - solution[state_number])
            self.assertTrue(difference < 1e-8)

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
        self.model = pypepa.parse_model(self.model_source)
        self.expected_solution = [ (("P", "Q"), 0.4),
                                   (("P1", "Q"), 0.2),
                                   (("P", "Q1"), 0.2),
                                   (("P1", "Q1"), 0.2)
                                 ]

class TestSimpleDoubleCoop(TestSimpleNoCoop):
    """Similar to the above case we're only using super here because we can
       and so many of th expected results are the same.
    """
    def setUp(self):
        super(TestSimpleDoubleCoop, self).setUp()
        self.model_source = simple_components + "\nP <a, b> Q"
        self.model = pypepa.parse_model(self.model_source)
        self.expected_state_space_size = 2
        self.expected_solution = [ (("P", "Q"), 0.5),
                                   (("P1", "Q1"), 0.5)
                                 ]

class TestSimpleAlias(TestSimpleNoCoop):
    """Similar to the above case we're only using super here because we can
       and so many of the expected results are the same.
    """
    def setUp(self):
        super(TestSimpleAlias, self).setUp()
        self.model_source = "A = P;\n" + simple_components + "\nP || Q"
        self.model = pypepa.parse_model(self.model_source)

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
    def test_generator_matrix(self):
        super(TestSimpleAlias, self).test_steady_state_solve()

class TestSimpleChoice(TestSimpleNoCoop):
    def setUp(self):
        self.model_source = simple_choice_component + "\nP"
        self.model = pypepa.parse_model(self.model_source)

        self.expected_used_process_names = set(["P", "P1", "P2"])
        self.expected_defined_process_names = self.expected_used_process_names

        self.expected_actions_dictionary = dict()
        self.expected_actions_dictionary["P"] = [ Action("a", 1.0, "P1"),
                                                  Action("b", 1.0, "P2") ]
        self.expected_actions_dictionary["P1" ] = [ Action("c", 1.0, "P") ]
        self.expected_actions_dictionary["P2" ] = [ Action("d", 1.0, "P") ]

        self.expected_initial_state = "P"
        self.expected_state_space_size = 3

class TestChoiceAlias(TestSimpleNoCoop):
    def setUp(self):
        self.model_source = """P = P1 + P2;
                               P1 = (a, 1.0).P3;
                               P2 = (b, 1.0).P3;
                               P3 = (c, 1.0).P;
                               P
                            """
        self.model = pypepa.parse_model(self.model_source)

        self.expected_used_process_names = set(["P", "P1", "P2", "P3"])
        self.expected_defined_process_names = self.expected_used_process_names
    
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
    def test_generator_matrix(self):
        super(TestChoiceAlias, self).test_steady_state_solve()

class TestPypepa(unittest.TestCase):
    """A simple test only because I'm not sure how to generically test the
       parsing of the system equation. Once I figure that out I can move this
       into TestSimpleNoCoop
    """
    def test_cooperation_parser(self):
        model_source = simple_components + "\n P <a> P"
        model = pypepa.parse_model(model_source)
        self.assertEqual(model.system_equation.cooperation_set, ["a"])
        model_source = simple_components + "\nP || Q"
        model = pypepa.parse_model(model_source)
        self.assertEqual(model.system_equation.cooperation_set, [])
        model_source = simple_components + "\nP <a,b>Q"
        model = pypepa.parse_model(model_source)
        self.assertEqual(model.system_equation.cooperation_set, ["a", "b"])

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
