#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pypepa
----------------------------------

Tests for `pypepa` module.
"""

import unittest
import logging

from pypepa import pypepa

simple_components = """
P = (a,r).P1;
P1 = (b, r).P;

Q = (a,r).Q1;
Q1 = (b, r).Q;
"""

simple_choice_component = """
P = (a, r).P1 + (b, r).P2;
P1 = (c, r).P;
P2 = (d, r).P;
"""

class TestModelBase(unittest.TestCase):
    """A simple couple of definitions with a system equation involving no
       cooperation. This also acts as a good base class to inherit from for
       testing a model. One needs only to override the model definition and
       the expected answers.
    """
    def setUp(self):
        self.model_source = simple_components + "\nP || Q"
        # It would be nice to actually test the parsing but that doesn't
        # seem to be possible.
        self.model = pypepa.parse_model(self.model_source)
        self.expected_used_process_names = set(["P", "P1", "Q", "Q1"])
        self.expected_defined_process_names = set(["P", "P1", "Q", "Q1"])
    
        self.expected_actions_dictionary = dict()
        self.expected_actions_dictionary["P"] = [ "a" ]
        self.expected_actions_dictionary["P1" ] = [ "b" ]
        self.expected_actions_dictionary["Q" ] = [ "a" ]
        self.expected_actions_dictionary["Q1" ] = [ "b" ]
    
        self.expected_successors_dictionary = dict()
        self.expected_successors_dictionary["P"] = [ "P1" ]
        self.expected_successors_dictionary["P1"] = [ "P" ]
        self.expected_successors_dictionary["Q"] = [ "Q1" ]
        self.expected_successors_dictionary["Q1"] = [ "Q" ]

    def test_used_names(self):
        used_names = self.model.used_process_names()
        self.assertEqual(used_names, self.expected_used_process_names)

    def test_defined_names(self):
        defined_names = pypepa.defined_process_names(self.model)
        self.assertEqual(defined_names, self.expected_defined_process_names)

    # TODO: Somehow test the correct parsing of coopertions (system equation)

    def test_actions(self):
        actual_actions = self.model.get_process_actions()
        self.assertEqual(actual_actions, self.expected_actions_dictionary)

    def test_successors(self):
        actual_successors = self.model.get_successors()
        self.assertEqual(actual_successors, self.expected_successors_dictionary)

class TestSimpleSingleCoop(TestModelBase):
    def setUp(self):
        # This model has most of the same results as the base test case
        # So I'll just call super here rather than copy and paste, but in
        # general it shouldn't be necessary to call super here.
        super(TestSimpleSingleCoop, self).setUp()
        self.model_source = simple_components + "\nP <a> Q"
        self.model = pypepa.parse_model(self.model_source)

class TestSimpleDoubleCoop(TestModelBase):
    def setUp(self):
        # Similar to the above case we're only using super here because we can
        # and so many of th expected results are the same.
        super(TestSimpleDoubleCoop, self).setUp()
        self.model_source = simple_components + "\nP <a, b> Q"
        self.model = pypepa.parse_model(self.model_source)

class TestSimpleAlias(TestModelBase):
    # TODO: Find out a way to still use @unittest.expectedFailure
    # I think I can just override each expected failure case but still call
    # the super version of it.
    def setUp(self):
        # Similar to the above case we're only using super here because we can
        # and so many of th expected results are the same.
        super(TestSimpleAlias, self).setUp()
        self.model_source = "A = P;\n" + simple_components + "\nP || Q"
        self.model = pypepa.parse_model(self.model_source)

        self.expected_defined_process_names.add("A")
    
        self.expected_actions_dictionary["A"] = self.expected_actions_dictionary["P"]
        self.expected_successors_dictionary["A"] = self.expected_successors_dictionary["P"] 

class TestSimpleChoice(TestModelBase):
    def setUp(self):
        self.model_source = simple_choice_component + "\nP"
        self.model = pypepa.parse_model(self.model_source)

        self.expected_used_process_names = set(["P", "P1", "P2"])
        self.expected_defined_process_names = self.expected_used_process_names
    
        self.expected_actions_dictionary = dict()
        self.expected_actions_dictionary["P"] = [ "a", "b" ]
        self.expected_actions_dictionary["P1" ] = [ "c" ]
        self.expected_actions_dictionary["P2" ] = [ "d" ]
    
        self.expected_successors_dictionary = dict()
        self.expected_successors_dictionary["P"] = [ "P1", "P2" ]
        self.expected_successors_dictionary["P1"] = [ "P" ]
        self.expected_successors_dictionary["P2"] = [ "P" ]

class TestPypepa(unittest.TestCase):
    """A simple test only because I'm not sure how to generically test the
       parsing of the system equation. Once I figure that out I can move this
       into TestModelBase
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
