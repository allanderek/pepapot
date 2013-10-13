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
Action = pypepa.Action

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
        self.expected_actions_dictionary["P"] = [ Action("a", "r", "P1") ]
        self.expected_actions_dictionary["P1" ] = [ Action ("b", "r", "P") ]
        self.expected_actions_dictionary["Q" ] = [ Action("a", "r", "Q1") ]
        self.expected_actions_dictionary["Q1" ] = [ Action("b", "r", "Q") ]
    
        self.expected_successors_dictionary = dict()
        self.expected_successors_dictionary["P"] = [ "P1" ]
        self.expected_successors_dictionary["P1"] = [ "P" ]
        self.expected_successors_dictionary["Q"] = [ "Q1" ]
        self.expected_successors_dictionary["Q1"] = [ "Q" ]

        self.expected_initial_state = [ "P", "Q" ]
        self.expected_state_space_size = 4

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

    def test_initial_state(self):
        initial_state = self.model.get_initial_state()
        self.assertEqual(initial_state.local_states, self.expected_initial_state)

    def test_state_space_size(self):
        state_space = pypepa.build_state_space(self.model)
        self.assertEqual(state_space.size, self.expected_state_space_size)

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
        self.expected_state_space_size = 2

class TestSimpleAlias(TestModelBase):
    def setUp(self):
        # Similar to the above case we're only using super here because we can
        # and so many of th expected results are the same.
        super(TestSimpleAlias, self).setUp()
        self.model_source = "A = P;\n" + simple_components + "\nP || Q"
        self.model = pypepa.parse_model(self.model_source)

        self.expected_defined_process_names.add("A")
    
        self.expected_actions_dictionary["A"] = self.expected_actions_dictionary["P"]
        self.expected_successors_dictionary["A"] = self.expected_successors_dictionary["P"] 

    # Note, if you expect everything to fail, you can decorate the class with
    # unittest.expectedFailure, however I prefer this as if you decorate the
    # entire class, then it is essentially the same as skipping the tests, that
    # is no report is given saying how many expected failures there were.
    @unittest.expectedFailure
    def test_actions(self):
        super(TestSimpleAlias, self).test_actions()

    @unittest.expectedFailure
    def test_successors(self):
        super(TestSimpleAlias, self).test_successors()

class TestSimpleChoice(TestModelBase):
    def setUp(self):
        self.model_source = simple_choice_component + "\nP"
        self.model = pypepa.parse_model(self.model_source)

        self.expected_used_process_names = set(["P", "P1", "P2"])
        self.expected_defined_process_names = self.expected_used_process_names
    
        self.expected_actions_dictionary = dict()
        self.expected_actions_dictionary["P"] = [ Action("a", "r", "P1"),
                                                  Action("b", "r", "P2") ]
        self.expected_actions_dictionary["P1" ] = [ Action("c", "r", "P") ]
        self.expected_actions_dictionary["P2" ] = [ Action("d", "r", "P") ]
    
        self.expected_successors_dictionary = dict()
        self.expected_successors_dictionary["P"] = [ "P1", "P2" ]
        self.expected_successors_dictionary["P1"] = [ "P" ]
        self.expected_successors_dictionary["P2"] = [ "P" ]

        self.expected_initial_state = [ "P" ]
        self.expected_state_space_size = 3

class TestChoiceAlias(TestModelBase):
    def setUp(self):
        self.model_source = """P = P1 + P2;
                               P1 = (a, r).P3;
                               P2 = (b, r).P3;
                               P3 = (c, r).P;
                               P
                            """
        self.model = pypepa.parse_model(self.model_source)

        self.expected_used_process_names = set(["P", "P1", "P2", "P3"])
        self.expected_defined_process_names = self.expected_used_process_names
    
        self.expected_actions_dictionary = dict()
        self.expected_actions_dictionary["P"] = [ Action("a", "r", "P3"),
                                                  Action("b", "r", "P3") ]
        self.expected_actions_dictionary["P1" ] = [ Action("a", "r", "P3") ]
        self.expected_actions_dictionary["P2" ] = [ Action("b", "r", "P3") ]
        self.expected_actions_dictionary["P3" ] = [ Action("c", "r", "P") ]
    
        self.expected_successors_dictionary = dict()
        self.expected_successors_dictionary["P"] = [ "P3" ]
        self.expected_successors_dictionary["P1"] = [ "P3" ]
        self.expected_successors_dictionary["P2"] = [ "P3" ]
        self.expected_successors_dictionary["P3"] = [ "P" ]

        self.expected_initial_state = [ "P" ]
        self.expected_state_space_size = 4

    @unittest.expectedFailure
    def test_actions(self):
        super(TestChoiceAlias, self).test_actions()

    @unittest.expectedFailure
    def test_successors(self):
        super(TestChoiceAlias, self).test_successors()

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
