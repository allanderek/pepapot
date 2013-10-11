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
simple_no_coop = simple_components + "\nP || Q"
simple_single_coop = simple_components + "\nP < a > Q"
simple_double_coop = simple_components + "\nP <a,b> Q"

choice_component = """
P  (a, r).P1 + (b, r).P2;
P1 = (b, r).P;
P2 = (c, r).P;

P
"""

def create_expected_action_test(model, process, expected_actions):
    model = pypepa.parse_model(model)
    action_dictionary = model.get_process_actions()
    actual_actions = action_dictionary[process]
    self.assertEqual(actual_actions, expected_actions)
    
class TestPypepa(unittest.TestCase):
    def test_used_names(self):
        model = pypepa.parse_model(simple_no_coop)
        used_names = model.used_process_names()
        expected = set(["P", "P1", "Q", "Q1"])
        self.assertEqual(used_names, expected)

        simple_no_q = simple_components + "\nP"
        model = pypepa.parse_model(simple_no_q)
        used_names = model.used_process_names()
        expected = set(["P", "P1"])
        self.assertEqual(used_names, expected)

    def test_defined_names(self):
        model = pypepa.parse_model(simple_no_coop)
        defined_names = pypepa.defined_process_names(model)
        expected = set(["P", "P1", "Q", "Q1"])
        self.assertEqual(defined_names, expected)

    def test_cooperation_parser(self):
        model = pypepa.parse_model(simple_single_coop)
        self.assertEqual(model.system_equation.cooperation_set, ["a"])
        model = pypepa.parse_model(simple_no_coop)
        self.assertEqual(model.system_equation.cooperation_set, [])
        model = pypepa.parse_model(simple_double_coop)
        self.assertEqual(model.system_equation.cooperation_set, ["a", "b"])

    def test_actions(self):
        create_expected_action_test(simple_single_coop, "P", ["a"])

    def test_choice(self):
        create_expected_action_test(simple_single_coop, "P", ["a", "b"])

class ExpectedFailureTestCase(unittest.TestCase):
    @unittest.expectedFailure
    def test_aliases(self):
        model = "A = P;\n" + simple_no_coop
        create_expected_action_test(model, "A", "a")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
