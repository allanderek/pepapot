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

simple_source = """
P = (a,r).P1;
P1 = (b, r).P;

Q = (a,r).Q1;
Q1 = (b, r).Q;

P || Q
"""

class TestPypepa(unittest.TestCase):
    def test_used_names(self):
        model = pypepa.parse_model(simple_source)
        used_names = model.used_process_names()
        expected = set(["P", "P1", "Q", "Q1"])
        self.assertEqual(used_names, expected)
        
    def test_defined_names(self):
        model = pypepa.parse_model(simple_source)
        defined_names = pypepa.defined_process_names(model)
        expected = set(["P", "P1", "Q", "Q1"])
        self.assertEqual(defined_names, expected)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
