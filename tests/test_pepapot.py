﻿#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pepapot
----------------------------------

Tests for `pepapot` module.
"""

import random
import unittest
import io
import logging
import functools

from pepapot import pepapot
Action = pepapot.Action

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


def is_valid_gen_matrix(testcase, model_solver):
    for (row_number, row) in enumerate(model_solver.gen_matrix):
        testcase.assertAlmostEqual(0.0, sum(row))
        testcase.assertTrue(row[row_number] < 0.0)


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
        self.expected_actions_dictionary["P"] = [Action("a", 1.0, "P1")]
        self.expected_actions_dictionary["P1"] = [Action("b", 1.0, "P")]
        self.expected_actions_dictionary["Q"] = [Action("a", 1.0, "Q1")]
        self.expected_actions_dictionary["Q1"] = [Action("b", 1.0, "Q")]

        self.expected_shared_actions = set()

        self.expected_initial_state = ("P", "Q")
        self.expected_state_space_size = 4

        self.expected_solution = [(("P", "Q"), 0.25),
                                  (("P1", "Q"), 0.25),
                                  (("P", "Q1"), 0.25),
                                  (("P1", "Q1"), 0.25)]

        self.expected_utilisations = [dict([("P", 0.5),
                                            ("P1", 0.5)]),
                                      dict([("Q", 0.5),
                                            ("Q1", 0.5)])]

    def assertAlmostEqual(self, a, b, msg=None):
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
        message = str(a) + " is not approximately " + str(b)
        if msg is not None:
            message = msg + "\n   " + message
        self.assertTrue((abs(a - b)) < 1e-8, msg=message)

    # I had separate methods for testing each of these things, but I found
    # that unittest re-created this class for each test, hence I was not
    # remembering the previous computed values. Not a huge problem as
    # everything was computed fast enough. But still, I think I prefer this.
    def test_everything(self):
        model = pepapot.parse_model(self.model_source)
        model_solver = pepapot.ModelSolver(model)

        # Test the parser
        shared_actions = model.system_equation.get_shared_actions()
        self.assertEqual(shared_actions, self.expected_shared_actions)

       # Test the defined names
        defined_names = model.defined_process_names()
        self.assertEqual(defined_names, self.expected_defined_process_names)

        # Test the used names
        used_names = model.used_process_names()
        self.assertEqual(used_names, self.expected_used_process_names)

        # Test the set of actions
        actual_actions = model.get_process_actions()
        self.assertEqual(actual_actions, self.expected_actions_dictionary)

        # Test the initial state
        self.assertEqual(model_solver.initial_state,
                         self.expected_initial_state)

        # Test the size of the state space produced
        model_solver.log_state_space()
        self.assertEqual(len(model_solver.state_space),
                         self.expected_state_space_size)

        # Test the generator matrix
        is_valid_gen_matrix(self, model_solver)

        # Test the steady state solution
        steady_solution = model_solver.steady_solution
        self.assertAlmostEqual(sum(steady_solution), 1.0,
                               msg="Probabilities do not add up to one")
        for (state, expected_probability) in self.expected_solution:
            state_number, transitions = model_solver.state_space[state]
            probability = steady_solution[state_number]
            message = ("Probability for state: " + str(state) +
                       " is calculated as " + str(probability) +
                       " rather than the expected " +
                       str(expected_probability))
            self.assertAlmostEqual(probability, expected_probability,
                                   msg=message)

        # Test the steady state utilisations
        steady_utilisations = model_solver.steady_utilisations
        for actual_utils, expected_utils in zip(steady_utilisations,
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
        self.expected_solution = [(("P", "Q"), 0.4),
                                  (("P1", "Q"), 0.2),
                                  (("P", "Q1"), 0.2),
                                  (("P1", "Q1"), 0.2)]
        self.expected_utilisations = [dict([("P", 0.6),
                                            ("P1", 0.4)]),
                                      dict([("Q", 0.6),
                                            ("Q1", 0.4)])]


class TestSimpleDoubleCoop(TestSimpleNoCoop):
    """Similar to the above case we're only using super here because we can
       and so many of the expected results are the same.
    """
    def setUp(self):
        super(TestSimpleDoubleCoop, self).setUp()
        self.model_source = simple_components + "\nP <a, b> Q"
        self.expected_shared_actions = set(["a", "b"])
        self.expected_state_space_size = 2
        self.expected_solution = [(("P", "Q"), 0.5),
                                  (("P1", "Q1"), 0.5)]
        self.expected_utilisations = [dict([("P", 0.5),
                                            ("P1", 0.5)]),
                                      dict([("Q", 0.5),
                                            ("Q1", 0.5)])]


class TestApparentRate(TestSimpleNoCoop):
    """A test designed to test the computation of a simple apparent rate"""
    def setUp(self):
        self.model_source = """
P = (a, 1.0).P1;
P1 = (b, 1.0).P;

Q = (a, 1.0).Q1;
Q1 = (b, 2.0).Q;

R = (a, 1.0).R1;
R1 = (b, 10.0).R;

R <b> (P || Q)
        """
        self.expected_shared_actions = set(["b"])

        set_of_names = set(["P", "P1", "Q", "Q1", "R", "R1"])
        self.expected_used_process_names = set_of_names
        self.expected_defined_process_names = set_of_names

        self.expected_actions_dictionary = dict()
        self.expected_actions_dictionary["P"] = [Action("a", 1.0, "P1")]
        self.expected_actions_dictionary["P1"] = [Action("b", 1.0, "P")]
        self.expected_actions_dictionary["Q"] = [Action("a", 1.0, "Q1")]
        self.expected_actions_dictionary["Q1"] = [Action("b", 2.0, "Q")]
        self.expected_actions_dictionary["R"] = [Action("a", 1.0, "R1")]
        self.expected_actions_dictionary["R1"] = [Action("b", 10.0, "R")]

        self.expected_initial_state = ("R", ("P", "Q"))
        self.expected_state_space_size = 8

        self.expected_solution = [(("R", ("P", "Q")), 0.0684931506849),
                                  (("R", ("P1", "Q")), 0.184931506849),
                                  (("R", ("P", "Q1")), 0.109589041096),
                                  (("R", ("P1", "Q1")), 0.294520547945),

                                  (("R1", ("P", "Q")), 0.0342465753425),
                                  (("R1", ("P1", "Q")), 0.109589041096),
                                  (("R1", ("P", "Q1")), 0.0479452054795),
                                  (("R1", ("P1", "Q1")), 0.150684931507)]

        self.expected_utilisations = [dict([("R", 0.657534246575),
                                            ("R1", 0.342465753425)]),
                                      dict([("P", 0.260273972603),
                                            ("P1", 0.739726027397)]),
                                      dict([("Q", 0.397260273973),
                                            ("Q1", 0.602739726027)])]


class TestSimpleArray(TestSimpleNoCoop):
    def setUp(self):
        super(TestSimpleArray, self).setUp()
        self.model_source = simple_components + "\nP[3] <a> Q[3]"
        self.expected_shared_actions = set(["a"])
        self.expected_state_space_size = 16
        self.expected_initial_state = ((('P', 3), ('P1', 0)),
                                       (('Q', 3), ('Q1', 0)))
        self.expected_solution = [
            (((('P', 3), ('P1', 0)),
              (('Q', 3), ('Q1', 0))), 0.057908355442009764),
            (((('P', 2), ('P1', 1)),
              (('Q', 2), ('Q1', 1))), 0.16193377223150243),
            (((('P', 1), ('P1', 2)),
              (('Q', 1), ('Q1', 2))), 0.10592512528249978),
            (((('P', 3), ('P1', 0)),
              (('Q', 2), ('Q1', 1))), 0.08686253316301464),
            (((('P', 2), ('P1', 1)),
              (('Q', 3), ('Q1', 0))), 0.08686253316301465),
            (((('P', 0), ('P1', 3)),
              (('Q', 0), ('Q1', 3))), 0.017654187547083297),
            (((('P', 2), ('P1', 1)),
              (('Q', 1), ('Q1', 2))), 0.11850250564999511),
            (((('P', 1), ('P1', 2)),
              (('Q', 2), ('Q1', 1))), 0.11850250564999514),
            (((('P', 1), ('P1', 2)),
              (('Q', 0), ('Q1', 3))), 0.03429301365824899),
            (((('P', 0), ('P1', 3)),
              (('Q', 1), ('Q1', 2))), 0.03429301365824901),
            (((('P', 3), ('P1', 0)),
              (('Q', 1), ('Q1', 2))), 0.049326913628770744),
            (((('P', 1), ('P1', 2)),
              (('Q', 3), ('Q1', 0))), 0.04932691362877076),
            (((('P', 2), ('P1', 1)),
              (('Q', 0), ('Q1', 3))), 0.029478235236317183),
            (((('P', 0), ('P1', 3)),
              (('Q', 2), ('Q1', 1))), 0.029478235236317187),
            (((('P', 3), ('P1', 0)),
              (('Q', 0), ('Q1', 3))), 0.009826078412105728),
            (((('P', 0), ('P1', 3)),
              (('Q', 3), ('Q1', 0))), 0.009826078412105735)]
        self.expected_utilisations = [dict([("P", 1.7133732927188765),
                                            ("P1", 1.2866267072811248)]),
                                      dict([("Q", 1.7133732927188765),
                                            ("Q1", 1.2866267072811248)])]


class TestSelfLoopArray(TestSimpleNoCoop):
    """Test whether we correctly deal with a process with a self-loop. Partly
       this test was added to gain extra coverage of tests which were missing
       a couple of lines which expressely dealt with that situation for
       aggregation.
    """
    def setUp(self):
        self.model_source = """P = (a, 1.0).P;
                               Q = (a, 1.0).Q1;
                               Q1 = (b, 1.0).Q;
                               P[3] <a> Q[3]
                            """
        self.expected_used_process_names = set(["P", "Q", "Q1"])
        self.expected_defined_process_names = set(["P", "Q", "Q1"])

        self.expected_actions_dictionary = dict()
        self.expected_actions_dictionary["P"] = [Action("a", 1.0, "P")]
        self.expected_actions_dictionary["Q"] = [Action("a", 1.0, "Q1")]
        self.expected_actions_dictionary["Q1"] = [Action("b", 1.0, "Q")]
        self.expected_shared_actions = set(["a"])
        self.expected_state_space_size = 4
        self.expected_initial_state = ((('P', 3),),
                                       (('Q', 3), ('Q1', 0)))
        self.expected_solution = [(((('P', 3),),
                                    (('Q', 3), ('Q1', 0))), 0.125),
                                  (((('P', 3),),
                                    (('Q', 2), ('Q1', 1))), 0.375),
                                  (((('P', 3),),
                                    (('Q', 1), ('Q1', 2))), 0.375),
                                  (((('P', 3),),
                                    (('Q', 0), ('Q1', 3))), 0.125)
                                  ]
        self.expected_utilisations = [dict([("P", 3.0)]),
                                      dict([("Q", 1.5),
                                            ("Q1", 1.5)
                                            ])]


class TestThreeStateArray(TestSimpleNoCoop):
    """Test whether we correctly deal with a process with a self-loop. Partly
       this test was added to gain extra coverage of tests which were missing
       a couple of lines which expressely dealt with that situation for
       aggregation.
    """
    def setUp(self):
        self.model_source = """P  = (a, 1.0).P1;
                               P1 = (b, 1.0).P2;
                               P2 = (c, 1.0).P;
                               P[3]
                            """
        self.expected_used_process_names = set(["P", "P1", "P2"])
        self.expected_defined_process_names = set(["P", "P1", "P2"])

        self.expected_actions_dictionary = dict()
        self.expected_actions_dictionary["P"] = [Action("a", 1.0, "P1")]
        self.expected_actions_dictionary["P1"] = [Action("b", 1.0, "P2")]
        self.expected_actions_dictionary["P2"] = [Action("c", 1.0, "P")]
        self.expected_shared_actions = set([])
        self.expected_state_space_size = 10
        self.expected_initial_state = (('P', 3), ('P1', 0), ('P2', 0))
        self.expected_solution = [
            ((('P', 3), ('P1', 0), ('P2', 0)), 0.03703703703703704),
            ((('P', 2), ('P1', 1), ('P2', 0)), 0.11111111111111112),
            ((('P', 2), ('P1', 0), ('P2', 1)), 0.11111111111111113),
            ((('P', 1), ('P1', 2), ('P2', 0)), 0.11111111111111112),
            ((('P', 1), ('P1', 1), ('P2', 1)), 0.22222222222222227),
            ((('P', 1), ('P1', 0), ('P2', 2)), 0.11111111111111112),
            ((('P', 0), ('P1', 3), ('P2', 0)), 0.03703703703703704),
            ((('P', 0), ('P1', 2), ('P2', 1)), 0.11111111111111113),
            ((('P', 0), ('P1', 1), ('P2', 2)), 0.11111111111111113),
            ((('P', 0), ('P1', 0), ('P2', 3)), 0.03703703703703705)
            ]
        self.expected_utilisations = [dict([("P", 1.0),
                                            ("P1", 1.0),
                                            ("P2", 1.0)
                                            ])]


class TestSimpleAlias(TestSimpleNoCoop):
    """Similar to the above case we're only using super here because we can
       and so many of the expected results are the same.
    """
    def setUp(self):
        super(TestSimpleAlias, self).setUp()
        self.model_source = "A = P;\n" + simple_components + "\nP || Q"
        self.expected_defined_process_names.add("A")
        a_actions = self.expected_actions_dictionary["P"]
        self.expected_actions_dictionary["A"] = a_actions

    # Note, if you expect everything to fail, you can decorate the class with
    # unittest.expectedFailure, however I prefer this as if you decorate the
    # entire class, then it is essentially the same as skipping the tests, that
    # is no report is given saying how many expected failures there were.
    # Additionally note that I would like to be able to have more fine-grained
    # control on what I expect to fail here. For example I do not expect
    # parsing to fail here.
    @unittest.expectedFailure
    def test_everything(self):
        super(TestSimpleAlias, self).test_actions()


class TestSimpleChoice(TestSimpleNoCoop):
    def setUp(self):
        self.model_source = simple_choice_component + "\nP"

        self.expected_used_process_names = set(["P", "P1", "P2"])
        self.expected_defined_process_names = self.expected_used_process_names

        self.expected_actions_dictionary = dict()
        self.expected_actions_dictionary["P"] = [Action("a", 1.0, "P1"),
                                                 Action("b", 1.0, "P2")]
        self.expected_actions_dictionary["P1"] = [Action("c", 1.0, "P")]
        self.expected_actions_dictionary["P2"] = [Action("d", 1.0, "P")]

        self.expected_shared_actions = set()

        self.expected_initial_state = "P"
        self.expected_state_space_size = 3

        self.expected_solution = [("P", 1.0 / 3.0),
                                  ("P1", 1.0 / 3.0),
                                  ("P2", 1.0 / 3.0)]

        self.expected_utilisations = [dict(self.expected_solution)]


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
        self.expected_actions_dictionary["P"] = [Action("a", 1.0, "P3"),
                                                 Action("b", 1.0, "P3")]
        self.expected_actions_dictionary["P1"] = [Action("a", 1.0, "P3")]
        self.expected_actions_dictionary["P2"] = [Action("b", 1.0, "P3")]
        self.expected_actions_dictionary["P3"] = [Action("c", 1.0, "P")]

        self.expected_initial_state = "P"
        self.expected_state_space_size = 4

    @unittest.expectedFailure
    def test_everything(self):
        super(TestChoiceAlias, self).test_actions()


# The goal is to build a method which will generate a random PEPA model. This
# can then be used to do some randomised testing. To do that we require to
# have some properties about the results which we can test. The first and
# most obvious is simply that parsing has been successful.
class RandomPepa(object):
    def __init__(self):
        self.processes = []
        self.process_definitions = []

    def generate_process_definitions(self):
        for i in range(random.randint(1, 4)):
            head_name = "P_" + str(i) + "_0"
            self.processes.append(head_name)

            tail_name = "P_" + str(i) + "_1"
            head_successor = pepapot.ProcessIdentifier(tail_name)
            head_rhs = pepapot.PrefixNode("a", 1.0, head_successor)
            head_definition = pepapot.ProcessDefinition(head_name, head_rhs)
            self.process_definitions.append(head_definition)

            tail_successor = pepapot.ProcessIdentifier(head_name)
            tail_rhs = pepapot.PrefixNode("b", 1.0, tail_successor)
            tail_definition = pepapot.ProcessDefinition(tail_name, tail_rhs)
            self.process_definitions.append(tail_definition)

    def generate_system_equation(self):
        def combine(left, right):
            return pepapot.ParsedSystemCooperation(left, [], right)
        processes = [pepapot.ParsedNamedComponent(x) for x in self.processes]
        self.system_equation = functools.reduce(combine, processes)

    def generate_model(self):
        self.generate_process_definitions()
        self.generate_system_equation()
        self.model = pepapot.ParsedModel(self.process_definitions,
                                         self.system_equation)

    def get_model_source(self):
        return self.model.format()


class TestRandom(unittest.TestCase):
    def setUp(self):
        self.random_pepa = RandomPepa()
        self.random_pepa.generate_model()
        self.model_source = self.random_pepa.get_model_source()

    def test_model(self):
        for i in range(10):
            logging.info("The random model source:")
            logging.info(self.model_source)
            model = pepapot.parse_model(self.model_source)
            model_solver = pepapot.ModelSolver(model)
            is_valid_gen_matrix(self, model_solver)


class TestCommandLine(unittest.TestCase):
    def test_simple(self):
        memory_file = io.StringIO()
        pepapot.run_command_line(memory_file, ["steady", "util",
                                               "models/simple.pepa"])
        actual_output = memory_file.getvalue()
        actual_lines = actual_output.split("\n")
        expected_lines = ["P1 : 0.4", "P : 0.6", "Q : 0.6", "Q1 : 0.4"]
        for line in expected_lines:
            self.assertIn(line, actual_lines)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
