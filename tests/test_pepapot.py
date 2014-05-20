#!/usr/bin/env python
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


## We begin with testing for the expression parser and evaluator
class TestExpression(unittest.TestCase):
    def setUp(self):
        self.expression_source = "1 + 2"
        self.expected_result = 3
        self.environment = None

    def evaluate_expression(self):
        parsed = pepapot.expr_grammar.parseString(self.expression_source,
                                                  parseAll=True)
        self.expression = parsed[0]
        result = self.expression.get_value(environment=self.environment)
        return result

    def test_evaluation(self):
        result = self.evaluate_expression()
        self.assertEqual(result, self.expected_result)


class TestNameExpression(TestExpression):
    def setUp(self):
        self.expression_source = "x * 10"
        self.expected_result = 100
        self.environment = {"x": pepapot.Expression.num_expression(10.0)}


class TestMissingNameExpression(TestExpression):
    def setUp(self):
        self.expression_source = "x * y"
        self.environment = {"x": pepapot.Expression.num_expression(10.0)}

    def test_evaluation(self):
        self.assertRaises(AssertionError, self.evaluate_expression)

        # Now test that it raises AssertionError also if we do not provide an
        # environment, this should occur for any name in the expression
        self.environment = None
        self.assertRaises(AssertionError, self.evaluate_expression)


class TestUnknownFunctionExpression(TestExpression):
    """ Simply tests we do not evaluate an unknown function to some default
    """
    def setUp(self):
        self.expression_source = "apr(1.0, x)"
        self.environment = {"x": pepapot.Expression.num_expression(10.0)}

    def test_evaluation(self):
        self.assertRaises(ValueError, self.evaluate_expression)


class TestBinopClassMethods(unittest.TestCase):
    """ Mostly here just to make sure the convenience class methods for
        creating binary operator expressions work as expected.
    """
    def setUp(self):
        # Test the expression: 1 + (2 * 3) - (10 / 5)
        self.expected_result = 5.0
        left = pepapot.Expression.num_expression(2.0)
        right = pepapot.Expression.num_expression(3.0)
        multexp = pepapot.Expression.multiply(left, right)
        left = pepapot.Expression.num_expression(10.0)
        right = pepapot.Expression.num_expression(5.0)
        divexp = pepapot.Expression.divide(left, right)
        subexp = pepapot.Expression.subtract(multexp, divexp)
        left = pepapot.Expression.num_expression(1.0)
        add_exp = pepapot.Expression.addition(left, subexp)
        self.expression = add_exp

    def test_evaulation(self):
        self.assertEqual(self.expression.get_value(), self.expected_result)


class TestConstantDefinitions(unittest.TestCase):
    def setUp(self):
        self.source = """a = 1.0;
                         b = a + 10.0;
                         c = b - 1.0;
                         d = c * c;
                         e = d / 20.0;
                         f = 2.0 ** 3;
                         g = 1.0 + 2.0 + 3.0;
                         h = 2.0 ** 3.0 ** 2.0;
                      """
        self.expected = {"a": 1.0,
                         "b": 11.0,
                         "c": 10.0,
                         "d": 100.0,
                         "e": 5.0,
                         "f": 8.0,
                         "g": 6.0,
                         "h": 512.0
                         }

    def test_evaluation(self):
        grammar = pepapot.ConstantDefinition.list_grammar
        parse_result = grammar.parseString(self.source, parseAll=True)
        definitions = parse_result[0]
        environment = pepapot.constant_def_environment(definitions)
        self.assertDictEqual(environment, self.expected)


class TestReductions(unittest.TestCase):
    """ Tests the reduction of a set of constant definitions. Note that there
        are deliberately some variables which are *not* in the environment,
        such that some of the expressions cannot be wholly reduced to a
        to a number.
    """
    def setUp(self):
        self.maxDiff = 2000
        self.source = """x = 1.0;
                         y = x + 2.0;
                         z = q + 2.0 * 3.0;
                         a = x + y;
                         b = x + y + z;
                      """
        self.expected = """ x = 1.0;
                            y = 3.0;
                            z = q + 6.0;
                            a = 4.0;
                            b = 4.0 + (q + 6.0);
                        """

    def test_reduction(self):
        grammar = pepapot.ConstantDefinition.list_grammar
        source_parse = grammar.parseString(self.source, parseAll=True)
        source_defs = source_parse[0]

        expected_parse = grammar.parseString(self.expected, parseAll=True)
        expected_defs = expected_parse[0]

        pepapot.reduce_definitions(source_defs)
        source_env = pepapot.definition_environment(source_defs)
        expected_env = pepapot.definition_environment(expected_defs)
        self.assertDictEqual(source_env, expected_env)


top_rate = pepapot.TopRate()


class TestTopRateArithmetic(unittest.TestCase):
    """ Just some simple tests on the arithmetic involved with TopRate.
        Essentially we are testing that the arithmetic which will be used in
        the apparent rate calculations works correctly.
    """
    def test_everything(self):
        value = top_rate + 10
        self.assertEqual(value, top_rate)
        value = 10 + top_rate
        self.assertEqual(value, top_rate)

        value = sum([1.0, 2.0, 3.0, top_rate])
        self.assertEqual(value, top_rate)

        value = 2.0 / top_rate
        self.assertEqual(value, 0)

        value = top_rate / 2.0
        self.assertEqual(value, top_rate)

        value = top_rate / top_rate
        self.assertEqual(value, 1)

        value = 3.0 * top_rate
        self.assertEqual(value, top_rate)

        value = (3.0 * top_rate) / top_rate
        self.assertEqual(value, 1)

        value = 3.0 * (top_rate / top_rate)
        self.assertEqual(value, 3)

        # It is just possible, that the left hand rate of a cooperation is 0
        # and the right hand rate of a cooperation is T, in which case the
        # shared action should have rate 0. Although the way the apparent
        # rate calculation is currently done:
        # r1/r_a1 * r2/r_a2 * min(r_a1, r_a2)
        # If say r1 was 0 and r2 was T, then min(r_a1, r_a2) would be zero
        # and r2/r_a2 would be 1, so we would not actually multiply T by
        # anything. However, in case we refactor that, it is good to keep this
        # test here.
        value = 0 * top_rate
        self.assertEqual(value, 0)
        value = top_rate * 0
        self.assertEqual(value, 0)


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
one_expr = pepapot.Expression.num_expression(1.0)
two_expr = pepapot.Expression.num_expression(2.0)
r_expr = pepapot.Expression.name_expression("r")
s_expr = pepapot.Expression.name_expression("s")


def is_valid_gen_matrix(testcase, model_solver):
    for (row_number, row) in enumerate(model_solver.gen_matrix):
        testcase.assertAlmostEqual(0.0, sum(row))
        testcase.assertTrue(row[row_number] < 0.0)


class TestSimpleNoCoop(unittest.TestCase):
    """This tests a very simple test model. It also serves as a base class
       from which all other cases testing particular models should derive.
       A subclass should re-write the setup method populating all of the
       expected values with appropriate ones.
    """
    def setUp(self):
        self.model_source = simple_components + "\nP || Q"
        self.expected_used_process_names = set(["P", "P1", "Q", "Q1"])
        self.expected_defined_process_names = set(["P", "P1", "Q", "Q1"])

        self.expected_actions_dictionary = dict()
        self.expected_actions_dictionary["P"] = [Action("a", one_expr, "P1")]
        self.expected_actions_dictionary["P1"] = [Action("b", one_expr, "P")]
        self.expected_actions_dictionary["Q"] = [Action("a", one_expr, "Q1")]
        self.expected_actions_dictionary["Q1"] = [Action("b", one_expr, "Q")]

        self.expected_shared_actions = set()

        self.expected_warnings = []
        self.expected_errors = []

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

    # I had separate methods for testing each of these things, but I found
    # that unittest re-created this class for each test, hence I was not
    # remembering the previous computed values. Not a huge problem as
    # everything was computed fast enough. But still, I think I prefer this.
    def test_everything(self):
        model = pepapot.parse_model(self.model_source)

        # Test the parser
        system_equation = model.system_equation
        shared_actions = pepapot.CompSharedActions.get_result(system_equation)
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

        # Test the static analysis
        static_analysis = model.perform_static_analysis()
        self.assertEqual(static_analysis.warnings, self.expected_warnings)
        self.assertEqual(static_analysis.errors, self.expected_errors)

        # If there are any errors then we may abandon the rest of the test
        # at this point, since we do not expect the model to compile
        if static_analysis.errors or self.expected_errors:
            return

        # Now it is time to actually build the state-space and solve the model
        model_solver = pepapot.ModelSolver(model)

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
        # Note that this means that if you set 'self.expected_solution' to []
        # you are going to pass this part of the test, but the above might be
        # enough anyway.
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

        ten_expr = pepapot.Expression.num_expression(10.0)
        self.expected_actions_dictionary = dict()
        self.expected_actions_dictionary["P"] = [Action("a", one_expr, "P1")]
        self.expected_actions_dictionary["P1"] = [Action("b", one_expr, "P")]
        self.expected_actions_dictionary["Q"] = [Action("a", one_expr, "Q1")]
        self.expected_actions_dictionary["Q1"] = [Action("b", two_expr, "Q")]
        self.expected_actions_dictionary["R"] = [Action("a", one_expr, "R1")]
        self.expected_actions_dictionary["R1"] = [Action("b", ten_expr, "R")]

        self.expected_warnings = []
        self.expected_errors = []

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


class TestApparentRateWithConstantDefs(TestApparentRate):
    def setUp(self):
        super(TestApparentRateWithConstantDefs, self).setUp()
        self.model_source = """
r = 1.0;
s = 2.0;
t = 5.0;
P = (a, r).P1;
P1 = (b, r).P;

Q = (a, r).Q1;
Q1 = (b, s).Q;

R = (a, r).R1;
R1 = (b, t * s).R;

R <b> (P || Q)
        """
        t_expr = pepapot.Expression.name_expression("t")
        t_s_expr = pepapot.Expression.multiply(t_expr, s_expr)
        self.expected_actions_dictionary = dict()
        self.expected_actions_dictionary["P"] = [Action("a", r_expr, "P1")]
        self.expected_actions_dictionary["P1"] = [Action("b", r_expr, "P")]
        self.expected_actions_dictionary["Q"] = [Action("a", r_expr, "Q1")]
        self.expected_actions_dictionary["Q1"] = [Action("b", s_expr, "Q")]
        self.expected_actions_dictionary["R"] = [Action("a", r_expr, "R1")]
        self.expected_actions_dictionary["R1"] = [Action("b", t_s_expr, "R")]


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
        self.expected_actions_dictionary["P"] = [Action("a", one_expr, "P")]
        self.expected_actions_dictionary["Q"] = [Action("a", one_expr, "Q1")]
        self.expected_actions_dictionary["Q1"] = [Action("b", one_expr, "Q")]
        self.expected_shared_actions = set(["a"])

        self.expected_warnings = []
        self.expected_errors = []

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
        self.expected_actions_dictionary["P"] = [Action("a", one_expr, "P1")]
        self.expected_actions_dictionary["P1"] = [Action("b", one_expr, "P2")]
        self.expected_actions_dictionary["P2"] = [Action("c", one_expr, "P")]
        self.expected_shared_actions = set([])

        self.expected_warnings = []
        self.expected_errors = []

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
        self.model_source = """A = P;
                               P = (a,1.0).P1;
                               P1 = (b, 1.0).A;

                               Q = (a,1.0).Q1;
                               Q1 = (b, 1.0).Q;

                               A || Q
                            """
        self.expected_defined_process_names.add("A")
        self.expected_used_process_names.add("A")

        self.expected_actions_dictionary = dict()
        self.expected_actions_dictionary["P"] = [Action("a", one_expr, "P1")]
        self.expected_actions_dictionary["P1"] = [Action("b", one_expr, "A")]
        self.expected_actions_dictionary["Q"] = [Action("a", one_expr, "Q1")]
        self.expected_actions_dictionary["Q1"] = [Action("b", one_expr, "Q")]
        self.expected_actions_dictionary["A"] = [Action("a", one_expr, "P1")]

        self.expected_initial_state = ("A", "Q")
        self.expected_state_space_size = 4

        self.expected_solution = [(("A", "Q"), 0.25),
                                  (("P1", "Q"), 0.25),
                                  (("A", "Q1"), 0.25),
                                  (("P1", "Q1"), 0.25)]

        self.expected_utilisations = [dict([("A", 0.5),
                                            ("P1", 0.5)]),
                                      dict([("Q", 0.5),
                                            ("Q1", 0.5)])]


class TestAwkwardAlias(TestSimpleNoCoop):
    def setUp(self):
        self.model_source = """A  = P;
                               P  = (a, 1.0).P1;
                               P1 = (b, 1.0).P;
                               A
                            """
        self.expected_defined_process_names = set(["A", "P", "P1"])
        self.expected_used_process_names = set(["A", "P", "P1"])

        self.expected_actions_dictionary = dict()
        self.expected_actions_dictionary["A"] = [Action("a", one_expr, "P1")]
        self.expected_actions_dictionary["P"] = [Action("a", one_expr, "P1")]
        self.expected_actions_dictionary["P1"] = [Action("b", one_expr, "P")]

        self.expected_shared_actions = set([])

        self.expected_warnings = []
        self.expected_errors = []

        self.expected_initial_state = ("A")
        self.expected_state_space_size = 2

        self.expected_solution = [(("A"), 0.5),
                                  (("P1"), 0.5)]

        self.expected_utilisations = [dict([("A", 0.5), ("P1", 0.5)])]

    # This test is non-trivial to get correct. The problem is that we end up
    # with three states rather than two. These correspond to the states:
    # A, P and P1. Of course A and P are the same state. So we need a way of
    # noting that. The alternative is just to accept that this has some
    # transient states to begin with.
    @unittest.expectedFailure
    def test_everything(self):
        super(TestAwkwardAlias, self).test_everything()


# TODO: In addition to this test we should have a test which actually has
# a complex apparent rate calculation using the T rate.
class TestTopRate(TestSimpleSingleCoop):
    """ Just a very simple test to get started on the development of support
        for the top rate T.
    """
    def setUp(self):
        super(TestTopRate, self).setUp()
        self.model_source = """P  = (a, 1.0).P1;
                               P1 = (b, 1.0).P;
                               Q  = (a, T).Q1;
                               Q1 = (b, 1.0).Q;
                               P <a> Q
                            """
        self.expected_actions_dictionary["Q"] = [Action("a", top_rate, "Q1")]


class TestTopRatePassiveCoop(TestSimpleNoCoop):
    def setUp(self):
        self.model_source = """ P  = (a, 1.0).P1;
                                P1 = (b, 1.0).P;
                                Q  = (a, 1.0).Q1;
                                Q1 = (b, T).Q;
                                R  = (a, 1.0).R1;
                                R1 = (b, T).R;
                                P <b> (Q <b> R)
                            """

        self.expected_shared_actions = set(["b"])
        process_names = set(["P", "P1", "Q", "Q1", "R", "R1"])
        self.expected_defined_process_names = process_names
        self.expected_used_process_names = process_names
        actions_dictionary = dict()
        actions_dictionary["P"] = [Action("a", one_expr, "P1")]
        actions_dictionary["P1"] = [Action("b", one_expr, "P")]
        actions_dictionary["Q"] = [Action("a", one_expr, "Q1")]
        actions_dictionary["Q1"] = [Action("b", top_rate, "Q")]
        actions_dictionary["R"] = [Action("a", one_expr, "R1")]
        actions_dictionary["R1"] = [Action("b", top_rate, "R")]
        self.expected_actions_dictionary = actions_dictionary

        self.expected_warnings = []
        self.expected_errors = []

        self.expected_initial_state = ('P', ('Q', 'R'))
        self.expected_state_space_size = 8

        self.expected_solution = [(('P', ('Q', 'R')), 0.117647058824),
                                  (('P1', ('Q', 'R')), 0.0588235294118),
                                  (('P', ('Q1', 'R')), 0.0588235294118),
                                  (('P1', ('Q1', 'R')), 0.117647058824),
                                  (('P', ('Q1', 'R1')), 0.117647058824),
                                  (('P1', ('Q1', 'R1')), 0.352941176471),
                                  (('P', ('Q', 'R1')), 0.0588235294118),
                                  (('P1', ('Q', 'R1')), 0.117647058824)]

        self.expected_utilisations = [dict([("P", 0.35294117647058831),
                                            ("P1", 0.64705882352941169)]),
                                      dict([("Q", 0.35294117647058831),
                                            ("Q1", 0.64705882352941191)]),
                                      dict([("R", 0.35294117647058831),
                                            ("R1", 0.64705882352941191)])]


class TestPassiveActiveChoice(TestSimpleNoCoop):
    def setUp(self):
        self.model_source = """ P  = (a, 2.0).P1;
                                P1 = (b, 1.0).P;
                                Q  = (a, 1.0).Q1;
                                Q1 = (b, 1.0).Q;
                                R  = (a, T).R1;
                                R1 = (b, 1.0).R;
                                P <a> (Q || R)
                            """
        self.expected_shared_actions = set(["a"])

        process_names = set(["P", "P1", "Q", "Q1", "R", "R1"])
        self.expected_defined_process_names = process_names
        self.expected_used_process_names = process_names
        actions_dictionary = dict()
        actions_dictionary["P"] = [Action("a", two_expr, "P1")]
        actions_dictionary["P1"] = [Action("b", one_expr, "P")]
        actions_dictionary["Q"] = [Action("a", one_expr, "Q1")]
        actions_dictionary["Q1"] = [Action("b", one_expr, "Q")]
        actions_dictionary["R"] = [Action("a", top_rate, "R1")]
        actions_dictionary["R1"] = [Action("b", one_expr, "R")]
        self.expected_actions_dictionary = actions_dictionary

        self.expected_warnings = []
        self.expected_errors = []

        self.expected_initial_state = ('P', ('Q', 'R'))
        self.expected_state_space_size = 8

        self.expected_solution = [(('P', ('Q', 'R')), 0.21359223301),
                                  (('P1', ('Q', 'R')), 0.2718446601942),
                                  (('P', ('Q1', 'R')), 0.0194174757282),
                                  (('P1', ('Q1', 'R')), 0.0291262135922),
                                  (('P', ('Q1', 'R1')), 0.0291262135922),
                                  (('P1', ('Q1', 'R1')), 0.0582524271845),
                                  (('P', ('Q', 'R1')), 0.135922330097),
                                  (('P1', ('Q', 'R1')), 0.242718446602)]

        self.expected_utilisations = [dict([("P", 0.39805825242718446),
                                            ("P1", 0.60194174757281549)]),
                                      dict([("Q", 0.86407766990291268),
                                            ("Q1", 0.13592233009708737)]),
                                      dict([("R", 0.53398058252427183),
                                            ("R1", 0.46601941747572811)])]


class TestSimpleChoice(TestSimpleNoCoop):
    def setUp(self):
        self.model_source = simple_choice_component + "\nP"

        self.expected_used_process_names = set(["P", "P1", "P2"])
        self.expected_defined_process_names = self.expected_used_process_names

        self.expected_actions_dictionary = dict()
        self.expected_actions_dictionary["P"] = [Action("a", one_expr, "P1"),
                                                 Action("b", one_expr, "P2")]
        self.expected_actions_dictionary["P1"] = [Action("c", one_expr, "P")]
        self.expected_actions_dictionary["P2"] = [Action("d", one_expr, "P")]

        self.expected_shared_actions = set()

        self.expected_warnings = []
        self.expected_errors = []

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
                               P3 = (c, 2.0).P;
                               P
                            """

        self.expected_used_process_names = set(["P", "P1", "P2", "P3"])
        self.expected_defined_process_names = self.expected_used_process_names

        self.expected_shared_actions = set()

        self.expected_actions_dictionary = dict()
        self.expected_actions_dictionary["P"] = [Action("a", one_expr, "P3"),
                                                 Action("b", one_expr, "P3")]
        self.expected_actions_dictionary["P1"] = [Action("a", one_expr, "P3")]
        self.expected_actions_dictionary["P2"] = [Action("b", one_expr, "P3")]
        self.expected_actions_dictionary["P3"] = [Action("c", two_expr, "P")]

        self.expected_warnings = []
        self.expected_errors = []

        self.expected_initial_state = "P"
        self.expected_state_space_size = 2

        self.expected_solution = [("P", 0.5), ("P3", 0.5)]
        self.expected_utilisations = [dict(self.expected_solution)]


class PepaUnusedRateName(TestSimpleNoCoop):
    def setUp(self):
        super(PepaUnusedRateName, self).setUp()
        self.model_source = "j = 10.0;" + self.model_source

        self.expected_warnings = [pepapot.PepaUnusedRateNameWarning("j")]


class PepaUndefinedRateName(TestSimpleNoCoop):
    def setUp(self):
        self.model_source = """r = 1.0;
                               P = (a, r).P1;
                               P1 = (b, s).P;
                               P
                            """
        self.expected_used_process_names = set(["P", "P1"])
        self.expected_defined_process_names = self.expected_used_process_names

        self.expected_shared_actions = set()

        self.expected_actions_dictionary = dict()
        self.expected_actions_dictionary["P"] = [Action("a", r_expr, "P1")]
        self.expected_actions_dictionary["P1"] = [Action("b", s_expr, "P")]

        self.expected_warnings = []
        self.expected_errors = [pepapot.PepaUndefinedRateNameError("s")]


# The goal is to build a method which will generate a random PEPA model. This
# can then be used to do some randomised testing. To do that we require to
# have some properties about the results which we can test. The first and
# most obvious is simply that parsing has been successful.
class RandomPepa(object):
    def __init__(self):
        # The list of logical processes, in other words the names which will
        # be used in the system equation
        self.processes = []
        # All the process definitions including those which are successors of
        # the names used in the system equation and hence won't be in the
        # system equation themselves.
        self.process_definitions = []

    def prefix_definition(self, name, action, rate, target_name):
        successor = pepapot.ProcessIdentifier(target_name)
        prefix_node = pepapot.PrefixNode(action, rate, successor)
        definition = pepapot.ProcessDefinition(name, prefix_node)
        self.process_definitions.append(definition)

    def choice_definition(self, name, lhs, rhs):
        choice_node = pepapot.ChoiceNode(lhs, rhs)
        definition = pepapot.ProcessDefinition(name, choice_node)
        self.process_definitions.append(definition)

    def generate_process_definitions(self):
        for i in range(random.randint(1, 4)):
            head_name = "P_" + str(i) + "_0"
            self.processes.append(head_name)

            if random.choice([True, False]):
                tail_name = "P_" + str(i) + "_1"
                self.prefix_definition(head_name, "a", 1.0, tail_name)
                self.prefix_definition(tail_name, "b", 1.0, head_name)
            else:
                left_name = "P_" + str(i) + "_l"
                right_name = "P_" + str(i) + "_r"
                left_successor = pepapot.ProcessIdentifier(left_name)
                left_prefix = pepapot.PrefixNode("a", 1.0, left_successor)
                right_successor = pepapot.ProcessIdentifier(right_name)
                right_prefix = pepapot.PrefixNode("b", 1.0, right_successor)

                self.choice_definition(head_name, left_prefix, right_prefix)
                self.prefix_definition(left_name, "c", 1.0, head_name)
                self.prefix_definition(right_name, "d", 1.0, head_name)

    def generate_system_equation(self):
        def combine(left, right):
            return pepapot.ParsedSystemCooperation(left, [], right)
        processes = [pepapot.ParsedNamedComponent(x) for x in self.processes]
        self.system_equation = functools.reduce(combine, processes)

    def generate_model(self):
        self.generate_process_definitions()
        self.generate_system_equation()
        # TODO: We should generate some constant definitions and possibly
        # use them in the rates
        self.model = pepapot.ParsedModel([], self.process_definitions,
                                         self.system_equation)

    def get_model_source(self):
        return self.model.format()


class TestRandom(unittest.TestCase):
    @staticmethod
    def random_model_source():
        random_pepa = RandomPepa()
        random_pepa.generate_model()
        return random_pepa.get_model_source()

    def setUp(self):
        self.model_sources = [self.random_model_source() for x in range(10)]

    def test_model(self):
        for model_source in self.model_sources:
            logging.info("The random model source:")
            logging.info(model_source)
            model = pepapot.parse_model(model_source)
            model_solver = pepapot.ModelSolver(model)
            is_valid_gen_matrix(self, model_solver)


class TestParseError(unittest.TestCase):
    """ More documentation than test. This is intended to make sure that we
        know what will happen when the source is not syntactically correct.
    """
    def setUp(self):
        self.model_source = "P = (, 1.0).P1; P <a> P"

    def test_parse_error(self):
        from pyparsing import ParseException
        self.assertRaises(ParseException,
                          pepapot.parse_model, self.model_source)


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

## Now we test for Bio-PEPA models

## Obviously this is not so far a valid Bio-PEPA model.
simple_biopepa_model = """
delta = 1.0;

kineticLawOf d : delta * M;

M = (d, 1) << M;

M[1]
"""


class TestSimpleBioModel(unittest.TestCase):
    def setUp(self):
        self.model_source = simple_biopepa_model
        self.expected_number_species = 1
        self.expected_populations = {'M': 1}
        self.expected_result = {'M': 4.54009266e-05}
        self.configuration = pepapot.Configuration()

    def test_everything(self):
        model = pepapot.parse_biomodel(self.model_source)

        # Test the parser
        number_species = len(model.species_defs)
        self.assertEqual(number_species, self.expected_number_species)

        population_env = {p.species_name: p.amount for p in model.populations}
        self.assertEqual(population_env, self.expected_populations)

        # Test the solver
        model_solver = pepapot.BioModelSolver(model)
        result = model_solver.solve_odes(self.configuration)

        for species, population in self.expected_result.items():
            row = result.rows[-1]
            index = result.column_names.index(species)
            self.assertAlmostEqual(row[index], population)

reverse_reaction_biopepa_model = """
delta = 1.0;
gamma = 0.5;

kineticLawOf r : delta * A;
kineticLawOf rm : gamma * B;

A = (r, 1) << A + (rm, 1) >> A;
B = (r, 1) >> B + (rm, 1) << B;

A[100] <*> B[100]
"""


class TestReverseBioModel(TestSimpleBioModel):
    def setUp(self):
        self.model_source = reverse_reaction_biopepa_model
        self.expected_number_species = 2
        self.expected_populations = {'A': 100, 'B': 100}
        self.expected_result = {'A': 66.66667694, 'B': 133.33332306}
        self.configuration = pepapot.Configuration()

michaelis_menton_biopepa_model = """
delta = 1.0;
gamma = 0.1 * delta;

kineticLawOf r : fMA(delta);
kineticLawOf rm : fMA(gamma);
kineticLawOf s : fMA(delta);

E = (r, 1) << + (rm, 1) >> + (s, 1) >>;
S = (r, 1) << + (rm, 1) >>;
ES = (r, 1) >> + (rm, 1) << + (s, 1) <<;
P = (s, 1) >>;

E[100] <*> S[100] <*> ES[0] <*> P[0]
"""


class TestMMBioModel(TestSimpleBioModel):
    def setUp(self):
        self.model_source = michaelis_menton_biopepa_model
        self.expected_number_species = 4
        self.expected_populations = {'E': 100, 'S': 100, 'ES': 0.0, 'P': 0.0}
        self.expected_result = {'E': 99.995274249890841,
                                'S': 4.7736425237154927e-06,
                                'ES': 0.0047257501091977306,
                                'P': 99.995269476248424
                                }
        self.configuration = pepapot.Configuration()


class TestBioSyntaxSugar(unittest.TestCase):
    """ A simple test class which tests that two models are equivalent
        despite having different syntaxes. This particular one is testing
        the syntax sugar of behaviours such that one can simply leave off
        the process name attached to the behaviour, which is generally the
        same as the name of the process being defined.
    """
    def setUp(self):
        self.configuration = pepapot.Configuration()
        self.left_model_source = reverse_reaction_biopepa_model
        self.right_model_source = """
        delta = 1.0;
        gamma = 0.5;

        kineticLawOf r : delta * A;
        kineticLawOf rm : gamma * B;

        A = (r, 1) << + (rm, 1) >> ;
        B = (r, 1) >> + (rm, 1) << ;

        A[100] <*> B[100]
        """

    def get_result(self, model_source):
        model = pepapot.parse_biomodel(model_source)
        model_solver = pepapot.BioModelSolver(model)
        result = model_solver.solve_odes(self.configuration)
        return result

    def test_equivalence(self):
        left_result = self.get_result(self.left_model_source)
        right_result = self.get_result(self.right_model_source)

        left_final_row = list(left_result.rows[-1])
        right_final_row = list(right_result.rows[-1])
        self.assertListEqual(left_final_row, right_final_row)


class TestBioStoichSugar(TestBioSyntaxSugar):
    """ This Biological syntax sugar tests for the stoichiometry being 1
        where the stoich and parentheses of the behaviour can be omitted.
    """
    def setUp(self):
        self.configuration = pepapot.Configuration()
        self.left_model_source = reverse_reaction_biopepa_model
        self.right_model_source = """
        delta = 1.0;
        gamma = 0.5;

        kineticLawOf r : delta * A;
        kineticLawOf rm : gamma * B;

        A = r << + rm >> ;
        B = r >> + rm << ;

        A[100] <*> B[100]
        """


class TestBioFMASyntax(TestBioSyntaxSugar):
    """ This tests the fMA sugar for rate laws """
    def setUp(self):
        self.configuration = pepapot.Configuration()
        self.left_model_source = reverse_reaction_biopepa_model
        self.right_model_source = """
        gamma = 0.5;

        kineticLawOf r : fMA(1.0);
        kineticLawOf rm : fMA(gamma);

        A = (r, 1) << A + (rm, 1) >> A;
        B = (r, 1) >> B + (rm, 1) << B;

        A[100] <*> B[100]
        """


class TestBioActiviator(TestSimpleBioModel):
    def setUp(self):
        self.model_source = """ kineticLawOf a: fMA(1.0);

                                A = a >> ;
                                B = a (+) ;

                                A[0] <*> B[100]
                            """
        self.expected_number_species = 2
        self.expected_populations = {'A': 0, 'B': 100}
        self.expected_result = {'A': 999.99999999999977,
                                'B': 100,
                                }
        self.configuration = pepapot.Configuration()


class TestBioStoichiometryTwo(TestSimpleBioModel):
    def setUp(self):
        self.model_source = """ kineticLawOf a: fMA(1.0);
                                kineticLawOf b: fMA(1.0);

                                A = a >> + (b, 2) << ;
                                B = a (+) ;

                                A[0] <*> B[100]
                            """
        self.expected_number_species = 2
        self.expected_populations = {'A': 0, 'B': 100}
        self.expected_result = {'A': 7.0710678118656602,
                                'B': 100,
                                }
        self.configuration = pepapot.Configuration()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
