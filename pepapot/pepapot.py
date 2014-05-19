"""pepapot.

Usage:
  pepapot.py steady util <name>...
  pepapot.py -h | --help
  pepapot.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  """
import logging
from collections import namedtuple
import functools
import operator

from docopt import docopt
import pyparsing
from pyparsing import Combine, Or, Optional, Literal, Suppress
import numpy
from scipy.integrate import odeint
from lazy import lazy


def list_product(factors):
    """ Simple utility the same as 'sum' but for the product of the arguments.
        Note: returns 1 for the empty list, which seems reasonable, given that
        sum([]) = 0.
    """
    result = 1
    for factor in factors:
        result *= factor
    return result


def evaluate_function_app(name, arg_values):
    if name == "plus" or name == "+":
        return sum(arg_values)
    elif name == "times" or name == "*":
        return list_product(arg_values)
    elif name == "minus" or name == "-":
        # What should we do if there is only one argument, I think we
        # should treat '(-) x' the same as '0 - x'.
        answer = arg_values[0]
        for arg in arg_values[1:]:
            answer -= arg
        return answer
    elif name == "divide" or name == "/":
        answer = arg_values[0]
        for arg in arg_values[1:]:
            answer /= arg
        return answer
    elif name == "power" or name == "**":
        # power is interesting because it associates to the right
        exponent = 1
        # counts downwards from the last index to the 0.
        # As an example, consider power(3,2,3), the answer should be
        # 3 ** (2 ** 3) = 3 ** 8 = 6561, not (3 ** 2) ** 3 = 9 ** 3 = 81
        # going through our loop here we have
        # exp = 1
        # exp = 3 ** exp = 3
        # exp = 2 ** exp = 2 ** 3 = 8
        # exp = 3 ** exp = 3 ** 8 = 6561
        for i in range(len(arg_values) - 1, -1, -1):
            exponent = arg_values[i] ** exponent
        return exponent
    else:
        raise ValueError("Unknown function name: " + name)


class Expression:
    """ A new simpler representation of expressions in which we only have
        one kind of expression. The idea is that reduce and get_value can be
        coded as in terms of a single recursion.
    """
    def __init__(self):
        self.name = None
        self.number = None
        self.arguments = []

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and 
                self.name == other.name and
                self.number == other.number and
                self.arguments == other.arguments)

    @classmethod
    def num_expression(cls, number):
        expression = cls()
        expression.number = number
        return expression

    @classmethod
    def name_expression (cls, name):
        expression = cls()
        expression.name = name
        return expression

    @classmethod
    def apply_expression(cls, name, arguments):
        expression = cls()
        expression.name = name
        expression.arguments = arguments
        return expression

    @classmethod
    def addition(cls, left, right):
        return cls.apply_expression("+", [left, right])

    @classmethod
    def subtract(cls, left, right):
        return cls.apply_expression("-", [left, right])

    @classmethod
    def multiply(cls, left, right):
        return cls.apply_expression("*", [left, right])

    @classmethod
    def divide(cls, left, right):
        return cls.apply_expression("/", [left, right])

    @classmethod
    def power(cls, left, right):
        return cls.apply_expression("**", [left, right])


    def used_names(self):
        names = set()
        if self.name:
            names.add(self.name)
        for arg in self.arguments:
            names.update(arg.used_names())

        return names

    def get_value(self, environment=None):
        """ Returns the value of an expression in the given environment if
            any. Raises an assertion error if the expression cannot be reduced
            to a value.
        """
        reduced_expression = self.reduce_expr(environment=environment)
        assert (reduced_expression.number is not None)
        return reduced_expression.number

    def reduce_expr(self, environment=None):
        if self.number is not None:
            return self
        if not self.arguments:
            # We have a name expression so if the environment is None or
            # or the name is not in the environment then we cannot reduce
            # any further so just return the current expression.
            if not environment or self.name not in environment:
                return self
            expression = environment[self.name]
            return expression.reduce_expr(environment=environment)

        # If we get here then we have an application expression, so we must
        # first reduce all the arguments and then we may or may not be able
        # to reduce the entire expression to a number or not.
        arguments = [a.reduce_expr(environment)
                     for a in self.arguments]
        arg_values = [a.number for a in arguments]

        if any(v is None for v in arg_values):
            return Expression.apply_expression(self.name, arguments)
        else:
            result_number = evaluate_function_app(self.name, arg_values)
            return Expression.num_expression(result_number)


class Visitor(object):
    def __init__(self):
        self.result = None

    def visit_get_results(self, entity):
        """ Performs the visit and also returns the result, sort of useful
            for doing this within a list comprehension.
        """
        entity.visit(self)
        return self.result

    @classmethod
    def get_result(cls, entity, *args):
        visitor = cls(*args)
        entity.visit(visitor)
        return visitor.result


Action = namedtuple('Action', ["action", "rate", "successor"])


def make_identifier_grammar(start_characters):
    identifier_start = pyparsing.Word(start_characters, exact=1)
    identifier_remainder = Optional(pyparsing.Word(pyparsing.alphanums + "_"))
    identifier_grammar = identifier_start + identifier_remainder
    return pyparsing.Combine(identifier_grammar)

lower_identifier = make_identifier_grammar("abcdefghijklmnopqrstuvwxyz")
upper_identifier = make_identifier_grammar("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
identifier = make_identifier_grammar(pyparsing.alphas)

plusorminus = Literal('+') | Literal('-')
number = pyparsing.Word(pyparsing.nums)
integer = Combine(Optional(plusorminus) + number)
decimal_fraction = Literal('.') + number
scientific_enotation = pyparsing.CaselessLiteral('E') + integer
floatnumber = Combine(integer + Optional(decimal_fraction) +
                      Optional(scientific_enotation))


# A helper to create grammar element which must be surrounded by parentheses
# but you then wish to ignore the parentheses
def parenthetical_grammar(element_grammar):
    return Suppress("(") + element_grammar + Suppress(")")


def create_expression_grammar(identifier_grammar):
    expr_grammar = pyparsing.Forward()

    def num_expr_parse_action(tokens):
        return Expression.num_expression(float(tokens[0]))

    num_expr = floatnumber.copy()
    num_expr.setParseAction(num_expr_parse_action)

    def apply_expr_parse_action(tokens):
        if len(tokens) == 1:
            return Expression.name_expression(tokens[0])
        else:
            return Expression.apply_expression(tokens[0], tokens[1:])
    arg_expr_list = pyparsing.delimitedList(expr_grammar)
    opt_arg_list = Optional(parenthetical_grammar(arg_expr_list))
    apply_expr = identifier_grammar + opt_arg_list
    apply_expr.setParseAction(apply_expr_parse_action)

    atom_expr = Or([num_expr, apply_expr])

    multop = pyparsing.oneOf('* /')
    plusop = pyparsing.oneOf('+ -')

    def binop_parse_action(tokens):
        elements = tokens[0]
        operators = elements[1::2]
        exprs = elements[::2]
        assert len(exprs) - len(operators) == 1
        exprs_iter = iter(exprs)
        result_expr = next(exprs_iter)
        # Note: iterating in this order would not be correct if the binary
        # operator associates to the right, as with **, since for
        # [2, ** , 3, ** 2] we would get build up the apply expression
        # corresponding to (2 ** 3) ** 2, which is not what we want. However,
        # pyparsing seems to do the correct thing and give this function
        # two separate calls one for [3, **, 2] and then again for
        # [2, ** , Apply(**, [3,2])].
        for oper, expression in zip(operators, exprs_iter):
            args = [result_expr, expression]
            result_expr = Expression.apply_expression(oper, args)
        return result_expr

    precedences = [("**", 2, pyparsing.opAssoc.RIGHT, binop_parse_action),
                   (multop, 2, pyparsing.opAssoc.LEFT, binop_parse_action),
                   (plusop, 2, pyparsing.opAssoc.LEFT, binop_parse_action),
                   ]
    expr_grammar << pyparsing.operatorPrecedence(atom_expr, precedences)
    return expr_grammar

lower_expr_grammar = create_expression_grammar(lower_identifier)
expr_grammar = create_expression_grammar(identifier)


class ConstantDefinition(object):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    grammar = identifier + "=" + expr_grammar + ";"
    list_grammar = pyparsing.Group(pyparsing.ZeroOrMore(grammar))

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tokens[0], tokens[2])

ConstantDefinition.grammar.setParseAction(ConstantDefinition.from_tokens)


# One problem that we still have is that in PEPA, it is difficult to
# distinguish between an expression and a process, for example:
# P = Q + R;
# Could be either, unless we insist that rates are lower case and processes
# are upper case? This is what we do, so here we have a separate class for
# PEPA specific constant definitions which is desparingly similar to the
# generic constant definition above, but only allows lower-case identifiers
# in the associated expressions.
#
# However if we wanted to allow functional rates then `Q + R`
# becomes a reasonable rate expression. One solution is to allow that in a
# prefix but not in a constant definition. Another is to have a separate
# syntax for rate constants which is a little bit of a shame that we would
# then not be backwards compatible with existing PEPA software.
class PEPAConstantDef(object):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    grammar = identifier + "=" + lower_expr_grammar + ";"
    list_grammar = pyparsing.Group(pyparsing.ZeroOrMore(grammar))

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tokens[0], tokens[2])

PEPAConstantDef.grammar.setParseAction(PEPAConstantDef.from_tokens)

# Generally what we wish to do is *reduce* a set of definitions because some
# of these will have functional rates. Currently we are not supporting that
# for PEPA but we are for Bio-PEPA. This reduces a list of definitions which
# for PEPA will mean (for a valid set of definitions) that we reduce all the
# right hand sides to simple number expressions.
def definition_environment(definitions, environment=None,
                           rhs_fun=None, inplace=True):
    if environment is None:
        environment = dict()

    for definition in definitions:
        rhs = definition.rhs
        if rhs_fun:
            rhs = rhs_fun(definition.rhs, environment)
            if inplace:
                definition.rhs = rhs
        environment[definition.lhs] = rhs

    return environment


def reduce_definitions(definitions, environment=None, inplace=True):
    """ Reduces the definitions and also puts those definitions into an
        environment. If 'inplace' is True, it updates the right hand side of
        each definition in place, otherwise it leaves the definitions as they
        were and just returns the environment.
    """
    def rhs_fun(rhs, env):
        return rhs.reduce_expr(environment=env)
    return definition_environment(definitions, environment=environment,
                                  rhs_fun=rhs_fun, inplace=inplace)


# Unlike the above function, this assumes all the constants will reduce to a
# simple number expression and then extracts that number.
def constant_def_environment(definitions, environment=None):
    """ Turns a list of constant definitions into an environment which maps
        the names defined in the definitions to values. This does not have
        much use outside of testing
    """
    reduced_env = reduce_definitions(definitions, environment, inplace=False)
    value_environment = dict()
    for name, expression in reduced_env.items():
        value_environment[name] = expression.get_value()
    return value_environment


class ProcessIdentifier(object):
    def __init__(self, name):
        self.name = name

    grammar = identifier.copy()

    def visit(self, visitor):
        visitor.visit_ProcessIdentifier(self)

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tokens[0])

    def __str__(self):
        return self.name

    def format(self):
        return self.name


ProcessIdentifier.grammar.setParseAction(ProcessIdentifier.from_tokens)
process_leaf = pyparsing.Forward()


class TopRate(Expression):
    # TopRate is only equal to another TopRate
    # Note that we would almost get the same behaviour has here if we simply
    # said TopRate was equal to float("inf"). However unfortunately:
    # float("inf") / float("inf") = nan, not, as we would wish, 1.0. Similarly
    # 0.0 * float("inf") = nan, and not, 0.0 as we would wish.
    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __lt__(self, other):
        return False

    def __gt__(self, other):
        return True

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self

    def __mul__(self, other):
        if other == 0.0:
            return 0.0
        return self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if other == self:
            return 1
        else:
            return self

    def __rtruediv__(self, other):
        # You might think we should check 'other' for equality with self, as
        # we do in '__truediv__' above. However if other == TopRate, then that
        # is when '__truediv__' would have been called on 'other' anyway.
        return 0

    def visit(self, visitor):
        """Implements the visit method allowing ExpressionVisitors to work"""
        visitor.visit_TopRate(self)


class PrefixNode(object):
    def __init__(self, action, rate, successor):
        self.action = action
        self.rate = rate
        self.successor = successor

    top_rate_grammar = Literal("T")
    top_rate_grammar.setParseAction(lambda _t: TopRate())

    rate_grammar = Or([top_rate_grammar, expr_grammar])

    # This grammar then actually allows for functional rates because it is
    # allowing any identifier via the use of 'expr_grammar'.
    grammar = "(" + identifier + "," + rate_grammar + ")" + "." + process_leaf

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tokens[1], tokens[3], tokens[6])

    def visit(self, visitor):
        visitor.visit_PrefixNode(self)

    def format(self):
        return "".join(["(", self.action, ", ", str(self.rate),
                        ").", self.successor.format()])

PrefixNode.grammar.setParseAction(PrefixNode.from_tokens)
process_leaf << Or([PrefixNode.grammar, ProcessIdentifier.grammar])


class ChoiceNode(object):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def visit(self, visitor):
        visitor.visit_ChoiceNode(self)

    def format(self):
        # I believe there is no need for  parentheses, since we cannot make
        # a mistake since the only binary operator is +. Currently we cannot
        # have P = (a, r).(Q + R); which would make things ambiguous since
        # P = (a, r).Q + R could be either P = ((a.r).P) + R; or
        # P = (a, r).(Q + R);
        return " ".join([self.lhs.format(), "+", self.rhs.format()])


class ProcessVisitor(Visitor):
    def visit_ProcessIdentifier(self, _process):
        pass

    def visit_PrefixNode(self, process):
        process.successor.visit(self)

    def visit_ChoiceNode(self, process):
        process.lhs.visit(self)
        process.rhs.visit(self)


class ProcessPossibleActionsVisitor(ProcessVisitor):
    def __init__(self):
        super(ProcessPossibleActionsVisitor, self).__init__()
        self.result = []

    def visit_PrefixNode(self, process):
        action = Action(process.action, process.rate, str(process.successor))
        self.result.append(action)


class UsedProcessNamesVisitor(ProcessVisitor):
    def __init__(self):
        super(UsedProcessNamesVisitor, self).__init__()
        self.result = set()

    def visit_ProcessIdentifier(self, process):
        self.result.add(process.name)


class UsedRateNamesProcessVisitor(ProcessVisitor):
    def __init__(self):
        super(UsedRateNamesProcessVisitor, self).__init__()
        self.result = set()

    def visit_PrefixNode(self, prefix):
        used_rate_names = prefix.rate.used_names()
        self.result.update(used_rate_names)
        prefix.successor.visit(self)


class ProcessImmediateAliasesVisitor(ProcessVisitor):
    def __init__(self):
        super(ProcessImmediateAliasesVisitor, self).__init__()
        self.result = []

    def visit_ProcessIdentifier(self, process):
        self.result.append(process.name)

    def visit_PrefixNode(self, process):
        # Note here that we do not recursively visit the successor, because
        # we are looking for immediate aliases here. We do not *add* to the
        # aliases here, but note that we do not set result to []. Because it
        # might be that we have, for example: P = P1 + (a, r).P2; Whilst
        # visiting the prefix we would not wish to overwrite the result of
        # visiting the left hand node.
        pass


class ProcessConcretiseActionsVisitor(ProcessVisitor):
    def __init__(self, environment=None):
        super(ProcessConcretiseActionsVisitor, self).__init__()
        self.environment = environment

    def visit_PrefixNode(self, process):
        if process.rate != TopRate():
            env = self.environment
            process.rate = process.rate.get_value(environment=env)


class ProcessDefinition(object):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def process_grammar_action(tokens):
        return functools.reduce(lambda l, r: ChoiceNode(l, r), tokens)
    process_grammar = pyparsing.delimitedList(process_leaf, delim="+")
    process_grammar.setParseAction(process_grammar_action)

    grammar = identifier + "=" + process_grammar + ";"
    list_grammar = pyparsing.Group(pyparsing.OneOrMore(grammar))

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tokens[0], tokens[2])

    def format(self):
        return " ".join([self.lhs, "=", self.rhs.format(), ";"])

ProcessDefinition.grammar.setParseAction(ProcessDefinition.from_tokens)


class ParsedNamedComponent(object):
    def __init__(self, name):
        self.identifier = name

    grammar = identifier.copy()

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tokens[0])

    def visit(self, visitor):
        visitor.visit_NamedComponent(self)

    def format(self):
        return self.identifier

ParsedNamedComponent.grammar.setParseAction(ParsedNamedComponent.from_tokens)


class ParsedAggregation(object):
    def __init__(self, lhs, amount):
        self.lhs = lhs
        self.amount = amount

    # Forces this to be a non-negative integer, though could be zero. Arguably
    # we may want to allow decimals here, obviously only appropriate for
    # translation to ODEs.
    array_suffix = "[" + number + "]"
    array_suffix.setParseAction(lambda x: int(x[1]))
    # This way means that aggregation can only be applied to a single
    # identifier such as "P[10]". We could also allow for example
    # "(P <a> Q)[10]".
    grammar = ParsedNamedComponent.grammar + Optional(array_suffix)

    @classmethod
    def from_tokens(cls, tokens):
        """ Another atypical 'from_tokens' implementation that might not
            actually return a ParsedAggregation, again this comes from the
            inclusion of 'Optional' in its grammar
        """
        if len(tokens) > 1:
            return cls(tokens[0], tokens[1])
        else:
            return tokens

    def visit(self, visitor):
        visitor.visit_Aggregation(self)


ParsedAggregation.grammar.setParseAction(ParsedAggregation.from_tokens)
system_equation_grammar = pyparsing.Forward()
system_equation_paren = "(" + system_equation_grammar + ")"
system_equation_paren.setParseAction(lambda x: x[1])
system_equation_atom = Or([ParsedAggregation.grammar,
                           system_equation_paren])


class ParsedSystemCooperation(object):
    def __init__(self, lhs, coop_set, rhs):
        self.lhs = lhs
        self.cooperation_set = coop_set
        self.rhs = rhs

    action_names_grammar = pyparsing.delimitedList(identifier, ",")
    activity_list = Suppress("<") + action_names_grammar + Suppress(">")
    coop_set_grammar = Or([Suppress(Literal("||")),
                           Suppress(Literal("<>")),
                           activity_list])
    # It's a double list because otherwise the system_equation_parser will
    # assume the list returned is a set of tokens and concatenate it in
    # with the other tokens. In this way we get a parse result from 'grammar'
    # for ("P <a,b,c> Q") to be [ Named, [a,b,c], Named ] which is what we
    # want, rather than [ Named, a, b, c, Named ].
    # Now, why 'lambda t: [t]' doesn't work, I do not understand. I would have
    # thought that '[t] == [[x for x in t]]', but try changing it and you will
    # see that the tests fail. I do not understand why.
    coop_set_grammar.setParseAction(lambda t: [[x for x in t]])

    grammar = (system_equation_atom + Optional(coop_set_grammar +
                                               system_equation_grammar))

    @classmethod
    def from_tokens(cls, tokens):
        """ Another non-typical definition of 'from_tokens' in which the
            result may not actually be a ParsedSystemCooperation. Once again
            this stems from the user of Optional in the grammar.
        """
        if len(tokens) > 1:
            return cls(tokens[0], tokens[1], tokens[2])
        else:
            return tokens

    def visit(self, visitor):
        visitor.visit_SystemCooperation(self)

    def format(self):
        coop_string = "<" + ", ".join(self.cooperation_set) + ">"
        return " ".join([self.lhs.format(), coop_string, self.rhs.format()])


system_equation_grammar << ParsedSystemCooperation.grammar
system_equation_grammar.setParseAction(ParsedSystemCooperation.from_tokens)


class ComponentVisitor(Visitor):
    def visit_NamedComponent(self, _component):
        pass

    def visit_Aggregation(self, component):
        component.lhs.visit(self)

    def visit_SystemCooperation(self, component):
        component.lhs.visit(self)
        component.rhs.visit(self)


class CompUsedProcessNames(ComponentVisitor):
    def __init__(self):
        super(CompUsedProcessNames, self).__init__()
        self.result = set()

    def visit_NamedComponent(self, component):
        self.result.add(component.identifier)


class CompSharedActions(ComponentVisitor):
    """ This visitor is mostly for testing purposes. It allows us to test that
        the model we parsed is indeed the one we were thinking of. It is
        also conceivable that some other operation will wish to know all of
        the actions which are involved in at least one cooperation.
    """
    def __init__(self):
        super(CompSharedActions, self).__init__()
        self.result = set()

    def visit_SystemCooperation(self, component):
        component.lhs.visit(self)
        component.rhs.visit(self)
        self.result.update(component.cooperation_set)


class PepaWarning(object):
    pass


class PepaUnusedRateNameWarning(PepaWarning):
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name
        return NotImplemented


class PepaError(object):
    pass


class StaticAnalysis(object):
    def __init__(self):
        self.warnings = []
        self.errors = []


class ParsedModel(object):
    def __init__(self, constant_defs, proc_defs, sys_equation):
        self.constant_defs = constant_defs
        self.process_definitions = proc_defs
        self.system_equation = sys_equation

    # Note, this parser does not insist on the end of the input text.
    # Which means in theory you could have something *after* the model text,
    # which might indeed be what you are wishing for.
    grammar = (PEPAConstantDef.list_grammar +
               ProcessDefinition.list_grammar +
               system_equation_grammar)
    whole_input_grammar = grammar + pyparsing.StringEnd()

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tokens[0], tokens[1], tokens[2])

    def get_process_definition(self, name):
        """ Returns the process definition which defines the given name.
            This may raise StopIteration if no such definition exists
        """
        return next(x for x in self.process_definitions if x.lhs == name)

    def get_immediate_aliases(self):
        """ Builds a dictionary which maps a process name to the process
            names which it can become without doing any activities. So:
            A = P;
            A can become a P without any activities, similarly:
            A = P + Q;
            A can 'become' a P or a Q without doing any activities. The point
            here being that if we wish to know what activities A can perform
            then need to know what activities P and Q can perform.
        """
        # TODO: Have not yet implemented closure here, so if we had:
        # A = P; B = A; then P should be an immediate alias of B but we will
        # not determine that fact here, but we should.
        aliases = dict()
        for definition in self.process_definitions:
            name = definition.lhs
            process = definition.rhs
            aliases[name] = ProcessImmediateAliasesVisitor.get_result(process)
        # TODO: We should be able to do the closure here, without going back
        # to the definitions but simply from the dictionary itself.
        return aliases

    def get_components(self):
        """Returns a dictionary mapping each name used in the system equation
           to a list of names reachable via actions from that name.
        """
        # Note that we could do a bit of memoisation here, since
        # 'get_components' is used in both 'used_process_names' and
        # 'get_initial_state', but we do not expect this to take a long time.
        used_processes = CompUsedProcessNames.get_result(self.system_equation)
        components = dict()
        for name in used_processes:
            # Closure is a list and not a set, because we wish for the order
            # to be deterministic. This is because it is used in building
            # the initial state and hence would otherwise be awkward to test.
            closure = []
            name_queue = set([name])
            while name_queue:
                name = name_queue.pop()
                if name not in closure:
                    closure.append(name)
                    definition = self.get_process_definition(name)
                    process = definition.rhs
                    new_names = UsedProcessNamesVisitor.get_result(process)
                    name_queue.update(new_names)
            components[name] = closure
        return components

    def used_process_names(self):
        # I think that it is just possible for a name to be in two separate
        # components. If it does not link back to other names, for example
        # P = (a, r).Stop; Here it is possible that P is in two separate
        # components. However, it's questionable as to whether that is wrong
        # or not.
        components = self.get_components()
        used_names = set()
        for names in components.values():
            used_names.update(set(names))
        return used_names

    def get_process_actions(self):
        actions_dictionary = dict()
        for definition in self.process_definitions:
            actions = ProcessPossibleActionsVisitor.get_result(definition.rhs)
            actions_dictionary[definition.lhs] = actions

        # A slight problem, if we have A = B; and B = P; and P = (a,r).P1;
        # we do *not* wish for A to have *two* copies of the action
        # (a,r).P1. But the aliases for A will be B and P, so we cannot simply
        # say "get all the immediate aliases of A and add their actions as
        # actions that A can do, because B will be able to do (a,r).P1 as well
        # as P, so we will conclude that A can perform (a,r).P1 twice. Instead
        # we build up a dictionary of aliased actions and only add those.
        aliased_actions = dict()
        for name, aliases in self.get_immediate_aliases().items():
            these_aliased_actions = []
            for alias in aliases:
                these_aliased_actions += actions_dictionary[alias]
            aliased_actions[name] = these_aliased_actions

        # Unfortunately we cannot simply say:
        # actions_dictionary.update(aliased_actions)
        # Because of the choice operator a process may have both:
        # P = (a,r).P1 + B;
        for name, entry in aliased_actions.items():
            actions_list = actions_dictionary.get(name, []) + entry
            actions_dictionary[name] = actions_list
        return actions_dictionary

    def defined_process_names(self):
        """Return the list of defined process names"""
        names = [definition.lhs for definition in self.process_definitions]
        return set(names)

    def get_defined_rate_names(self):
        """Return the list of defined rate names"""
        return {definition.lhs for definition in self.constant_defs}

    def get_used_rate_names(self):
        used_names = set()
        for definition in self.constant_defs:
            names = ExpressionUsedNamesVisitor.get_result(definition.rhs)
            used_names.update(names)

        for definition in self.process_definitions:
            names = UsedRateNamesProcessVisitor.get_result(definition.rhs)
            used_names.update(names)

        return used_names

    def perform_static_analysis(self):
        results = StaticAnalysis()

        defined_rate_names = self.get_defined_rate_names()
        used_rate_names = self.get_used_rate_names()
        for rate_name in defined_rate_names:
            if rate_name not in used_rate_names:
                warning = PepaUnusedRateNameWarning(rate_name)
                results.warnings.append(warning)

        return results

    def format(self):
        proc_def_strings = [p.format() for p in self.process_definitions]
        proc_defs = "\n".join(proc_def_strings)
        sys_equation = self.system_equation.format()
        return "\n".join([proc_defs, sys_equation])


ParsedModel.grammar.setParseAction(ParsedModel.from_tokens)


def parse_model(model_string):
    """Parses a model ensuring that we have consumed the entire input"""
    return ParsedModel.whole_input_grammar.parseString(model_string)[0]


Transition = namedtuple('Transition', ["action", "rate", "successor"])
StateInfo = namedtuple('StateInfo', ["state_number", "transitions"])


class InitialStateVisitor(ComponentVisitor):
    def __init__(self, components):
        super(InitialStateVisitor, self).__init__()
        self.components = components
        self.result = None

    def visit_NamedComponent(self, component):
        self.result = component.identifier

    def visit_Aggregation(self, component):
        component.lhs.visit(self)
        # This assumes that lhs will be an identifier, which as I write this
        # is enforced by the parser, but ultimately it would be good to allow
        # (P <*> Q)[X]
        initial_name = self.result
        state_names = self.components[initial_name]
        pairs = [(x, component.amount if x == initial_name else 0)
                 for x in state_names]
        # An aggregation state is a tuple consisting of pairs. Each pair is
        # the name of a local state and the number of components in that state
        self.result = tuple(pairs)

    def visit_SystemCooperation(self, component):
        component.lhs.visit(self)
        lhs = self.result
        component.rhs.visit(self)
        rhs = self.result
        # A cooperation state is simply a pair consisting of the left and
        # right sub-states
        self.result = (lhs, rhs)


class MemoisationBuilder(object):
    """ A somewhat abstract builder class which memorises the states which
        it has already seen and explored, thus remembering the transitions it
        has already computed for sub-states of the global state space.
    """
    def __init__(self):
        self.state_dictionary = dict()

    def _compute_transitions(self, state):  # pragma: no cover
        # Just using a pragma at the moment to exclude this from coverage
        # Could otherwise use a '.coveragerc' file as described at:
        # http://nedbatchelder.com/code/coverage/config.html#config
        raise Exception("Unimplemented abstract method: _compute_transitions")

    def get_transitions(self, state):
        state_information = self.state_dictionary.get(state, None)
        if state_information:
            return state_information.transitions
        transitions = self._compute_transitions(state)
        state_information = StateInfo(len(self.state_dictionary), transitions)
        self.state_dictionary[state] = state_information
        return transitions


class LeafBuilder(MemoisationBuilder):
    def __init__(self, actions_dictionary):
        super(LeafBuilder, self).__init__()
        self.actions_dictionary = actions_dictionary

    def _compute_transitions(self, state):
        # Leaf states are simply names
        actions = self.actions_dictionary[state]
        transitions = [Transition(a.action, a.rate, a.successor)
                       for a in actions]
        return transitions


class AggregationBuilder(MemoisationBuilder):
    def __init__(self, lhs):
        super(AggregationBuilder, self).__init__()
        self.lhs = lhs

    def _compute_transitions(self, state):
        new_transitions = []
        # An aggregation, or array state is a tuple mapping
        # lhs states to numbers
        for index, (local_state, num) in enumerate(state):
            if num > 0:
                local_transitions = self.lhs.get_transitions(local_state)
                for transition in local_transitions:
                    # The successor state equals the current state but with
                    # one fewer of the local state and one more of the
                    # transition's target. However, if both the current local
                    # state and the target of the transition are the same then
                    # we do not need to update the state.
                    if local_state == transition.successor:
                        successor = state
                    else:
                        def new_number(s, n):
                            if s == local_state:
                                return n - 1
                            elif s == transition.successor:
                                return n + 1
                            else:
                                return n
                        successor = tuple([(s, new_number(s, n))
                                           for (s, n) in state])
                    # I'm not 100% this always correct. Should we rather add
                    # a number of new transitions (ie. num) where each
                    # transition has the original rate?
                    new_transition = Transition(transition.action,
                                                num * transition.rate,
                                                successor)
                    new_transitions.append(new_transition)
        return new_transitions


class CoopBuilder(MemoisationBuilder):
    def __init__(self, lhs, coop_set, rhs):
        super(CoopBuilder, self).__init__()
        self.lhs = lhs
        self.coop_set = coop_set
        self.rhs = rhs

    def _compute_transitions(self, state):
        # A cooperation state is simply a pair of the left and right states
        left_state, right_state = state
        left_transitions = self.lhs.get_transitions(left_state)
        right_transitions = self.rhs.get_transitions(right_state)
        transitions = []
        for transition in left_transitions:
            if transition.action not in self.coop_set:
                new_state = (transition.successor, right_state)
                new_transition = transition._replace(successor=new_state)
                transitions.append(new_transition)
        for transition in right_transitions:
            if transition.action not in self.coop_set:
                new_state = (left_state, transition.successor)
                new_transition = transition._replace(successor=new_state)
                transitions.append(new_transition)
        for action in self.coop_set:
            # This then implements the so-called "apparent rate" for this
            # shared activity. This is calculated as:
            # (r1/ra(lhs)) * (r2/ra(rhs)) * min(ra(lhs), ra(rhs))
            # Where r1 is the rate of the left transition, and r2 is the
            # rate of the right transition, ra(lhs) is the total rate at which
            # the left-hand process can perform the given action and similarly
            # for ra(rhs).
            left_shared = [t for t in left_transitions if t.action == action]
            right_shared = [t for t in right_transitions if t.action == action]
            left_rate = sum([t.rate for t in left_shared])
            right_rate = sum([t.rate for t in right_shared])
            governing_rate = min(left_rate, right_rate)
            transition_pairs = [(l, r)
                                for l in left_shared
                                for r in right_shared]
            for (left, right) in transition_pairs:
                rate = ((left.rate / left_rate) *
                        (right.rate / right_rate) *
                        governing_rate)
                new_state = (left.successor, right.successor)
                new_transition = Transition(action, rate, new_state)
                transitions.append(new_transition)

        return transitions


# So the StateBuilderHelper builds up an object of Leaf, Aggregation and
# CoopBuilders which has the same structure as the PEPA model. Hence the
# structure of the PEPA model is captured in this object and remains static
# whilst the state space is built. Each kind of builder knows how to
# build the statespace for its portion of the tree of the PEPA model structure
# whether that requires recursively building the state-space of the subtrees
# or not. To build up that
class StateBuilderHelper(ComponentVisitor):
    def __init__(self, actions_dictionary):
        super(StateBuilderHelper, self).__init__()
        self.actions_dictionary = actions_dictionary
        self.result = None

    def visit_NamedComponent(self, _component):
        self.result = LeafBuilder(self.actions_dictionary)

    def visit_Aggregation(self, component):
        component.lhs.visit(self)
        self.result = AggregationBuilder(self.result)

    def visit_SystemCooperation(self, component):
        component.lhs.visit(self)
        lhs = self.result
        component.rhs.visit(self)
        rhs = self.result
        self.result = CoopBuilder(lhs, component.cooperation_set, rhs)


class LeafUtilisations(object):
    def __init__(self):
        self.utilisations = dict()

    def utilise_state(self, state, probability):
        # We assume that state is a string representing the local state of
        # the process.
        previous_utilisation = self.utilisations.get(state, 0.0)
        self.utilisations[state] = previous_utilisation + probability

    def get_utilisations(self):
        return [self.utilisations]


class AggregationUtilisations(object):
    def __init__(self):
        self.utilisations = dict()

    def utilise_state(self, state, probability):
        # We assume state is a tuple mapping names to numbers
        for local_state, num in state:
            additional_util = probability * num
            previous_util = self.utilisations.get(local_state, 0.0)
            self.utilisations[local_state] = previous_util + additional_util

    def get_utilisations(self):
        return [self.utilisations]


class CoopUtilisations(object):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def utilise_state(self, state, probability):
        left, right = state
        self.lhs.utilise_state(left, probability)
        self.rhs.utilise_state(right, probability)

    def get_utilisations(self):
        return self.lhs.get_utilisations() + self.rhs.get_utilisations()


class UtilisationsBuilderHelper(ComponentVisitor):
    def __init__(self):
        super(UtilisationsBuilderHelper, self).__init__()
        self.result = None

    def visit_NamedComponent(self, _component):
        self.result = LeafUtilisations()

    def visit_Aggregation(self, component):
        # Currently we do not need to visit the lhs of an aggregation,
        # however, that is because we only allow the lhs to be a name,
        # something that is enforced by the parser. If we were to allow
        # any component, eg (P <*> Q)[10] then we would have to visit the lhs
        self.result = AggregationUtilisations()

    def visit_SystemCooperation(self, component):
        component.lhs.visit(self)
        lhs = self.result
        component.rhs.visit(self)
        rhs = self.result
        self.result = CoopUtilisations(lhs, rhs)


# Note, we must concretise the actions now such that all later operations
# need not worry about the fact that rates and populations might have
# been expressions. We do this in the initialiser for the model solver.
# It could be that really we might want, for example, functional rates.
# I think the absolute best way to do this would be to have the rate
# expression have a value field which is lazily evaluated, but this is awkward
# because of the need for an environment in which to evaluate the expression.
# For now I have done the simplest thing that would work. There are some
# fragilities, for example if you examine the model *after* you have created
# the model solver you will be reading concrete rate values when you might
# have been expecting expressions. This is even if you have not done anything
# with the model solver yet.
#
# Note, we could think about doing the concretising in the initial_state
# method, but it is lazily evaluated so you have to be careful you do not, for
# example call `model.get_process_actions()` before you inspect the
# `initial_state`, eg. if the concretising were done in the
# `initial_state` method, we could *not* do:
#        system_equation = self.model.system_equation
#        process_actions = self.model.get_process_actions()
#        state_builder = StateBuilderHelper.get_result(system_equation,
#                                                      process_actions)
#       explore_queue = set([self.initial_state])


class ModelSolver(object):
    """A full state space exploring model solver. This solver builds the
       entire state-space of the model and from that derives a CTMC which is
       then solved.
    """
    def __init__(self, model):
        self.model = model

        # In theory we could allow functional rates, and having done so we
        # could allow those to be defined in a 'constant' definition such as
        # r = P * 2.0; where P is a process name and hence refers to a
        # current population. But since that is not yet being asked for, we
        # will simply reduce all the constant definitions to values and then
        # apply those throughout the process definitions.
        environment = constant_def_environment(self.model.constant_defs)

        concretiser = ProcessConcretiseActionsVisitor(environment)
        for proc_def in self.model.process_definitions:
            proc_def.rhs.visit(concretiser)

    @lazy
    def initial_state(self):
        components = self.model.get_components()
        system_equation = self.model.system_equation
        self._initial_state = InitialStateVisitor.get_result(system_equation,
                                                             components)
        return self._initial_state

    @lazy
    def state_space(self):
        system_equation = self.model.system_equation
        process_actions = self.model.get_process_actions()
        state_builder = StateBuilderHelper.get_result(system_equation,
                                                      process_actions)
        explore_queue = set([self.initial_state])
        explored = set()
        while (explore_queue):
            current_state = explore_queue.pop()
            transitions = state_builder.get_transitions(current_state)
            successor_states = [t.successor for t in transitions]
            explored.add(current_state)
            for new_state in successor_states:
                # Note that we should be careful if the new_state is the same
                # as the current state. We won't put it in the explore_queue
                # since the current state should be in explored. However it
                # will mean we have a self-loop, and we should probably flag
                # that at some point.
                if new_state not in explored and new_state != current_state:
                    explore_queue.add(new_state)
        state_space = state_builder.state_dictionary
        return state_space

    def log_state_space(self, state_space=None):
        if state_space is None:
            state_space = self.state_space
        logging.info("State space:")
        for (state, state_info) in state_space.items():
            logging.info("State: " + str(state))
            for transition in state_info.transitions:
                logging.info("    (" + transition.action +
                             ", " + str(transition.rate) +
                             ")." + str(transition.successor))

    @lazy
    def gen_matrix(self):
        """ Build the generator matrix based on the state space of the model
        """
        # State space is a dictionary which maps a state representation to
        # information about that state. Crucially, the state number and the
        # outgoing transitions. We could possibly store the state number
        # together with the state itself, which would be useful because then
        # the transitions would not need to look up the target states'
        # numbers. This would require the state space build to give a number
        # to each state as it is discovered, which in turn would require that
        # it still stores some set/lookup of the state representation to the
        # state number.
        size = len(self.state_space)
        gen_matrix = numpy.zeros((size, size), dtype=numpy.float64)
        for state_number, transitions in self.state_space.values():
            # For the current state we can obtain the set of transitions.
            # This should be known as we would have done this during
            # state_space exploration hence we can given None as the actions
            # dictionary
            total_out_rate = 0.0
            for transition in transitions:
                target_state = transition.successor
                target_info = self.state_space[target_state]
                target_number = target_info.state_number
                # It is += since there may be more than one transition to the
                # same target state from the current state.
                gen_matrix[state_number, target_number] += transition.rate
                total_out_rate += transition.rate
            gen_matrix[state_number, state_number] = -total_out_rate
        return gen_matrix

    @lazy
    def steady_solution(self):
        """Solve the generator matrix to obtain a steady solution"""
        size = len(self.gen_matrix)
        solution_vector = numpy.zeros(size, dtype=numpy.float64)
        solution_vector[0] = 1
        # This is the normalisation bit
        self.gen_matrix[:, 0] = 1
        # Note that here we must transpose the matrix, but arguably we could
        # just build it in the transposed form, since we never use the
        # transposed-form. This would include the above normalisation line.
        result = numpy.linalg.solve(self.gen_matrix.transpose(),
                                    solution_vector)
        return result

    @lazy
    def steady_utilisations(self):
        """ From the steady state create a dictionary of utilisations for
            each component in the system equation
        """
        sys_eqn = self.model.system_equation
        utilisation_builder = UtilisationsBuilderHelper.get_result(sys_eqn)
        for (state, (state_number, transitions)) in self.state_space.items():
            probability = self.steady_solution[state_number]
            utilisation_builder.utilise_state(state, probability)
        return utilisation_builder.get_utilisations()

    def output_steady_utilisations(self, writer):
        for dictionary in self.steady_utilisations:
            writer.write("----------------\n")
            for component, probability in dictionary.items():
                writer.write(component)
                writer.write(" : ")
                writer.write(str(probability))
                writer.write("\n")

# Bio-PEPA stuff


class BioRateDefinition(object):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    grammar = "kineticLawOf" + identifier + ":" + expr_grammar + ";"
    list_grammar = pyparsing.Group(pyparsing.OneOrMore(grammar))

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tokens[1], tokens[3])

BioRateDefinition.grammar.setParseAction(BioRateDefinition.from_tokens)


class BioBehaviour(object):
    def __init__(self, reaction, stoich, role, species):
        self.reaction_name = reaction
        self.stoichiometry = stoich
        self.role = role
        self.species = species

    # If the stoichiometry is 1, then instead of writing "(r, 1)" we allow
    # the modeller to simply write "r".
    # TODO: Consider making the parentheses optional in any case, and then
    # we can simply make the comma-stoich optional.
    prefix_identifier = identifier.copy()
    prefix_identifier.setParseAction(lambda tokens: (tokens[0], 1))

    full_prefix_grammar = "(" + identifier + "," + integer + ")"
    full_prefix_parse_action = lambda tokens: (tokens[1], int(tokens[3]))
    full_prefix_grammar.setParseAction(full_prefix_parse_action)

    prefix_grammar = Or([prefix_identifier, full_prefix_grammar])

    op_strings = ["<<", ">>", "(+)", "(-)", "(.)"]
    role_grammar = Or([Literal(op) for op in op_strings])

    # The true syntax calls for (a,r) << P; where P is the name of the process
    # being updated by the behaviour. However since this is (in the absence
    # of locations) always the same as the process being defined, it is
    # permitted to simply omit it.
    process_update_identifier = Optional(identifier, default=None)
    grammar = prefix_grammar + role_grammar + process_update_identifier

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tokens[0][0], tokens[0][1], tokens[1], tokens[2])

    def get_expression(self, kinetic_laws):
        if self.role == "<<":
            modifier = -1 * self.stoichiometry
        elif self.role == ">>":
            modifier = 1 * self.stoichiometry
        else:
            modifier = 0

        expr = kinetic_laws[self.reaction_name]
        if modifier == 0:
            expr = NumExpression(0.0)
        elif modifier != 1:
            expr = ApplyExpression.multiply(NumExpression(modifier), expr)

        return expr


BioBehaviour.grammar.setParseAction(BioBehaviour.from_tokens)


class BioSpeciesDefinition(object):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    behaviours = pyparsing.delimitedList(BioBehaviour.grammar, delim="+")
    grammar = identifier + "=" + pyparsing.Group(behaviours) + ";"
    list_grammar = pyparsing.Group(pyparsing.OneOrMore(grammar))

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tokens[0], tokens[2])


BioSpeciesDefinition.grammar.setParseAction(BioSpeciesDefinition.from_tokens)


class BioPopulation(object):
    def __init__(self, species, amount):
        self.species_name = species
        self.amount = amount

    grammar = identifier + "[" + integer + "]"

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tokens[0], int(tokens[2]))

BioPopulation.grammar.setParseAction(BioPopulation.from_tokens)

biosystem_grammar = pyparsing.delimitedList(BioPopulation.grammar,
                                            delim="<*>")


def remove_rate_laws(expression, multipliers):
    if expression.name and expression.name == "fMA":
        # TODO: If there are no reactants? I think just the rate expression,
        # which is what this does.
        assert(len(expression.arguments == 1))
        result_expr = expression.arguments[0]
        for (species, stoich) in self.multipliers:
            species_expr = Expression.name_expression(species)
            if stoich != 1:
                # If the stoichiometry is not 1, then we have to raise the
                # speices to the power of the stoichiometry. So if we have
                # fMA(1.0), on a reaction X + Y -> ..., where X has
                # stoichiometry 2, then we get fMA(1.0) = X^2 * Y * 1.0
                stoich_expr = Expression.num_expression(stoich)
                species_expr = Expression.power(species_expr, stoich_expr)
            result_expression = Expression.multiply(result_expr, species_expr)
        return result_expr


class ParsedBioModel(object):
    def __init__(self, constants, kinetic_laws, species, populations):
        self.constants = constants
        self.kinetic_laws = kinetic_laws
        self.species_defs = species
        self.populations = dict()
        for population in populations:
            self.populations[population.species_name] = population.amount

    # Note, this parser does not insist on the end of the input text.
    # Which means in theory you could have something *after* the model text,
    # which might indeed be what you are wishing for.
    grammar = (ConstantDefinition.list_grammar +
               BioRateDefinition.list_grammar +
               BioSpeciesDefinition.list_grammar +
               pyparsing.Group(biosystem_grammar))
    whole_input_grammar = grammar + pyparsing.StringEnd()

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tokens[0], tokens[1], tokens[2], tokens[3])

    def expand_rate_laws(self):
        """ A method to expand the rate laws which are simple convenience
            functions for the user. So we wish to turn:
            kineticLawOf r : fMA(x);
            into
            kineticLawOf r : x * A * B;
            Assuming that A and B are reactants or activators for the
            reaction r
        """
        # First build up a dictionary mapping reaction names to reactants
        # and activators (together with their stoichiometry)
        multipliers = dict()
        for species_def in self.species_defs:
            species_name = species_def.lhs
            behaviours = species_def.rhs
            for behaviour in behaviours:
                if behaviour.role in ["<<", "(+)"]:
                    entry = (species_name, behaviour.stoichiometry)
                    if behaviour.reaction_name in multipliers:
                        entry_list = multipliers[behaviour.reaction_name]
                    else:
                        entry_list = []
                        multipliers[behaviour.reaction_name] = entry_list
                    entry_list.append(entry)
        for kinetic_law in self.kinetic_laws:
            rhs_multipliers = multipliers[kinetic_law.lhs]
            new_expr = remove_rate_laws(kinetic_law.rhs, rhs_multipliers)
            kinetic_law.rhs = new_expr

ParsedBioModel.grammar.setParseAction(ParsedBioModel.from_tokens)


def def_list_as_dictionary(definitions):
    dictionary = dict()
    for definition in definitions:
        dictionary[definition.lhs] = definition.rhs
    return dictionary


def parse_biomodel(model_string):
    """Parses a bio-model ensuring that we have consumed the entire input"""
    return ParsedBioModel.whole_input_grammar.parseString(model_string)[0]


class Configuration(object):
    def __init__(self):
        self.start_time = 0.0
        self.stop_time = 10.0
        self.out_interval = 1.0


class TimeCourse(object):
    def __init__(self, names, rows):
        self.column_names = names
        self.rows = rows


def get_time_grid(configuration):
    """ From a solver configuration return the time points which should
        be returned from the solver
    """
    start_time = configuration.start_time
    stop_time = configuration.stop_time
    out_interval = configuration.out_interval
    # putting stop beyond the actual stop time means that the output
    # will actually include the stop time. Note that in some cases this
    # may result in stop_time + out_interval actually appearing in the
    # output as well, see numpy.arange documentation.
    return numpy.arange(start=start_time,
                        stop=stop_time + out_interval,
                        step=out_interval)


class BioModelSolver(object):
    def __init__(self, model):
        self.model = model

    # TODO: When I am confident that the expression reduction implementation
    # is mature and sound, then I should apply it somewhere here. I probably
    # want a separate method so that all solver methods (odes, ssa, etc.) do
    # not need to re-implement it again. Should be quite simple.

    def solve_odes(self, configuration):
        """ Solves the model, to give a timeseries, by converting the model
            to a series of ODEs.
        """
        self.model.expand_rate_laws()
        kinetic_laws = def_list_as_dictionary(self.model.kinetic_laws)
        # For each species we build an expression to calculate its rate of
        # change based on the current population
        species_names = []
        species_gradients = []
        for species_def in self.model.species_defs:
            species = species_def.lhs
            species_names.append(species)
            behaviours = species_def.rhs
            behaviour_exprs = [b.get_expression(kinetic_laws)
                               for b in behaviours]
            add_exprs = lambda l, r: ApplyExpression("+", [l, r])
            expr = functools.reduce(add_exprs, behaviour_exprs)
            species_gradients.append(expr)

        # We need an environment in which to evaluate each expression, this
        # will consist of the populations (which will be overridden at each
        # time point, and any constants that are used. Theoretically we could
        # instead reduce all of these expressions to remove any use of
        # variables which are assigned to constants (even if indirectly).
        # If we did this, we could have a new environment created inside
        # 'get_rhs' and this would only be used to create the initials.
        # We get the environment of constants *first* since if (later) we
        # allow those constants to contain species variables we would not wish
        # to reduce them using the initial populations.
        environment = constant_def_environment(self.model.constants)
        # So with the environment containing the constant definitions we add
        # the initial populations, these will be overridden at each step.
        environment.update(self.model.populations)

        # TODO: Check what happens if we have (d, 2) (+) E; that is a
        # stoichiometry that is not one for a modifier, rather than a reactant
        # or product. I think this just should not be allowed, correct?

        def get_rhs(current_pops, time):
            """ The main function passed to the solver, it calculates from the
                current populations of species, the rate of change of each
                species. Also given a 'time' which may be used in the
                equations Essentially then, solves for each ODE the right hand
                side of the ode at the given populations and time.
            """
            environment["time"] = time
            for species, population in zip(species_names, current_pops):
                environment[species] = population
                result = [expr.get_value(environment=environment)
                          for expr in species_gradients]
            return result

        # The difficulty here is that initials must be the same order as
        # 'results'
        initials = [environment[name] for name in species_names]

        time_grid = get_time_grid(configuration)
        # Solve the ODEs
        solution = odeint(get_rhs, initials, time_grid)

        timecourse = TimeCourse(species_names, solution)
        return timecourse


# Now the command-line stuff
def run_command_line(default_outfile, argv=None):
    """The default_out argument is used to specify a *default* output file.
       We should also have a command-line option to specify the output file.
       The reason this method takes it in as an argument is to allow
       testing of the command-line interface by output to a memory_file
       (io.StringIO) which can then be inspected.
    """
    arguments = docopt(__doc__, argv=argv, version='pepapot 0.1')
    for filename in arguments['<name>']:
        if arguments['steady'] and arguments['util']:
            with open(filename, "r") as file:
                model = parse_model(file.read())
            model_solver = ModelSolver(model)
            model_solver.output_steady_utilisations(default_outfile)

if __name__ == "__main__":  # pragma: no cover
    import sys
    run_command_line(sys.stdout)
