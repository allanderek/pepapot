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

from docopt import docopt
import pyparsing
from pyparsing import Combine, Or, Optional, Literal, Suppress
import numpy
from scipy.integrate import odeint
from lazy import lazy


class Expression:
    """ The base class for all classes which represent the AST of some
        kind of expression"""
    def __init__(self):
        pass


class NumExpression(Expression):
    """A class to represent the AST of an number literal expression"""
    def __init__(self, number):
        super(NumExpression, self).__init__()
        self.number = number

    def __eq__(self, other_expression):
        # TODO: A touch questionable, we should decide whether we want
        # *equivalent* expressions to equal each other.
        return self.get_value() == other_expression.get_value()

    def visit(self, visitor):
        """Implements the visit method allowing ExpressionVisitors to work"""
        visitor.visit_NumExpression(self)

    def get_value(self, environment=None):
        """Returns the underlying value of this expression"""
        return self.number


class NameExpression(Expression):
    """A class to represent the AST of a variable (name) expression"""
    def __init__(self, name):
        super(NameExpression, self).__init__()
        self.name = name

    def visit(self, visitor):
        """Implements the visit method allowing ExpressionVisitors to work"""
        visitor.visit_NameExpression(self)

    def get_value(self, environment=None):
        """ Evalutes this expression based on the given variable_dictionary,
            If the environment is given and defines the name which is this
            expression then we return whatever the environment defines this
            name to be. Otherwise, as per 'get_value' we raise the expception
            KeyError.
        """
        # This is simple, we just do whatever looking up in the environment
        # does, which is the advertised behaviour of get_value.
        if environment is not None:
            return environment[self.name]
        else:
            raise KeyError(self.name)


def list_product(factors):
    """ Simple utility the same as 'sum' but for the product of the arguments.
        Note: returns 1 for the empty list, which seems reasonable, given that
        sum([]) = 0.
    """
    result = 1
    for factor in factors:
        result *= factor
    return result


class ApplyExpression(Expression):
    """ A class to represent the AST of an apply expression, applying a
        named function to a list of argument expressions
    """
    def __init__(self, name, args):
        super(ApplyExpression, self).__init__()
        self.name = name
        self.args = args

    @classmethod
    def addition(cls, left, right):
        return cls("+", [left, right])

    @classmethod
    def subtract(cls, left, right):
        return cls("-", [left, right])

    @classmethod
    def multiply(cls, left, right):
        return cls("*", [left, right])

    @classmethod
    def divide(cls, left, right):
        return cls("/", [left, right])

    def visit(self, visitor):
        """Implements the visit method allowing ExpressionVisitors to work"""
        visitor.visit_ApplyExpression(self)

    def get_value(self, environment=None):
        """ Return the value to which the expression evaluates. If any
            environment is given it should be used to resolve any names in
            the sub-expressions of this expression. If the expression is
            irreducible, generally because it contains a name which is not
            in the given environment (or none is given) then None is returned.
        """
        arg_values = [arg.get_value(environment=environment)
                      for arg in self.args]
        if self.name == "plus" or self.name == "+":
            return sum(arg_values)
        elif self.name == "times" or self.name == "*":
            return list_product(arg_values)
        elif self.name == "minus" or self.name == "-":
            # What should we do if there is only one argument, I think we
            # should treat '(-) x' the same as '0 - x'.
            answer = arg_values[0]
            for arg in arg_values[1:]:
                answer -= arg
            return answer
        elif self.name == "divide" or self.name == "/":
            answer = arg_values[0]
            for arg in arg_values[1:]:
                answer /= arg
            return answer
        elif self.name == "power" or self.name == "**":
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
            raise ValueError("Unknown function name: " + self.name)


class ExpressionVisitor(object):
    """ A parent class for classes which descend through the abstract syntax
        of expressions, generally storing a result along the way.
        There are two kinds of expression visitors, ones which do not modify
        the expressions but are merely used to descend through the expression
        tree perhaps building up a result, such as the set of used variable
        names. ExpressionVisitor implements that kind of Visitor. The second
        kind of visitor is used to modify the given expression, for example
        you may wish to reduce expressions or remove some kind of sugar. See
        ExpressionModifierVisitor for that kind of visitor.
    """
    def __init__(self):
        self.result = None

    def generic_visit(self, expression):
        """ The main entry for visiting generic expression whose type we do
            not yet know, this is the most klutchy part of this, but there is
            no way around this.
        """
        expression.visit(self)

    def generic_visit_get_results(self, expression):
        """ Performs the visit and also returns the result, sort of useful
            for doing this within a list comprehension.
        """
        self.generic_visit(expression)
        return self.result

    def visit_NumExpression(self, expression):
        """Visit a NumExpression element"""
        pass

    def visit_NameExpression(self, expression):
        """Visit a NameExpression"""
        pass

    def visit_ApplyExpression(self, expression):
        """Visit an ApplyExpression element"""
        for arg in expression.args:
            arg.visit(self)


class ExpressionModifierVisitor(ExpressionVisitor):
    """ ExpressionModifierVisitor builds ontop of ExpressionVisitor to supply
        a base class for the kind of visitor which needs to return a new,
        possibly modified expression. The main difference is that in
        ExpressionModifierVisitor the result is set to the expression itself.
        In short, use this one if your visitor may need to return an entirely
        new expression, rather than simply modify the current one in place.
        So for example, 'reduce' might wish to turn a:
        ApplyExpression('*', [NumExpression(1), expr])
        into simply
        expr
        To do so it could not simply modify the given ApplyExpression in place
        since that may be contained within another kind of expression itself
        and 'expr' might be something other than an ApplyExpression.
    """
    def visit_NumExpression(self, expression):
        """Visit a NumExpression element"""
        self.result = expression

    def visit_NameExpression(self, expression):
        """Visit a NameExpression"""
        self.result = expression

    def visit_ApplyExpression(self, expression):
        """ Visit an ApplyExpression element. Note that if you override this
            you will likely still wish to recursively visit the argument
            expressions, you can do this by calling this method using 'super'.
            Also note that if you really wish to leave the original expression
            untouched then you need to do the recursive calling yourself.
        """
        expression.args = [self.generic_visit_get_results(e)
                           for e in expression.args]
        self.result = expression

Action = namedtuple('Action', ["action", "rate", "successor"])


identifier_start = pyparsing.Word(pyparsing.alphas + "_", exact=1)
identifier_remainder = pyparsing.Word(pyparsing.alphanums + "_")
identifier = pyparsing.Combine(identifier_start +
                               Optional(identifier_remainder))

# TODO: Check out pyparsing.operatorPrecedence and pyparsing.nestedExpr
# http://pyparsing.wikispaces.com/file/view/simpleArith.py/
# http://pyparsing.wikispaces.com/file/view/nested.py/32064753/nested.py
plusorminus = Literal('+') | Literal('-')
number = pyparsing.Word(pyparsing.nums)
integer = Combine(Optional(plusorminus) + number)
decimal_fraction = Literal('.') + number
scientific_enotation = pyparsing.CaselessLiteral('E') + integer
floatnumber = Combine(integer + Optional(decimal_fraction) +
                      Optional(scientific_enotation))


expr_grammar = pyparsing.Forward()
num_expr = floatnumber.copy()
num_expr.setParseAction(lambda tokens: NumExpression(float(tokens[0])))


# A helper to create grammar element which must be surrounded by parentheses
# but you then wish to ignore the parentheses
def parenthetical_grammar(element_grammar):
    return Suppress("(") + element_grammar + Suppress(")")


def apply_expr_parse_action(tokens):
    if len(tokens) == 1:
        return NameExpression(tokens[0])
    else:
        return ApplyExpression(tokens[0], tokens[1:])
arg_expr_list = pyparsing.delimitedList(expr_grammar)
apply_expr = identifier + Optional(parenthetical_grammar(arg_expr_list))
apply_expr.setParseAction(apply_expr_parse_action)


atom_expr = Or([num_expr, apply_expr])

multop = Literal('*') | Literal('/')
factor_expr = pyparsing.Forward()
factor_expr << atom_expr + Optional(multop + factor_expr)


def factor_parse_action(tokens):
    if len(tokens) > 1:
        operator = tokens[1]
        return ApplyExpression(operator, [tokens[0], tokens[2]])
    else:
        return tokens[0]
factor_expr.setParseAction(factor_parse_action)

term_expr = pyparsing.Forward()
term_expr << factor_expr + Optional(plusorminus + term_expr)


term_parse_action = factor_parse_action
term_expr.setParseAction(term_parse_action)


expr_grammar << term_expr
rate_grammar = expr_grammar


class ProcessIdentifier(object):
    def __init__(self, name):
        self.name = name

    grammar = identifier.copy()

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tokens[0])

    def __str__(self):
        return self.name

    def get_used_process_names(self):
        return set([self.name])

    def format(self):
        return self.name

ProcessIdentifier.grammar.setParseAction(ProcessIdentifier.from_tokens)
process_leaf = pyparsing.Forward()


class PrefixNode(object):
    def __init__(self, action, rate, successor):
        self.action = action
        self.rate = rate
        self.successor = successor

    grammar = "(" + identifier + "," + rate_grammar + ")" + "." + process_leaf

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tokens[1], tokens[3], tokens[6])

    def get_used_process_names(self):
        return self.successor.get_used_process_names()

    def get_possible_actions(self):
        return [Action(self.action, self.rate, str(self.successor))]

    def concretise_actions(self, environment=None):
        self.rate = self.rate.get_value()

    def format(self):
        return "".join(["(", self.action, ", ", str(self.rate),
                        ").", self.successor.format()])

PrefixNode.grammar.setParseAction(PrefixNode.from_tokens)
process_leaf << Or([PrefixNode.grammar, ProcessIdentifier.grammar])


class ChoiceNode(object):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def get_possible_actions(self):
        left_actions = self.lhs.get_possible_actions()
        right_actions = self.rhs.get_possible_actions()
        # Because we are not using sets here it is possible that we have
        # duplicates, this is interesting, I'm not sure what to make of,
        # for example, "P = (a,r).P1 + (a,r).P1",
        # should it occur at twice the rate?
        # We could detect duplicates at this stage and multiply the rate.
        # In fact they would not need to be duplicates, simply sum the rates,
        # ie.: "P = (a,r).P1 + (a,t).P1" is equivalent to "P = (a, r+t).P1".
        return left_actions + right_actions

    def concretise_actions(self, environment=None):
        self.lhs.concretise_actions(environment)
        self.rhs.concretise_actions(environment)

    def get_used_process_names(self):
        lhs = self.lhs.get_used_process_names()
        rhs = self.rhs.get_used_process_names()
        return lhs.union(rhs)

    def format(self):
        # I believe there is no need for  parentheses, since we cannot make
        # a mistake since the only binary operator is +. Currently we cannot
        # have P = (a, r).(Q + R); which would make things ambiguous since
        # P = (a, r).Q + R could be either P = ((a.r).P) + R; or
        # P = (a, r).(Q + R);
        return " ".join([self.lhs.format(), "+", self.rhs.format()])


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

    def get_used_process_names(self):
        return set([self.identifier])

    def get_shared_actions(self):
        """Mostly for testing purposes we return all activities shared
           at least once"""
        return set()

    def get_builder(self, builder_helper):
        return builder_helper.leaf(self.identifier)

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

    def get_used_process_names(self):
        return self.lhs.get_used_process_names()

    def get_shared_actions(self):
        return self.lhs.get_shared_actions()

    def get_builder(self, builder_helper):
        return builder_helper.aggregation(self.lhs, self.amount)

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

    def get_used_process_names(self):
        lhs = self.lhs.get_used_process_names()
        rhs = self.rhs.get_used_process_names()
        return lhs.union(rhs)

    def get_shared_actions(self):
        """Mostly for testing purposes we return all activities shared
           at least once"""
        left = self.lhs.get_shared_actions()
        right = self.rhs.get_shared_actions()
        these = set(self.cooperation_set)
        return these.union(left).union(right)

    def get_builder(self, builder_helper):
        return builder_helper.cooperation(self.lhs,
                                          self.cooperation_set,
                                          self.rhs)

    def format(self):
        coop_string = "<" + ", ".join(self.cooperation_set) + ">"
        return " ".join([self.lhs.format(), coop_string, self.rhs.format()])


system_equation_grammar << ParsedSystemCooperation.grammar
system_equation_grammar.setParseAction(ParsedSystemCooperation.from_tokens)


class ParsedModel(object):
    def __init__(self, proc_defs, sys_equation):
        self.process_definitions = proc_defs
        self.system_equation = sys_equation

    # Note, this parser does not insist on the end of the input text.
    # Which means in theory you could have something *after* the model text,
    # which might indeed be what you are wishing for.
    grammar = ProcessDefinition.list_grammar + system_equation_grammar
    whole_input_grammar = grammar + pyparsing.StringEnd()

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tokens[0], tokens[1])

    def get_process_definition(self, name):
        """ Returns the process definition which defines the given name.
            This may raise StopIteration if no such definition exists
        """
        return next(x for x in self.process_definitions if x.lhs == name)

    def get_components(self):
        """Returns a dictionary mapping each name used in the system equation
           to a list of names reachable via actions from that name.
        """
        # Note that we could do a bit of memoisation here, since
        # 'get_components' is used in both 'used_process_names' and
        # 'get_initial_state', but we do not expect this to take a long time.
        used_processes = self.system_equation.get_used_process_names()
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
                    new_names = definition.rhs.get_used_process_names()
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
            actions = definition.rhs.get_possible_actions()
            actions_dictionary[definition.lhs] = actions
        return actions_dictionary

    def defined_process_names(self):
        """Return the list of defined process names"""
        names = [definition.lhs for definition in self.process_definitions]
        return set(names)

    def get_builder(self, builder_helper):
        return self.system_equation.get_builder(builder_helper)

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


class InitialStateBuilderHelper(object):
    def __init__(self, components):
        self.components = components

    def leaf(self, identifier):
        # A leaf state is simply an identifier
        return identifier

    def aggregation(self, lhs, amount):
        # This assumes that lhs will be an identifier, which as I write this
        # is enforced by the parser, but ultimately it would be good to allow
        # (P <*> Q)[X]
        initial_name = lhs.get_builder(self)
        state_names = self.components[initial_name]
        pairs = [(x, amount if x == initial_name else 0) for x in state_names]
        # An aggregation state is a tuple consisting of pairs. Each pair is
        # the name of a local state and the number of components in that state
        return tuple(pairs)

    def cooperation(self, lhs, _coop_set, rhs):
        # A cooperation state is simply a pair consisting of the left and
        # right sub-states
        return (lhs.get_builder(self), rhs.get_builder(self))


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


class StateBuilderHelper(object):
    def __init__(self, actions_dictionary):
        self.actions_dictionary = actions_dictionary

    def leaf(self, _identifier):
        return LeafBuilder(self.actions_dictionary)

    def aggregation(self, lhs, _amount):
        return AggregationBuilder(lhs.get_builder(self))

    def cooperation(self, lhs, coop_set, rhs):
        return CoopBuilder(lhs.get_builder(self), coop_set,
                           rhs.get_builder(self))


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


class UtilisationsBuilderHelper(object):
    def __init__(self):
        pass

    def leaf(self, _identifier):
        return LeafUtilisations()

    def aggregation(self, lhs, _amount):
        return AggregationUtilisations()

    def cooperation(self, lhs, _coop_set, rhs):
        return CoopUtilisations(lhs.get_builder(self), rhs.get_builder(self))

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
#       builder_helper = StateBuilderHelper(self.model.get_process_actions())
#       state_builder = self.model.get_builder(builder_helper)
#       explore_queue = set([self.initial_state])


class ModelSolver(object):
    """A full state space exploring model solver. This solver builds the
       entire state-space of the model and from that derives a CTMC which is
       then solved.
    """
    def __init__(self, model):
        self.model = model
        # TODO: Clearly we should be passing in some kind of environment to
        # concretise_actions, that environment should come from the rate
        # constant definitions of the model.
        for proc_def in self.model.process_definitions:
            proc_def.rhs.concretise_actions()

    @lazy
    def initial_state(self):
        components = self.model.get_components()
        builder_helper = InitialStateBuilderHelper(components)
        self._initial_state = self.model.get_builder(builder_helper)
        return self._initial_state

    @lazy
    def state_space(self):
        builder_helper = StateBuilderHelper(self.model.get_process_actions())
        state_builder = self.model.get_builder(builder_helper)
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
        builder_helper = UtilisationsBuilderHelper()
        utilisation_builder = self.model.get_builder(builder_helper)
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


# TODO: Can we make an abstract base class for definitions?
# TODO: We can at least do this for constants, currently the PEPA
# implementation above does not allow for constant declarations, one problem
# is that in PEPA, it is difficult to distinguish between an expression and
# a process, for example:
# P = Q + R;
# Could be either, unless we insist that rates are lower case and processes
# are upper case? Not sure I wish to do that.
class BioRateConstant(object):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    grammar = identifier + "=" + expr_grammar + ";"
    list_grammar = pyparsing.Group(pyparsing.OneOrMore(grammar))

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tokens[0], tokens[2])

BioRateConstant.grammar.setParseAction(BioRateConstant.from_tokens)


class BioRateDefinition(object):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    grammar = "kineticLawOf" + identifier + ":" + rate_grammar + ";"
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


class RemoveRateLawsVisitor(ExpressionModifierVisitor):
    """ Removes the rate laws syntax sugar from an expression. Currently only
        fMA(r) is implemented. Note this uses ExpressionModifierVisitor, so
        if you call this you will likely used 'generic_visit_get_results' as
        the original expression may not be modified but a new one returned
        in its place. For fMA we could arguably do this using an ordinary
        ExpressionVisitor since the result is still going to be an
        ApplyExpression anyway, but this seems cleaner.
    """
    def __init__(self, multipliers):
        super(RemoveRateLawsVisitor, self).__init__()
        self.multipliers = multipliers

    def visit_ApplyExpression(self, apply_expr):
        super(RemoveRateLawsVisitor, self).visit_ApplyExpression(apply_expr)
        # TODO: If there are no reactants? I think just the rate expression,
        # which is what this does.
        if apply_expr.name == "fMA":
            assert(len(apply_expr.args) == 1)
            arg_expression = apply_expr.args[0]
            arg_expression.visit(self)
            expr = arg_expression

            for (species, stoich) in self.multipliers:
                species_expr = NameExpression(species)
                if stoich != 1:
                    num_expr = NumExpression(stoich)
                    species_expr = ApplyExpression.multiply(num_expr,
                                                            species_expr)
                expr = ApplyExpression.multiply(expr, species_expr)
            self.result = expr


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
    grammar = (BioRateConstant.list_grammar +
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
            visitor = RemoveRateLawsVisitor(multipliers[kinetic_law.lhs])
            new_expr = visitor.generic_visit_get_results(kinetic_law.rhs)
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
        environment = self.model.populations.copy()

        # For now though we will just add the constants to the environment
        for constant_def in self.model.constants:
            name = constant_def.lhs
            value = constant_def.rhs.get_value(environment=environment)
            environment[name] = value

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
