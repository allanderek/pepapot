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

    def show_expr(self):
        """ This is a virtual method stub, this method should be overridden
            by any class inheriting from this class"""
        raise NotImplementedError("Expression is really an abstract class")

    def used_names(self):
        """ This is a virtual method stud, this method should be overridden
            by any class inheriting from this class. The overriding method
            should return a set of names used within the expression
        """
        raise NotImplementedError("Expression is really an abstract class")

    def get_value(self, environment=None):
        """ Returns the underlying value of this expression. For complex
            expressions a dictionary mapping names to values may be supplied.
            We raise the exception 'KeyError', if the value cannot be derived,
            this will generally be because a name used in the expression is
            not defined in the given environment (or there is no given
            environment).
        """
        # pylint: disable=W0613
        # pylint: disable=R0201
        raise ValueError("Virtual method 'get_value' called")

    def reduce(self, environment=None):
        """ Similar to 'get_value' except that we always return an expression,
            and in the case that the expression is not wholly reducible to a
            number, it may be reducible to a more simpler expression, for
            example: R * (factor ^ 2)
            if we are given 'factor' as a constant, let's say mapped to '2',
            then we can return the reduced expression: R * 4
            The idea is that if the expression is something like a rate
            expression which must be re-evaluated many times, then we can save
            time by partially evaluating it. However if the expression cannot
            be reduced then we simply return the original expression.
        """
        # pylint: disable=W0613
        # pylint: disable=R0201
        return self

    def munge_names(self, function):
        """ Munges the names used within the expression using the function
            supplied. This is a virtual method stud, see below on our comment
            of the remove_rate_law_sugar method. Essentially I think I should
            be able to do something much nicer, using a visitor pattern.
            Again here this is a bit more than a stub since all the simple
            expressions which cannot contain any names do not need to override
            this stub implementation.
        """
        # pylint: disable=W0613
        # pylint: disable=R0201
        return None

    def remove_rate_law_sugar(self, reaction=None):
        """ This is a virtual method stub, this method should be overridden
            by any class inheriting from this class. In fact we should be
            doing this with something like a visitor pattern, but I have not
            yet fully groked visitor patterns for python.
            Well it's a bit more than a stub, all the very simple expressions
            which don't have sub-expressions do not need to override this.
        """
        # pylint: disable=W0613
        return self


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

    def show_expr(self):
        """Display the underlying number of the numerical expression"""
        return str(self.number)

    def get_value(self, environment=None):
        """Returns the underlying value of this expression"""
        return self.number

    def reduce(self, environment=None):
        """ Returns a reduced expression, it may not be entirely reduced to
            a concrete number, but this is as far as we can reduce it
        """
        return self

    def used_names(self):
        """Return the set of used names, clearly here there are none"""
        return []


class NameExpression(Expression):
    """A class to represent the AST of a variable (name) expression"""
    def __init__(self, name):
        super(NameExpression, self).__init__()
        self.name = name

    def visit(self, visitor):
        """Implements the visit method allowing ExpressionVisitors to work"""
        visitor.visit_NameExpression(self)

    def show_expr(self):
        """Format as a string the name expression"""
        return self.name

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

    def reduce(self, environment=None):
        """ Attempts to reduce this expression, since this is a name
            expression we can reduce it to a number if the value is in the
            constant value mapping provided, otherwise we just return
            ourselves.
        """
        try:
            value = self.get_value(environment=environment)
            return NumExpression(value)
        except KeyError:
            return self

    def used_names(self):
        """Return the set of names used within this expression"""
        return set([self.name])

    def munge_names(self, function):
        self.name = function(self.name)


def show_apply_expression(function_name, children):
    """ Formats an apply expression as a string and returns that string.
        Checks for common arithmetic operators and outputs the appropriate
        infix expression in the case that it finds one.
    """
    function_dict = {"plus": "+",
                     "minus": "-",
                     "divide": "/",
                     "times": "*",
                     "power": "^",
                     }
    # The check on the length of children is just in case someone
    # has managed to say apply 'times' to no arguments which would
    # otherwise cause an error when we attempt to print the first one.
    # It's unclear what we should do in that case, but for now I fall
    # through to the generic case and basically you'll end up with
    # just the 'times' (named as 'times' not as *) printed out.

    result = ""

    if function_name in function_dict and len(children) > 1:
        result += "("
        # Could just put the spaces in the dictionary above?
        operator = " " + function_dict[function_name] + " "
        result += operator.join(children)
        result += ")"
    else:
        result += function_name + "("
        result += ", ".join(children)
        result += ")"

    return result


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

    def visit(self, visitor):
        """Implements the visit method allowing ExpressionVisitors to work"""
        visitor.visit_ApplyExpression(self)

    def show_expr(self):
        """Format as a string the application expression"""
        arg_strings = [arg.show_expr() for arg in self.args]
        return show_apply_expression(self.name, arg_strings)

    def used_names(self):
        """Return the set of names used within this apply expression"""
        result_set = set()
        for expr in self.args:
            result_set = result_set.union(expr.used_names())
        return result_set

    def munge_names(self, function):
        """ Must munge all the names, we do not munge the name of the
            function of the apply expression however.
        """
        for child in self.args:
            child.munge_names(function)

    # TODO: This should not be in here if we really want the parsing of
    # expressions to be separate from Bio-PEPA evaluation. We should have a
    # general way to remove expressions. We could even restrict this to
    # specifically apply expressions but it should be more general.
    def remove_rate_law_sugar(self, reaction=None):
        # First apply this to all of the argument expressions.
        new_args = [arg.remove_rate_law_sugar(reaction) for arg in self.args]
        self.args = new_args

        if reaction is not None and self.name == "fMA":
            # Should do some more error checking, eg if there is exactly
            # one argument.
            mass_action_reactants = reaction.get_mass_action_participants()
            extra_args = [NameExpression(reactant.get_name())
                          for reactant in mass_action_reactants]
            all_args = new_args + extra_args
            # fMA should have exactly one argument, the additional arguments
            # are the populations of all the reactants/modifiers of the
            # reaction. It could be that there are no such, in otherwords we
            # have a source reaction
            if len(all_args) > 1:
                new_expr = ApplyExpression("times", new_args + extra_args)
                return new_expr
            else:
                # If there is only the original argument then just return that
                # even without the surrounding 'fMA' application.
                return all_args[0]
        else:
            new_expr = ApplyExpression(self.name, new_args)
            return new_expr

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
        elif self.name == "divide" or self.name == "/":
            answer = arg_values[0]
            for arg in arg_values[1:]:
                answer /= arg
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

    def reduce(self, environment=None):
        """ Attempts to reduce this expression, note that there are three
            possibilities, but the first is that the entire expression cannot
            be reduced any further, in which case we can just return the
            current expression, but we can ignore this possibility. Another
            possibility is that some of the argument expressions can be
            reduced but not all, in which case we return a new apply
            expression with the reduced (as far as they can be) argument
            expressions. Finally the case in which all expressions can be
            reduced, in which case we return the value.
        """
        # We can easily check the final case by simply calling 'get_value' on
        # this expression, if it doesn't reduce we can assume that at least
        # one argument expression doesn't reduce.
        # Note that this means we in some sense do a little of the work twice
        # in the worst case we may have many arguments which reduce to a value
        # and only one which does not, in which case we could have reduced
        # all arg expressions and then applied the function if they all
        # reduced to a NumExpression, otherwise build up the NameExpression
        # with the reduced expression. The point is that here we assume you
        # are doing this reduction once at the start of say a simulation and
        # hence you don't require this to be extremely fast, and this is a
        # very nice definition which means we need not write code to evaluate
        # plus/minus etc twice. Alternatively we could write 'get_value' in
        # terms of 'reduce' but that would mean building up NumExpressions
        # more than we needed to.
        try:
            value = self.get_value(environment=environment)
            return NumExpression(value)
        except KeyError:
            # We have a special case for the commutative expression plus and
            # times, here we try to pull out all of the argument expressions
            # which reduce to a value and sum or product them together. This
            # is simply so that we can turn the expression:
            # R * 0.2 * 10 where R is a dynamic variable into the expression
            # R * 2
            # which may save a bit of time during a simulation.
            if self.name == "plus" or self.name == "times":
                factors = []
                arg_expressions = []
                for arg in self.args:
                    try:
                        factors.append(arg.get_value(environment=environment))
                    except KeyError:
                        reduced_arg = arg.reduce(environment=environment)
                        arg_expressions.append(reduced_arg)

                # Based on whether it's plus or times we must (possibly) add
                # the correct factor argument into the list of
                # argument_expressions. At the end of this conditional
                # arg_expressions will be the correct set of argument
                # expressions.
                if self.name == "plus":
                    factor = sum(factors)
                    # So if factor is not zero then we must add it as an arg.
                    if factor != 0:
                        factor_exp = NumExpression(factor)
                        arg_expressions = [factor_exp] + arg_expressions
                else:  # assume it equals "times", we above check this.
                    factor = list_product(factors)
                    if factor != 1:
                        factor_exp = NumExpression(factor)
                        arg_expressions = [factor_exp] + arg_expressions
                # Now that we have the correct set of argument expressions
                # we may return the reduced apply expression, but we first
                # just check that we have not reduced it to a single
                # expression, in which case we can simply return that
                # expression, eg we may have started with R * 0.1 * 10, which
                # would reduce to R * 1, but since then the factor would be 1
                # we would not have added it as an arg_expression.
                if len(arg_expressions) == 1:
                    return arg_expressions[0]
                else:
                    return ApplyExpression(self.name, arg_expressions)
            else:
                # The easy case for non-commutative, we could go deeper and
                # try to partially evaluate some of these, for example
                # R - 3    - 1 could be made into R - 4. But for now the above
                # will do, since I believe that multiplications by more than
                # one constant are fairly common.
                arg_expressions = [arg.reduce(environment=environment)
                                   for arg in self.args]
                return ApplyExpression(self.name, arg_expressions)


class ExpressionVisitor(object):
    """ A parent class for classes which descend through the abstract syntax
        of expressions, generally storing a result along the way.
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

    ###################################
    # These are the unimplemented methods that you would be likely
    # to override for your expression visitor.
    # pylint: disable=C0103
    def visit_NumExpression(self, _expression):
        """Visit a NumExpression element"""
        # pylint: disable=R0201
        message = "visit_NumExpression element for expression visitor"
        raise NotImplementedError(message)

    def visit_NameExpression(self, _expression):
        """Visit a NameExpression"""
        # pylint: disable=R0201
        message = "visit_NameExpression element for expression visitor"
        raise NotImplementedError(message)

    def visit_ApplyExpression(self, _expression):
        """Visit an ApplyExpression element"""
        # pylint: disable=R0201
        message = "visit_ApplyExpression element for expression visitor"
        raise NotImplementedError(message)


Action = namedtuple('Action', ["action", "rate", "successor"])

identifier = pyparsing.Word(pyparsing.alphanums + "_")

# TODO: There is a fairly good calculator parsing example which includes
# identifiers as expressions. It can be found at:
# pyparsing.wikispaces.com/file/view/SimpleCalc.py/30112812/SimpleCalc.py
plusorminus = Literal('+') | Literal('-')
number = pyparsing.Word(pyparsing.nums)
integer = Combine(Optional(plusorminus) + number)
decimal_fraction = Literal('.') + number
scientific_enotation = pyparsing.CaselessLiteral('E') + integer
floatnumber = Combine(integer + Optional(decimal_fraction) +
                      Optional(scientific_enotation))


num_expr = floatnumber.copy()
num_expr.setParseAction(lambda tokens: NumExpression(float(tokens[0])))

name_expr = identifier.copy()
name_expr.setParseAction(lambda tokens: NameExpression(tokens[0]))

atom_expr = Or([num_expr, name_expr])

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


expr_grammar = term_expr
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
            expr = ApplyExpression("*", [NumExpression(modifier), expr])

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

    def format(self):
        behaviours_string = " + ".join([b.format() for b in self.rhs])
        return " ".join([self.lhs, "=", behaviours_string, ";"])

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
