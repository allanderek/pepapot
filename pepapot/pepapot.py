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

from docopt import docopt
import pyparsing
from pyparsing import Combine, Or, Optional, Literal, Suppress
import numpy
from lazy import lazy

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
expr = floatnumber.copy()
expr.setParseAction(lambda tokens: float(tokens[0]))

rate_grammar = expr


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

    def format(self):
        return "".join(["(", self.action, ", ", str(self.rate),
                        ").", self.successor.format()])

PrefixNode.grammar.setParseAction(PrefixNode.from_tokens)
process_leaf << Or([PrefixNode.grammar, ProcessIdentifier.grammar])


class ChoiceNode(object):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    grammar = pyparsing.Forward()
    grammar << process_leaf + Optional("+" + grammar)

    @classmethod
    def from_tokens(cls, tokens):
        """ A non-typical implementation here, since it might not actually
            produce a 'ChoiceNode' if the number of tokens indicate that the
            optional '+ process_grammar' part is empty.
        """
        if len(tokens) == 3:
            return cls(tokens[0], tokens[2])
        else:
            return tokens

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

ChoiceNode.grammar.setParseAction(ChoiceNode.from_tokens)
# This just sets up an alias because it is otherwise non-obvious that
# ChoiceNode.grammar represents the grammar for generic processes.
process_grammar = ChoiceNode.grammar


class ProcessDefinition(object):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

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


class ModelSolver(object):
    """A full state space exploring model solver. This solver builds the
       entire state-space of the model and from that derives a CTMC which is
       then solved.
    """
    def __init__(self, model):
        self.model = model

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

    grammar = identifier + "=" + rate_grammar + ";"
    list_grammar = pyparsing.Group(pyparsing.OneOrMore(grammar))

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tokens[1], tokens[3])

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

# TODO: Is it not possible to put these setParseActions within the class
# definition? I kind of assume not otherwise I would have done that but I do
# not see why not?
BioRateDefinition.grammar.setParseAction(BioRateDefinition.from_tokens)

class BioBehaviour(object):
    def __init__(self, reaction, stoich, role, species):
        self.reaction_name = reaction
        self.stoichiometry = stoich
        self.role = role
        self.species = species

    prefix_grammar = "(" + identifier + "," + integer + ")"
    role_grammar = "<<"
    grammar = prefix_grammar + role_grammar + identifier

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tokens[0], tokens[1], tokens[2], tokens[3])

BioBehaviour.grammar.setParseAction(BioBehaviour.from_tokens)

class BioSpeciesDefinition(object):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    # TODO: Obviously the right hand side should be a species grammar.
    grammar = identifier + "=" + BioBehaviour.grammar + ";"
    list_grammar = pyparsing.Group(pyparsing.OneOrMore(grammar))

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tokens[0], tokens[2])

    def format(self):
        return " ".join([self.lhs, "=", self.rhs.format(), ";"])

BioSpeciesDefinition.grammar.setParseAction(BioSpeciesDefinition.from_tokens)

class BioPopulation(object):
    def __init__(self, species, amount):
        self.species_name = species
        self.amount = amount

    grammar = identifier + "[" + integer + "]"

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tokens[0], tokens[2])

BioPopulation.grammar.setParseAction(BioPopulation.from_tokens)
biosystem_grammar = pyparsing.Forward()
biosystem_grammar << BioPopulation.grammar + Optional("<*>" + biosystem_grammar)

class ParsedBioModel(object):
    def __init__(self, constants, kinetic_laws, species, populations):
        self.constants = constants
        self.kinetic_laws = kinetic_laws
        self.species_defs = species
        self.populations = dict()
        for population in populations:
            self.populations[population.species_name] = int(population.amount)

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

def parse_biomodel(model_string):
    """Parses a bio-model ensuring that we have consumed the entire input"""
    return ParsedBioModel.whole_input_grammar.parseString(model_string)[0]


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
