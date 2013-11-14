"""pypepa.

Usage:
  pypepa.py steady util <name>...
  pypepa.py -h | --help
  pypepa.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  """
from docopt import docopt

import logging

from collections import namedtuple

import pyparsing
from pyparsing import Combine, OneOrMore, Or, Group, Optional
import numpy
from lazy import lazy

Action = namedtuple('Action', ["action", "rate", "successor" ])

identifier = pyparsing.Word(pyparsing.alphanums)

# TODO: There is a fairly good calculator parsing example which includes
# identifiers as expressions. It can be found at:
# http://pyparsing.wikispaces.com/file/view/SimpleCalc.py/30112812/SimpleCalc.py
plusorminus = pyparsing.Literal('+') | pyparsing.Literal('-')
number = pyparsing.Word(pyparsing.nums) 
integer = Combine( Optional(plusorminus) + number )
floatnumber = Combine( integer + 
                       Optional( pyparsing.Literal('.') + number ) +
                       Optional( pyparsing.CaselessLiteral('E') + integer )
                     )
expr = floatnumber.copy()
expr.setParseAction(lambda tokens: float(tokens[0]))

rate_grammar = expr

class ProcessIdentifier(object):
    def __init__(self, tokens):
        self.name = tokens[0]

    def __str__(self):
        return self.name

    def get_used_process_names(self):
        return set([ self.name ])
process_identifier = identifier.copy()
process_identifier.setParseAction(ProcessIdentifier)

process_leaf = pyparsing.Forward()

class PrefixNode(object):
    def __init__(self, tokens):
        self.action = tokens[1]
        self.rate = tokens[3]
        self.successor = tokens[6]

    def get_used_process_names(self):
        return self.successor.get_used_process_names()

    def get_possible_actions(self):
        return [ Action(self.action, self.rate, str(self.successor)) ]

prefix_grammar  = "(" + identifier + "," + rate_grammar + ")" + "." + process_leaf
prefix_grammar.setParseAction(PrefixNode)

class ChoiceNode(object):
    def __init__(self, tokens):
        self.lhs = tokens[0]
        self.rhs = tokens[2]

    def get_possible_actions(self):
        left_actions  = self.lhs.get_possible_actions()
        right_actions = self.rhs.get_possible_actions()
        # Because we are not using sets here it is possible that we have
        # duplicates, this is interesting, I'm not sure what to make of, for
        # examples "P = (a,r).P1 + (a,r).P1", should it occur at twice the rate?
        # We could detect duplicates at this stage and double the rate. In fact
        # they would not need to be duplicates, simply sum the rates, eg:
        # "P = (a,r).P1 + (a,t).P1" is equivalent to "P = (a, r+t).P1".
        return left_actions + right_actions

    def get_used_process_names(self):
        lhs = self.lhs.get_used_process_names()
        rhs = self.rhs.get_used_process_names()
        return lhs.union(rhs)

process_leaf << Or([prefix_grammar, process_identifier])
process_grammar = pyparsing.Forward()
process_grammar << process_leaf + Optional ("+" + process_grammar)
def create_process (tokens):
    if len(tokens) == 3:
        return ChoiceNode(tokens)
    else:
        return tokens
process_grammar.setParseAction(create_process)



class ProcessDefinition(object):
    def __init__(self, tokens):
        self.lhs = tokens[0]
        self.rhs = tokens[2]
process_definition_grammar = identifier + "=" + process_grammar + ";"
process_definition_grammar.setParseAction(ProcessDefinition)
process_definitions_grammar = Group(OneOrMore(process_definition_grammar))

activity_list_grammar = "<" + pyparsing.delimitedList(identifier, ",") + ">"
cooperation_set_grammar =Or([pyparsing.Literal("||"), activity_list_grammar])
def get_action_set(tokens):
    # It's a double list because otherwise the system_equation_parser will
    # assume the list returned is a set of tokens and concatenate it in with
    # the other tokens.
    return [[ x for x in tokens if x not in [ "||", "<", ">"] ]]
cooperation_set_grammar.setParseAction(get_action_set)


class ParsedNamedComponent(object):
    def __init__(self, tokens):
        self.identifier = tokens[0]
    def get_used_process_names(self):
        return set(self.identifier)
    def get_shared_actions(self):
        """Mostly for testing purposes we return all activities shared
           at least once"""
        return set()
    def get_builder(self, builder_helper):
        return builder_helper.leaf(self.identifier)

class ParsedAggregation(object):
    def __init__(self, tokens):
        self.lhs = tokens[0]
        self.amount = tokens[1]
    def get_used_process_names(self):
        return self.lhs.get_used_process_names()
    def get_shared_actions(self):
        return self.lhs.get_shared_actions()
    def get_builder(self, builder_helper):
        return builder_helper.aggregation(self.lhs, self.amount)

class ParsedSystemCooperation(object):
    def __init__(self, tokens):
        self.lhs = tokens[0]
        self.cooperation_set = tokens[1]
        self.rhs = tokens[2]

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

system_equation_grammar = pyparsing.Forward()
system_equation_ident = identifier.copy()
system_equation_ident.setParseAction(ParsedNamedComponent)
# Forces this to be a non-negative integer, though could be zero. Arguably
# we may want to allow decimals here, obviously only appropriate for
# translation to ODEs.
array_suffix = "[" + number + "]"
array_suffix.setParseAction(lambda x: int(x[1]))
# This way means that aggregation can only be applied to a single identifier
# such as "P[10]". We could also allow for example "(P <a> Q)[10]".
def create_aggregation(tokens):
    if len(tokens) > 1:
        return ParsedAggregation(tokens)
    else:
        return tokens
system_equation_aggregation = system_equation_ident + Optional(array_suffix)
system_equation_aggregation.setParseAction(create_aggregation)
system_equation_paren = "(" + system_equation_grammar + ")"
system_equation_paren.setParseAction(lambda x: x[1])
system_equation_atom = Or ([system_equation_aggregation,
                            system_equation_paren])

def create_system_component(tokens):
    if len(tokens) > 1:
        return ParsedSystemCooperation(tokens)
    else:
        return tokens
system_equation_grammar << (system_equation_atom +
                            Optional(cooperation_set_grammar + 
                                     system_equation_grammar))
system_equation_grammar.setParseAction(create_system_component)

class ParsedModel(object):
    def __init__(self, tokens):
        self.process_definitions = tokens[0]
        self.system_equation = tokens[1]

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
                    definition = [ x for x in self.process_definitions
                                   if x.lhs == name ][0]
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
        return set([definition.lhs for definition in self.process_definitions ])

    def get_builder(self, builder_helper):
        return self.system_equation.get_builder(builder_helper)

# Note, this parser does not insist on the end of the input text. Which means
# in theory you could have something *after* the model text, which might indeed
# be what you are wishing for. See parse_model for a whole input parser
model_grammar = process_definitions_grammar + system_equation_grammar
model_grammar.setParseAction(ParsedModel)

def parse_model(model_string):
    # Parses a model and also ensures that we have consumed the entire input
    whole_input_parser = model_grammar + pyparsing.StringEnd()
    return whole_input_parser.parseString(model_string)[0]


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
    def _compute_transitions(self, state): #pragma: no cover
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
        transitions = [ Transition(a.action, a.rate, a.successor) 
                        for a in actions ]
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
                        successor = tuple([ (s, new_number(s,n)) 
                                             for (s, n) in state ])
                    # I'm not 100% this always correct. Should we rather add
                    # a number of new transitions (ie. num) where each
                    # transition has the original rate?
                    new_transition = Transition (transition.action,
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
                new_transition = Transition(transition.action, transition.rate,  new_state)
                transitions.append(new_transition)
        for transition in right_transitions:
            if transition.action not in self.coop_set:
                new_state = (left_state, transition.successor)
                new_transition = Transition(transition.action, transition.rate,  new_state)
                transitions.append(new_transition)
        for action in self.coop_set:
            left_shared = [ t for t in left_transitions if t.action == action]
            right_shared = [ t for t in right_transitions if t.action == action]
            left_rate = sum([ t.rate for t in left_shared ])
            right_rate = sum([t.rate for t in right_shared])
            governing_rate = min(left_rate, right_rate)
            for (left, right) in [ (l, r) for l in left_shared for r in right_shared ]:
                rate = (left.rate / left_rate) * (right.rate / right_rate) * governing_rate
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
        # We assume that state is a string
        self.utilisations[state] = self.utilisations.get(state, 0.0) + probability
    def get_utilisations(self):
        return [ self.utilisations ]
    
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
        return [ self.utilisations ]

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
            successor_states = [ t.successor for t in transitions ]
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
                target_state_number = target_info.state_number
                # It is += since there may be more than one transition to the
                # same target state from the current state.
                gen_matrix[state_number, target_state_number] += transition.rate
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
        self.gen_matrix[:,0] = 1
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


# Now the command-line stuff
def run_command_line(argv=None):
    arguments = docopt(__doc__, version='pypepa 0.1')
    for filename in arguments['<name>']:
        if arguments['steady'] and arguments['util']:
            with open(filename, "r") as file:
                model = parse_model(file.read())
            model_solver = ModelSolver(model)
            steady_utilisations = model_solver.steady_utilisations
            print (steady_utilisations)

if __name__ == "__main__": # pragma: no cover
    run_command_line()

