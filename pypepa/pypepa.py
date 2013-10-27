"""The main source of the pypepa library"""

import argparse
import logging

from collections import namedtuple

import pyparsing
from pyparsing import Combine, OneOrMore, Or, Group, Optional
import numpy

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
    def get_initial_state(self):
        return self.identifier
    def get_state_builder(self, actions_dictionary):
        return LeafBuilder(actions_dictionary)

class ParsedAggregation(object):
    def __init__(self, tokens):
        self.lhs = tokens[0]
        self.amount = tokens[1]
    def get_shared_actions(self):
        return self.lhs.get_shared_actions()

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

    def get_initial_state(self):
        return (self.lhs.get_initial_state(), self.rhs.get_initial_state())
    def get_state_builder(self, actions_dictionary):
        return CoopBuilder(self.lhs.get_state_builder(actions_dictionary),
                           self.cooperation_set,
                           self.rhs.get_state_builder(actions_dictionary))

system_equation_grammar = pyparsing.Forward()
system_equation_ident = identifier.copy()
system_equation_ident.setParseAction(ParsedNamedComponent)
# Forces this to be a non-negative integer, though could be zero. Arguably
# we may want to allow decimals here, obviously only appropriate for
# translation to ODEs.
array_suffix = "[" + number + "]"
array_suffix.setParseAction(lambda x: x[1])
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

    def used_process_names(self):
        name_queue = self.system_equation.get_used_process_names()
        used_names = set ()
        while name_queue:
            name = name_queue.pop()
            if name not in used_names:
                used_names.add(name)
                definition = [ x for x in self.process_definitions
                                   if x.lhs == name ][0]
                new_names = definition.rhs.get_used_process_names()
                # Do not forget to *not* add the current name to the queue
                # since we have just popped it.
                name_queue.update([ x for x in new_names if x != name])
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

    def get_initial_state(self):
        return self.system_equation.get_initial_state()
    def get_state_builder(self):
        actions_dictionary = self.get_process_actions()
        return self.system_equation.get_state_builder(actions_dictionary)

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
class LeafBuilder(object):
    def __init__(self, actions_dictionary):
        self.leaves = 1
        self.state_dictionary = dict()
        self.actions_dictionary = actions_dictionary
        self.number_of_states = 0
    def get_transitions(self, state):
        actions = self.actions_dictionary[state]
        transitions = [ Transition(a.action, a.rate, a.successor) 
                        for a in actions ]
        self.state_dictionary[state] = StateInfo(self.number_of_states, transitions)
        self.number_of_states += 1
        return transitions

class CoopBuilder(object):
    def __init__(self, lhs, coop_set, rhs):
        self.lhs = lhs
        self.coop_set = coop_set
        self.rhs = rhs
        self.number_of_states = 0
        self.state_dictionary = dict()
        self.leaves = lhs.leaves + rhs.leaves

    def get_transitions(self, state):
        state_information = self.state_dictionary.get(state, None)
        if state_information:
            return state_information.transitions
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
            left_rate = sum([ t.rate for t in left_transitions ])
            right_rate = sum([t.rate for t in right_transitions])
            governing_rate = min(left_rate, right_rate)
            for (left, right) in [ (l, r) for l in left_shared for r in right_shared ]:
                rate = (left.rate / left_rate) * (right.rate / right_rate) * governing_rate
                new_state = (left.successor, right.successor)
                new_transition = Transition(action, rate, new_state)
                transitions.append(new_transition)

        state_number = self.number_of_states
        state_information = StateInfo(state_number, transitions)
        self.state_dictionary[state] = state_information
        self.number_of_states += 1
        return transitions

class ModelSolver(object):
    """A full state space exploring model solver. This solver builds the
       entire state-space of the model and from that derives a CTMC which is
       then solved.
    """
    def __init__(self, model):
        self.model = model

    @property
    def initial_state(self):
        if getattr(self, "_initial_state", None) is None:
            self._initial_state = self.model.get_initial_state()
        return self._initial_state

    @property
    def state_space(self):
        if getattr(self, "_state_space", None) is None:
            self._state_space = self.build_state_space()
        return self._state_space

    @property
    def gen_matrix(self):
        if getattr(self, "_gen_matrix", None) is None:
            self._gen_matrix = self.get_generator_matrix()
        return self._gen_matrix

    @property
    def steady_solution(self):
        if getattr(self, "_steady_solution", None) is None:
            self._steady_solution = self.solve_generator_matrix()
        return self._steady_solution

    @property
    def steady_utilisations(self):
        if getattr(self, "_steady_utilisations", None) is None:
            initial_state = self.model.get_initial_state()
            self._steady_utilisations = self.get_utilisations()
        return self._steady_utilisations

    def build_state_space(self):
        state_builder = self.model.get_state_builder()
        explore_queue = set([self.initial_state])
        explored = set()
        while (explore_queue):
            current_state = explore_queue.pop()
            transitions = state_builder.get_transitions(current_state)
            successor_states = [ t.successor for t in transitions ]
            explored.add(current_state)
            for new_state in successor_states:
                # Note that we should be careful if the new_state is the same as
                # the current state. We won't put it in the explore_queue since
                # the current state should be in explored. However it will mean 
                # we have a self-loop, and we should probably flag that at some
                # point.
                if new_state not in explored and new_state != current_state:
                    explore_queue.add(new_state)
        return state_builder.state_dictionary


    def get_generator_matrix(self):
        # State space is a dictionary which maps a state representation to
        # information about that state. Crucially, the state number and the
        # outgoing transitions. We could possibly store the state number
        # together with the state itself, which would be useful because then the
        # transitions would not need to look up the target states' numbers.
        # This would require the state space build to give a number to each
        # state as it is discovered, which in turn would require that it still
        # stores some set/lookup of the state representation to the state number.
        size = len(self.state_space)
        gen_matrix = numpy.zeros((size, size), dtype=numpy.float64)
        for state_number, transitions in self.state_space.values():
            # For the current state we can obtain the set of transitions.
            # This should be known as we would have done this during state_space
            # exploration hence we can given None as the actions dictionary
            total_out_rate = 0.0
            for transition in transitions:
                target_state = transition.successor
                target_info = self.state_space[target_state]
                target_state_number = target_info.state_number
                # It is += since there may be more than one transition to the same
                # target state from the current state.
                gen_matrix[state_number, target_state_number] += transition.rate
                total_out_rate += transition.rate
            gen_matrix[state_number, state_number] = -total_out_rate
        return gen_matrix

    def solve_generator_matrix(self):
        solution_vector = numpy.zeros(len(self.gen_matrix), dtype=numpy.float64)
        solution_vector[0] = 1
        # This is the normalisation bit
        self.gen_matrix[:,0] = 1
        # Note that here we must transpose the matrix, but arguably we could
        # just build it in the transposed form, since we never use the
        # transposed-form. This would include the above normalisation line.
        result = numpy.linalg.solve(self.gen_matrix.transpose(), solution_vector)
        return result

    def get_utilisations(self):
        def flatten_state(state):
            if isinstance(state, str):
                return [ state ]
            else:
                left, right = state
                return flatten_state(left) + flatten_state(right)
        # The flattening of the initial state here just lets us know how many
        # dictionaries we are going to return.
        dictionaries = [ dict() for x in flatten_state(self.initial_state) ]
        for (state, (state_number, transitions)) in self.state_space.items():
            probability = self.steady_solution[state_number]
            local_states = flatten_state(state)
            for dictionary, process_name in zip(dictionaries, local_states):
                current_probability = dictionary.get(process_name, 0.0)
                dictionary[process_name] = probability + current_probability
        return dictionaries

def analyse_model(model_string):
    model = parse_model(model_string)

    logging.debug (model)
    logging.debug ("Defined process names:")
    for name in defined_process_names(model):
        logging.debug ("    " + name)
    logging.debug ("Referenced process names:")
    for name in used_process_names(model):
        logging.debug ("    " + name)
            
def analyse_pepa_file(filename):
    with open(filename, "r") as pepa_file:
        model_string = pepa_file.read()
        analyse_model(model_string)


def run ():
    """perform the banalities of command-line argument processing 
        and then begin the appropriate command
    """
    description = "The pypepa command-line tool"
    parser = argparse.ArgumentParser(description=description)
    # Probably parse will not ultimately be a command, but is here for testing
    # in the early stages.
    command_choices = ["parse"]
    parser.add_argument('command', metavar='COMMAND', nargs='?', default="parse",
                        choices=command_choices, 
                        help="The command to perform: " + ", ".join(command_choices))
    parser.add_argument('filenames', metavar='F', nargs='+',
                        help="A PEPA file")
    log_choices = [ "info", "warning", "error", "critical", "debug" ]
    parser.add_argument('--loglevel', action='store',
                        choices=log_choices, default='info',
                        help="Set the level of the logger")
    parser.add_argument('--logfile', action='store', default=None,
                        help="The file to output the log to")

    arguments = parser.parse_args()

    # Initialise the logger
    numerical_level = getattr(logging, arguments.loglevel.upper(), None)
    if arguments.logfile:
        logging.basicConfig(filename=arguments.logfile, level=numerical_level)
    else:
        logging.basicConfig(level=numerical_level)
    # We could also change the format of the logging messages to
    # something like: format='%(levelname)s:%(message)s'

    for filename in arguments.filenames:
        if arguments.command == "parse":
            analyse_pepa_file(filename)


if __name__ == "__main__":
    run()
