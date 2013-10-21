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

system_equation_grammar = pyparsing.Forward()
class ParsedNamedComponent(object):
    def __init__(self, tokens):
        self.identifier = tokens[0]
    def get_used_process_names(self):
        return set(self.identifier)
    def get_initial_state(self):
        return self.identifier
    def get_state_builder(self, actions_dictionary):
        return LeafBuilder(actions_dictionary)

class ParsedSystemCooperation(object):
    def __init__(self, tokens):
        # Assuming the grammar below of "identifer + Optional (...)
        # Then the left hand side will always be a simple identifier, but
        # this won't be true if we allow for parentheses.
        self.lhs = ParsedNamedComponent(tokens[0])
        self.cooperation_set = tokens[1]
        self.rhs = tokens[2]

    def get_used_process_names(self):
        lhs = self.lhs.get_used_process_names()
        rhs = self.rhs.get_used_process_names()
        return lhs.union(rhs)

    def get_initial_state(self):
        return (self.lhs.get_initial_state(), self.rhs.get_initial_state())
    def get_state_builder(self, actions_dictionary):
        return CoopBuilder(self.lhs.get_state_builder(actions_dictionary),
                           self.cooperation_set,
                           self.rhs.get_state_builder(actions_dictionary))

def create_system_component(tokens):
    if len(tokens) > 1:
        return ParsedSystemCooperation(tokens)
    else:
        return ParsedNamedComponent(tokens)
system_equation_grammar << (identifier + 
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
        self.state_dictionary[state] = (self.number_of_states, transitions)
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
            return state_information
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
        for l_trans in left_transitions:
            if l_trans.action in self.coop_set:
                for r_trans in right_transitions:
                    if r_trans.action == l_trans.action:
                        # TODO: Clearly the rate here is incorrect
                        new_state = (l_trans.successor, r_trans.successor)
                        new_transition = Transition(l_trans.action, l_trans.rate, new_state)
                        transitions.append(new_transition)

        state_information = (self.number_of_states, transitions)
        self.state_dictionary[state] = state_information
        self.number_of_states += 1
        return transitions

def build_state_space(model):
    state_builder = model.get_state_builder()
    explore_queue = set([model.get_initial_state()])
    explored = set()
    limit = 10
    while (explore_queue and limit):
        limit -= 1
        current_state = explore_queue.pop()
        transitions = state_builder.get_transitions(current_state)
        successor_states = [ t.successor for t in transitions ]
        explored.add(current_state)
        for new_state in successor_states:
            # Note that we should be careful if the new_state is the same as
            # the current state. We won't put it in the explore_queue since
            # the current state should be in explored. However it will mean we
            # have a self-loop, and we should probably flag that at some point.
            if new_state not in explored and new_state != current_state:
                explore_queue.add(new_state)
    return state_builder.state_dictionary


def get_generator_matrix(state_space):
    # State space is a dictionary which maps a state representation to
    # information about that state. Crucially, the state number and the outgoing
    # transitions. We could possibly store the state number together with the
    # state itself, which would be useful because then the transitions would not
    # need to look up the target states' numbers. This would require the
    # state space build to give a number to each state as it is discovered,
    # which in turn would require that it still stores some set/lookup of the
    # state representation to the state number.

    size = len(state_space)
    gen_matrix = numpy.zeros((size, size), dtype=numpy.float64)
    for state_number, transitions in state_space.values():
        # For the current state we can obtain the set of transitions.
        # This should be known as we would have done this during state_space
        # exploration hence we can given None as the actions dictionary
        total_out_rate = 0.0
        for transition in transitions:
            target_state = transition.successor
            target_state_number = (state_space[target_state])[0]
            # It is += since there may be more than one transition to the same
            # target state from the current state.
            gen_matrix[state_number, target_state_number] += transition.rate
            total_out_rate += transition.rate
        gen_matrix[state_number, state_number] = -total_out_rate
    return gen_matrix

def solve_generator_matrix(gen_matrix):
    b = numpy.zeros(len(gen_matrix), dtype=numpy.float64)
    b[0] = 1
    # This is the normalisation bit
    gen_matrix[:,0] = 1
    result = numpy.linalg.solve(gen_matrix.transpose(),b)
    print(result)
    return result

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
