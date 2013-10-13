"""The main source of the pypepa library"""

import argparse
import logging

from collections import namedtuple

import pyparsing
from pyparsing import OneOrMore, Or, Group, Optional

Action = namedtuple('Action', ["action", "rate", "successor" ])

identifier = pyparsing.Word(pyparsing.alphanums)

expr = identifier

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

    def get_successors(self):
        return [ str(self.successor) ]
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

    def get_successors(self):
        lhs = self.lhs.get_successors()
        rhs = self.rhs.get_successors()
        return lhs + rhs

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
        return LocalState(self.identifier)
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
        return CoopState(self.lhs.get_initial_state(),
                         self.cooperation_set,
                         self.rhs.get_initial_state())

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

    def get_successors(self):
        successors_dictionary = dict()
        for definition in self.process_definitions:
            successors = definition.rhs.get_successors()
            successors_dictionary[definition.lhs] = successors
        return successors_dictionary

    def get_initial_state(self):
        return self.system_equation.get_initial_state()

# Note, this parser does not insist on the end of the input text. Which means
# in theory you could have something *after* the model text, which might indeed
# be what you are wishing for. See parse_model for a whole input parser
model_grammar = process_definitions_grammar + system_equation_grammar
model_grammar.setParseAction(ParsedModel)

def parse_model(model_string):
    # Parses a model and also ensures that we have consumed the entire input
    whole_input_parser = model_grammar + pyparsing.StringEnd()
    return whole_input_parser.parseString(model_string)[0]

def defined_process_names(model):
    """From a parsed model, return the list of defined process names"""
    return set([definition.lhs for definition in model.process_definitions ])

Transition = namedtuple('Transition', ["action", "rate", "successor"])
class LocalState(object):
    def __init__(self, identifier):
        self.identifier = identifier
        self.local_states = [ identifier ]
    def get_transitions(self, actions_dictionary):
        return [ Transition(t.action, t.rate, LocalState(t.successor))
                    for t in actions_dictionary[self.identifier] ]

class CoopState(object):
    def __init__(self, lhs, coop_set, rhs):
        self.lhs = lhs
        self.rhs = rhs
        self.coop_set = coop_set
        self.local_states = lhs.local_states + rhs.local_states

    def get_transitions(self, actions_dictionary):
        # TODO: This isn't doing the right thing if the action is in the
        # cooperation set.
        transitions = []
        left_actions  = self.lhs.get_transitions(actions_dictionary)
        right_actions = self.rhs.get_transitions(actions_dictionary)
        for transition in left_transitions:
            if transition.action not in self.coop_set:
                new_state = CoopState(transition.successor, 
                                      self.coop_set, self.rhs)
                new_transition = Transition(transition.action, transition.rate, 
                                            new_state)
                transitions.append(new_transition)

    


class PepaStateSpace(object):
    def __init__(self, model):
        self.model = model

    def build_state_space(self):
        initial_state = model.get_initial_state()
        explore_queue = set([initial_state])
        explored = set([])
        while (explore_queue):
            current_state = explore_queue.pop()
            successor_states = current_state.get_successor_states()
            for new_state in successor_states:
                if new_state != current_state and new_state not in explored:
                    explore_queue.add(new_state)
        return explored
            


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
