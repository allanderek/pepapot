"""The main source of the pypepa library"""

import argparse
import logging

import pyparsing
from pyparsing import OneOrMore, Or, Group, Optional

identifier = pyparsing.Word(pyparsing.alphanums)

expr = identifier

rate_grammar = expr

process_grammar = pyparsing.Forward()
class PrefixNode(object):
    def __init__(self, tokens):
        self.action = tokens[1]
        self.rate = tokens[3]
        self.successor = tokens[6]

    def get_used_process_names(self):
        return self.successor.get_used_process_names()
prefix_grammar  = "(" + identifier + "," + rate_grammar + ")" + "." + process_grammar
prefix_grammar.setParseAction(PrefixNode)

class ProcessIdentifier(object):
    def __init__(self, tokens):
        self.name = tokens[0]

    def get_used_process_names(self):
        return [ self.name ]
process_identifier = identifier.copy()
process_identifier.setParseAction(ProcessIdentifier)

process_grammar << Or([prefix_grammar, process_identifier])
class ProcessDefinition(object):
    def __init__(self, tokens):
        self.lhs = tokens[0]
        self.rhs = tokens[2]
process_definition_grammar = identifier + "=" + process_grammar + ";"
process_definition_grammar.setParseAction(ProcessDefinition)
process_definitions_grammar = Group(OneOrMore(process_definition_grammar))


system_equation_grammar = pyparsing.Forward()
class ParsedSystemComponent(object):
    def __init__(self, tokens):
        if len(tokens) > 1:
            # Assuming the grammar below of "identifer + Optional (...)
            # Then the left hand side will always be a simple identifier, but
            # this won't be true if we allow for parentheses.
            self.lhs = ParsedSystemComponent(tokens[0])
            self.rhs = tokens[2]
            self.identifier = None
        else:
            self.lhs = None
            self.rhs = None
            self.identifier = tokens[0]

    def get_used_names(self):
        if self.identifier:
            return set(self.identifier)
        else:
            lhs = self.lhs.get_used_names()
            rhs = self.rhs.get_used_names()
            return lhs.union(rhs)


system_equation_grammar << identifier + Optional("||" + system_equation_grammar)
system_equation_grammar.setParseAction(ParsedSystemComponent)

class ParsedModel(object):
    def __init__(self, tokens):
        self.process_definitions = tokens[0]
        self.system_equation = tokens[1]

    def used_process_names(self):
        name_queue = self.system_equation.get_used_names()
        used_names = set ()
        while name_queue:
            name = name_queue.pop()
            if name not in used_names:
                used_names.add(name)
                definition = [ x for x in self.process_definitions
                                   if x.lhs == name ][0]
                name_queue.update(definition.rhs.get_used_process_names())
        return used_names
model_grammar = process_definitions_grammar + system_equation_grammar
model_grammar.setParseAction(ParsedModel)

def parse_model(model_string):
    return model_grammar.parseString(model_string)[0]

def defined_process_names(model):
    """From a parsed model, return the list of defined process names"""
    return set([definition.lhs for definition in model.process_definitions ])

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
