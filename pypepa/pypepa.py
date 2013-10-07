"""The main source of the pypepa library"""

import argparse
import logging

import pyparsing
from pyparsing import OneOrMore, Or

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
process_definition = identifier + "=" + process_grammar + ";"

model_grammar = OneOrMore(pyparsing.Group(process_definition))

def parse_model(model_string):
    return model_grammar.parseString(model_string)

def defined_process_names(model):
    """From a parsed model, return the list of defined process names"""
    return [definition[0] for definition in model ]

def used_process_names(model):
    used_names = set()
    for definition in model:
        process = definition[2]
        for name in process.get_used_process_names():
            used_names.add(name)
    return used_names

def parse_file(filename):
    with open(filename, "r") as pepa_file:
        model_string = pepa_file.read()
        model = parse_model(model_string)

        logging.debug (model)
        logging.info ("Defined process names:")
        for name in defined_process_names(model):
            logging.info ("    " + name)
        logging.info ("Referenced process names:")
        for name in used_process_names(model):
            logging.info ("    " + name)

def initialise_logger(arguments):
  """Initialise the logging system, depending on the arguments
     which may set the log level and a log file"""
  log_level = arguments.loglevel
  if log_level:
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
      print ("The log level must be one of the following:")
      print ("    debug, info, warning, error, critical")
      print ("Exiting")
      sys.exit(1) 
  else:
    numeric_level = logging.INFO

  # We could also change the format of the logging messages to
  # something like: format='%(levelname)s:%(message)s'

  log_file = arguments.logfile
  if log_file:
    logging.basicConfig(filename=log_file, level=numeric_level)
  else:
    logging.basicConfig(level=numeric_level)

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
            parse_file(filename)


if __name__ == "__main__":
    run()
