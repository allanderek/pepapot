"""The main source of the pypepa library"""

import argparse
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
prefix_grammar  = "(" + identifier + "," + rate_grammar + ")" + "." + process_grammar
prefix_grammar.setParseAction(PrefixNode)

process_grammar << Or([prefix_grammar, identifier])
process_definition = identifier + "=" + process_grammar + ";"

model_grammar = OneOrMore(pyparsing.Group(process_definition))

def parse_model(model_string):
    return model_grammar.parseString(model_string)

def parse_file(filename):
    with open(filename, "r") as pepa_file:
        model_string = pepa_file.read()
        parse_results = parse_model(model_string)
        print (parse_results)

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

    arguments = parser.parse_args()
    for filename in arguments.filenames:
        if arguments.command == "parse":
            parse_file(filename)


if __name__ == "__main__":
    run()
