"""The main source of the pypepa library"""

import argparse
import pyparsing

identifier = pyparsing.Word(pyparsing.alphanums)

expr = identifier

rate_grammar = expr

prefix_grammar  = "(" + identifier + "," + rate_grammar + ")" + "." + identifier
process_grammar = prefix_grammar
process_definition = identifier + "=" + process_grammar + ";"


def parse_model(model_string):
    return process_definition.parseString(model_string)

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
