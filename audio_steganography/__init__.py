# -*- coding: utf-8 -*-

# File: __init__.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the main() function, parses arguments and uses
MethodsFacade class to run encoding and decoding.
"""

from .argument_parsing import parse_args
from .methods.method import Method
from .method_facade import MethodFacade
from .mode import Mode
from .exceptions import OutputFileExists, WavReadError
from .utils import error_exit
from .exit_codes import ExitCode
import sys
import json

def main():
    """The main function of the program.

    The function first parses user arguments and checks if they are valid.
    Then the MethodFacade class is used encode/decode using the specified
    method. Known exceptions are caught and presented as errors and program
    exits with an appropriate exit code.

    Any additional output arising from encode/decode functions is printed to
    STDOUT in JSON format.
    """
    args, parser = parse_args()
    if args.method is None:
        parser.print_help(sys.stderr)
        sys.exit(ExitCode.Ok.value)

    # Check if the method is valid
    try:
        method = Method[args.method]
    except KeyError:
        error_exit('invalid method specified', ExitCode.InvalidMethod)

    # Check if the mode is valid
    try:
        mode = Mode[args.mode]
    except KeyError:
        error_exit('invalid mode specified', ExitCode.InvalidMode)

    steganography = MethodFacade(
        method,
        mode,
        args.source,
        args.output,
        args.overwrite)

    additional_output = {}
    options = {}
    try:
        if mode == Mode.encode:
            steganography.set_text_to_encode(args.text)
            steganography.set_file_to_encode(args.file)
            additional_output = steganography.encode(**options)
        else:
            if method == Method.echo_single_kernel:
                options = {
                    'd0': args.d0,
                    'd1': args.d1,
                    'l': args.len,
                }
            additional_output = steganography.decode(**options)

    except OutputFileExists as e:
        error_exit(str(e), ExitCode.OutputFileExists)
    except FileNotFoundError as e:
        error_exit(str(e), ExitCode.FileNotFound)
    except WavReadError as e:
        error_exit(str(e), ExitCode.WavReadError)

    print(json.dumps(additional_output))
