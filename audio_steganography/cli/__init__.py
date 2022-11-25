# -*- coding: utf-8 -*-

# File: __init__.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the main() function, parses arguments and uses
MethodsFacade class to run encoding and decoding.
"""

from .argument_parsing import parse_args
from ..methods.method import Method
from .method_facade import MethodFacade
from .mode import Mode
from ..exceptions import OutputFileExists, WavReadError, SecretSizeTooLarge
from .utils import error_exit, get_attr
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
        args.overwrite,
    )

    additional_output = {}
    options = {
        Method.lsb: {
            (a:='depth'): get_attr(args, a),
            'l': get_attr(args, 'len'),
        },
        Method.echo_single_kernel: {
            (a:='d0'): get_attr(args, a),
            (a:='d1'): get_attr(args, a),
            (a:='delay_search'): get_attr(args, a),
            'l': get_attr(args, 'len'),
        },
    }

    try:
        if mode == Mode.encode:
            steganography.set_text_to_encode(args.text)
            steganography.set_file_to_encode(args.file)
            output, additional_output = steganography.encode(**options[method])

            stats = {}
            if args.stats:
                stats = steganography.get_stats(output, additional_output)
            additional_output = {**additional_output, **stats}
        else:
            output, additional_output = steganography.decode(**options[method])

    except OutputFileExists as e:
        error_exit(str(e), ExitCode.OutputFileExists)
    except FileNotFoundError as e:
        error_exit(str(e), ExitCode.FileNotFound)
    except WavReadError as e:
        error_exit(str(e), ExitCode.WavReadError)
    except SecretSizeTooLarge as e:
        error_exit(str(e), ExitCode.SecretSizeTooLarge)

    steganography.write_output(output)

    print(json.dumps(additional_output))