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
    try:
        if mode == Mode.encode:
            steganography.set_text_to_encode(args.text)
            steganography.set_file_to_encode(args.file)
            additional_output = steganography.encode()
        else:
            if method == Method.echo_single_kernel:
                additional_output = steganography.decode(d0=args.d0, d1=args.d1, l=args.len)

    except OutputFileExists as e:
        error_exit(str(e), ExitCode.OutputFileExists)
    except FileNotFoundError as e:
        error_exit(str(e), ExitCode.FileNotFound)
    except WavReadError as e:
        error_exit(str(e), ExitCode.WavReadError)

    print(json.dumps(additional_output))
