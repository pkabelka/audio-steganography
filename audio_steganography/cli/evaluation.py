#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File: evaluation.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This file contains the main() function for the audio steganography method
evaluation program.
"""

from ..methods import MethodEnum
from ..methods.method_base import MethodBase
from .method_facade import MethodFacade
from .mode import Mode
from ..exceptions import OutputFileExists, WavReadError, SecretSizeTooLarge
from .cli_utils import error_exit, get_attr
from .exit_codes import ExitCode
import argparse
from typing import Tuple, List, Any
import sys
import json

def parse_args() -> Tuple[Any, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-w',
        '--workdir', metavar='PATH',
        action='store',
        help='path to a directory containing datasets',
        required=True)

    parser.add_argument(
        '-o',
        '--output',
        metavar='OUTPUT_FILE',
        action='store',
        help='path to an output file; - outputs to STDOUT',
        default=None)

    parser.add_argument(
        '-y',
        '--overwrite',
        action='store_true',
        help='overwrite existing OUTPUT_FILE',
        default=False)

    parser.add_argument(
        'methods',
        action='store',
        choices=['ALL'] + [method.name for method in MethodEnum],
        nargs='*',
        help='method(s) to evaluate; default: ALL',
        default='ALL')

    # Parse all args
    args = parser.parse_args()
    return args, parser

def main():
    """The main function of the evaluation program.

    The function first parses user arguments and checks if they are valid.
    Output is written to a file or printed to STDOUT.
    """
    args, _ = parse_args()
    print(args, file=sys.stderr)

    # get the requested methods from args
    methods = [method.value for method in MethodEnum]
    if isinstance(args.methods, list) and 'ALL' not in args.methods:
        methods = []
        for method in args.methods:
            # Check if the method is valid
            try:
                methods.append(MethodEnum[method])
            except KeyError:
                error_exit('invalid method specified', ExitCode.InvalidMethod)
    print(methods, file=sys.stderr)

if __name__ == '__main__':
    main()
