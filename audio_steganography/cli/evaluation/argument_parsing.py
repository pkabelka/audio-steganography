# -*- coding: utf-8 -*-

# File: argument_parsing.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This file contains the function for parsing arguments of the evaluation
program.
"""

from ...methods import MethodEnum
import argparse
from typing import Tuple, Any

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
        '-l',
        '--log',
        action='store_true',
        help='enables information logging',
        default=False)

    parser.add_argument(
        '-e',
        '--extended',
        action='store_true',
        help='enables extended testing',
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
