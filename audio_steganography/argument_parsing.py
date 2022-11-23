# -*- coding: utf-8 -*-

# File: argument_parsing.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the parse_args function.
"""

from .methods.method import Method
from .mode import Mode
import argparse
from typing import Tuple, Any

def parse_args() -> Tuple[Any, argparse.ArgumentParser]:
    """This function parses the arguments when ran from the command line.

    The argparse package is used for parsing. The common arguments are added to
    the parser first. Then are added arguments specific to the steganography
    methods by calling their `get_encode_args` and `get_decode_args` methods.

    Lastly the function parses the arguments and returns the parsed object and
    the argument parser.

    Returns
    -------
    args : Namespace
        Parsed arguments object.
    parser : argparse.ArgumentParser
        The argument parser object.
    """

    # Add common args
    parent_parser = argparse.ArgumentParser(add_help=False)

    parent_parser.add_argument(
        '-s',
        '--source', metavar='SOURCE',
        action='store',
        help='combined with encode, specify cover file; combined with '+
            'decode, specify file to decode',
        required=True)

    parent_parser.add_argument(
        '-o',
        '--output',
        metavar='OUTPUT_FILE',
        action='store',
        help='without -o, -e produces a file with _METHOD appended and -d '+
            'produces SOURCE.out; -o - outputs bytes to STDOUT only in '+
            'decode mode',
        default=None)

    parent_parser.add_argument(
        '-y',
        '--overwrite',
        action='store_true',
        help='overwrite existing OUTPUT_FILE',
        default=False)

    # Add methods
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='method')

    for m in Method:
        method_parser = subparsers.add_parser(
            m.name,
            help=f'use {m.name} -h to show help')

        # Add encode and decode subcommands for each method
        method_mode_subparsers = method_parser.add_subparsers(dest='mode')

        # Add arguments specific to encode subcommand
        encode_subparser = method_mode_subparsers.add_parser(
            Mode.encode.value,
            parents=[parent_parser],
            help='')

        group = encode_subparser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            '-f',
            '--file', metavar='FILE_TO_ENCODE',
            action='store',
            help='file to encode in SOURCE',
            default=None)

        group.add_argument(
            '-t',
            '--text',
            metavar='TEXT_TO_ENCODE',
            action='store',
            help='text to encode in SOURCE',
            default=None)

        encode_subparser.add_argument(
            '--stats',
            action='store_true',
            help='output results of statistical tests after encoding',
            default=False)

        args = m.value.get_encode_args()
        for arg in args:
            encode_subparser.add_argument(*arg[0], **arg[1])

        # Add arguments specific to decode subcommand
        decode_subparser = method_mode_subparsers.add_parser(
            Mode.decode.value,
            parents=[parent_parser],
            help='')

        args = m.value.get_decode_args()
        for arg in args:
            decode_subparser.add_argument(*arg[0], **arg[1])

    # Parse all args
    args = parser.parse_args()
    return args, parser
