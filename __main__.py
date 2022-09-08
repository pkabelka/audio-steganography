#!/usr/bin/env python3

import argparse
from methods.method import Method
import audio_steganography
import sys

def main():
    methods_str = [m.name for m in Method]
    method_list_str = '\n  '.join(methods_str)

    # Parse arguments
    parent_parser = argparse.ArgumentParser(add_help=False)

    parent_parser.add_argument(
        '-m',
        '--method',
        metavar='METHOD',
        action='store',
        help='steganography method to use',
        required=True,
        choices=methods_str)

    parent_parser.add_argument(
        '-o',
        '--output',
        metavar='OUTPUT_FILE',
        action='store',
        default=None)

    parent_parser.add_argument(
        '-y',
        '--overwrite',
        action='store_true',
        help='overwrite existing OUTPUT_FILE',
        default=False)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand')

    parser_encode = subparsers.add_parser(
        audio_steganography.Mode.encode.value,
        parents=[parent_parser],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='List of valid methods:\n  ' + method_list_str,
        help=f'use {audio_steganography.Mode.encode.value} -h to show help')

    parser_encode.add_argument(
        '-c',
        '--cover',
        metavar='COVER_FILE',
        action='store',
        help='cover audio file',
        required=True)

    group = parser_encode.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-f',
        '--file', metavar='FILE_TO_ENCODE',
        action='store',
        help='file to encode in COVER_FILE',
        default=None)

    group.add_argument(
        '-t',
        '--text',
        metavar='TEXT_TO_ENCODE',
        action='store',
        help='text to encode in COVER_FILE',
        default=None)

    parser_decode = subparsers.add_parser(
        audio_steganography.Mode.decode.value,
        parents=[parent_parser],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='List of valid methods:\n  ' + method_list_str,
        help=f'use {audio_steganography.Mode.decode.value} -h to show help')

    parser_decode.add_argument(
        '-f',
        '--file', metavar='FILE_TO_DECODE',
        action='store',
        required=True)

    args = parser.parse_args()

    # Check if the method is valid
    try:
        method = Method[args.method]
    except KeyError:
        print(f'{sys.argv[0]}: error: invalid method specified', file=sys.stderr)
        sys.exit(1)

    # Check if the mode is valid
    try:
        mode = audio_steganography.Mode[args.subcommand]
    except KeyError:
        print(f'{sys.argv[0]}: error: invalid mode specified', file=sys.stderr)
        sys.exit(1)

    steganography = audio_steganography.AudioSteganography(
        method,
        mode,
        args.cover,
        args.file,
        args.text,
        args.output,
        args.overwrite)

    if args.subcommand == audio_steganography.Mode.encode.value:
        steganography.encode()
    elif args.subcommand == audio_steganography.Mode.decode.value:
        steganography.decode()

if __name__ == '__main__':
    main()
