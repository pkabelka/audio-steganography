from .methods.method import Method
from .mode import Mode
import argparse
import sys
import typing
import numpy as np
import scipy.io.wavfile

class AudioSteganography:
    def __init__(self,
                 method: Method,
                 mode: Mode,
                 cover_file: str,
                 file_to_encode: typing.Optional[str] = None,
                 text_to_encode: typing.Optional[str] = None,
                 output_file: typing.Optional[str] = None,
                 overwrite: bool = False):
        self.method = method
        self.mode = mode
        self.cover_file = cover_file
        self.file_to_encode = file_to_encode
        self.text_to_encode = text_to_encode
        self.output_file = output_file
        self.overwrite = overwrite
        self.data_to_encode = np.empty(0)
        self.data_to_decode = np.empty(0)

    def encode(self):
        self.prepare_data()
        self.method.value(self.cover_data, self.data_to_encode, self.mode).encode()

    def decode(self):
        self.prepare_data()
        self.method.value(self.cover_data, self.data_to_decode, self.mode).decode()

    def prepare_data(self):
        self.cover_sr, self.cover_data = scipy.io.wavfile.read(self.cover_file)
        self.cover_sr: int = self.cover_sr
        self.cover_data: np.ndarray[typing.Any, np.dtype[np.int16]] = self.cover_data

        encode = []
        if self.text_to_encode is not None:
            for c in self.text_to_encode:
                for b in '{0:08b}'.format(ord(c), 'b'):
                    encode.append(int(b))
            self.data_to_encode = np.array(encode)
        elif self.file_to_encode is not None:
            # TODO: files encoding
            pass

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
        Mode.encode.value,
        parents=[parent_parser],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='List of valid methods:\n  ' + method_list_str,
        help=f'use {Mode.encode.value} -h to show help')

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
        Mode.decode.value,
        parents=[parent_parser],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='List of valid methods:\n  ' + method_list_str,
        help=f'use {Mode.decode.value} -h to show help')

    parser_decode.add_argument(
        '-f',
        '--file', metavar='FILE_TO_DECODE',
        action='store',
        required=True)

    args = parser.parse_args()
    if args.subcommand is None:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # Check if the method is valid
    try:
        method = Method[args.method]
    except KeyError:
        print(f'{sys.argv[0]}: error: invalid method specified', file=sys.stderr)
        sys.exit(1)

    # Check if the mode is valid
    try:
        mode = Mode[args.subcommand]
    except KeyError:
        print(f'{sys.argv[0]}: error: invalid mode specified', file=sys.stderr)
        sys.exit(1)

    steganography = AudioSteganography(
        method,
        mode,
        args.cover,
        args.file,
        args.text,
        args.output,
        args.overwrite)

    if args.subcommand == Mode.encode.value:
        steganography.encode()
    elif args.subcommand == Mode.decode.value:
        steganography.decode()
