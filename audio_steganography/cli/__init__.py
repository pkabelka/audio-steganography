# -*- coding: utf-8 -*-

# File: __init__.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the main() function, parses arguments and uses
MethodsFacade class to run encoding and decoding.
"""

from .argument_parsing import parse_args
from ..methods import MethodEnum
from .method_facade import MethodFacade
from .mode import Mode
from ..exceptions import SecretSizeTooLarge
from .cli_utils import error_exit, get_attr
from .exit_codes import ExitCode
import sys
import json
import scipy.io.wavfile
import os.path
import numpy as np

def check_filename(source_name, output_file, method) -> str:
    """Creates the output file name according to the used method or uses
    the user specified file name and checks if the file already exists.

    If the file already exists and `self.overwrite` is not used, raises
    `OutputFileExists` exception.

    Returns
    -------
    fname : str
        New file name.
    """
    if output_file == '-':
        return output_file

    name, ext = os.path.splitext(source_name)
    # Filename when decoding
    fname = f'{name}_{method.name}.out'

    # Filename when encoding
    if mode == Mode.encode:
        fname = f'{name}_{method.name}{ext}'

    # Override filename with the user specified one
    if output_file is not None:
        fname = output_file

    return fname

def prepare_secret_data(text_to_encode: str, file_to_encode: str) -> np.ndarray:
    """Converts the input text or file to uint8 bit array.

    Returns
    -------
    out : NDArray[UINT8]
        New file name.
    """
    if text_to_encode is not None:
        return np.unpackbits(np.frombuffer(text_to_encode.encode('utf8'), np.uint8))

    elif file_to_encode is not None:
        return np.unpackbits(np.fromfile(file_to_encode, np.uint8))

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
        method = MethodEnum[args.method]
    except KeyError:
        error_exit('invalid method specified', ExitCode.InvalidMethod)

    # Check if the mode is valid
    try:
        mode = Mode[args.mode]
    except KeyError:
        error_exit('invalid mode specified', ExitCode.InvalidMode)

    # Read source WAV data
    try:
        source_sr, source_data = scipy.io.wavfile.read(args.source)
    except FileNotFoundError:
        error_exit('source file not found', ExitCode.FileNotFound)
    except ValueError as e:
        error_exit(str(e), ExitCode.WavReadError)

    # Check if the output exists
    output_file = check_filename(args.source, args.output, method)
    if os.path.exists(output_file) and not args.overwrite:
        error_exit('output file already exists', ExitCode.OutputFileExists)

    # Create facade for encoding/decoding
    steganography = MethodFacade(
        method,
        mode,
        source_data,
    )

    additional_output = {}
    options = {
        MethodEnum.lsb: {
            (a:='depth'): get_attr(args, a),
            (a:='only_needed'): get_attr(args, a),
            'l': get_attr(args, 'len'),
        },
        **dict.fromkeys(
            [
                MethodEnum.echo_single,
                MethodEnum.echo_bipolar,
                MethodEnum.echo_bf,
                MethodEnum.echo_bipolar_bf,
            ],
            {
                (a:='d0'): get_attr(args, a),
                (a:='d1'): get_attr(args, a),
                (a:='alpha'): get_attr(args, a),
                (a:='decay_rate'): get_attr(args, a),
                (a:='delay_search'): get_attr(args, a),
                'l': get_attr(args, 'len'),
            },
        ),
        MethodEnum.phase: {
            'l': get_attr(args, 'len'),
        },
        MethodEnum.dsss: {
            'l': get_attr(args, 'len'),
            'password': get_attr(args, 'password'),
            'alpha': get_attr(args, 'alpha'),
        },
        MethodEnum.silence_interval: {
            'l': get_attr(args, 'len'),
            (a:='min_silence_len'): get_attr(args, a),
        },
        MethodEnum.dsss_dft: {
            'l': get_attr(args, 'len'),
            'password': get_attr(args, 'password'),
            'alpha': get_attr(args, 'alpha'),
        },
        MethodEnum.tone_insertion: {
            'l': get_attr(args, 'len'),
            (a:='f0'): get_attr(args, a),
            (a:='f1'): get_attr(args, a),
        },
    }

    # Run encode/decode
    try:
        if mode == Mode.encode:
            steganography.data_to_encode = prepare_secret_data(args.text, args.file)
            output, additional_output = steganography.encode(**options.get(method, {}))

            stats = {}
            if args.stats:
                stats = steganography.get_stats(output, additional_output)
            additional_output = {**additional_output, **stats}
        else:
            output, additional_output = steganography.decode(**options.get(method, {}))

    except SecretSizeTooLarge as e:
        error_exit(str(e), ExitCode.SecretSizeTooLarge)

    # Write output
    if mode == Mode.encode:
        scipy.io.wavfile.write(output_file, source_sr, output)
    else:
        bytes_ = np.packbits(output).tobytes()
        # -o - works only in decode mode
        if output_file == '-':
            print(bytes_)
        else:
            with open(output_file, 'wb') as f:
                f.write(bytes_)

    print(json.dumps(additional_output))
