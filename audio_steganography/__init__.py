from .argument_parsing import parse_args
from .methods.method import Method
from .methods.method_base import MethodBase
from .mode import Mode
from .exceptions import OutputFileExists
import sys
import typing
import numpy as np
import scipy.io.wavfile
import os.path

class AudioSteganography:
    def __init__(self,
                 method: Method,
                 mode: Mode,
                 source: str,
                 output_file: typing.Optional[str] = None,
                 overwrite: bool = False):

        self.method = method
        self.mode = mode
        self.source = source
        self.output_file = output_file
        self.overwrite = overwrite

        self.text_to_encode = None
        self.file_to_encode = None

        self.data_to_encode = np.empty(0)


    def encode(self, *args, **kwargs):
        self.check_filename()

        self.prepare_data()

        method: MethodBase = self.method.value(self.data_to_encode, self.mode)
        method.set_cover_data(self.source_data)
        output = method.encode(*args, **kwargs)

        self.write_output(output)


    def decode(self, *args, **kwargs):
        self.check_filename()

        self.prepare_data()

        method: MethodBase = self.method.value(self.source_data, self.mode)
        output = method.decode(*args, **kwargs)

        self.write_output(output)


    def set_text_to_encode(self, text_to_encode: typing.Optional[str]):
        self.text_to_encode = text_to_encode


    def set_file_to_encode(self, file_to_encode: typing.Optional[str]):
        self.file_to_encode = file_to_encode


    def prepare_data(self):
        self.source_sr, self.source_data = scipy.io.wavfile.read(self.source)
        self.source_sr: int = self.source_sr
        self.source_data: np.ndarray[typing.Any, np.dtype[np.int16]] = self.source_data

        if self.mode == Mode.encode:
            if self.text_to_encode is not None:
                self.data_to_encode = np.unpackbits(
                    np.frombuffer(self.text_to_encode.encode('utf8'), np.uint8)
                )
            elif self.file_to_encode is not None:
                self.data_to_encode = np.unpackbits(
                    np.fromfile(self.file_to_encode, np.uint8)
                )


    def write_output(self, output: np.ndarray):
        fname = self.check_filename()

        if self.mode == Mode.encode:
            scipy.io.wavfile.write(fname, self.source_sr, output)
        else:
            with open(fname, 'wb') as f:
                f.write(np.packbits(output).tobytes())


    def check_filename(self) -> str:
        name, ext = os.path.splitext(self.source)
        # Filename when decoding
        fname = f'{name}_{self.method.name}.out'

        # Filename when encoding
        if self.mode == Mode.encode:
            fname = f'{name}_{self.method.name}{ext}'

        # Override filename with the user specified one
        if self.output_file is not None:
            fname = self.output_file

        if os.path.exists(fname) and not self.overwrite:
            raise OutputFileExists('Output file already exists!')

        return fname


def main():
    args, parser = parse_args()
    if args.method is None:
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
        mode = Mode[args.mode]
    except KeyError:
        print(f'{sys.argv[0]}: error: invalid mode specified', file=sys.stderr)
        sys.exit(1)

    steganography = AudioSteganography(
        method,
        mode,
        args.source,
        args.output,
        args.overwrite)

    if mode == Mode.encode:
        steganography.set_text_to_encode(args.text)
        steganography.set_file_to_encode(args.file)

    try:
        if method == Method.echo_single_kernel:
            if mode == Mode.encode:
                steganography.encode()
            else:
                steganography.decode(d0=args.d0, d1=args.d1, l=args.len)

    except OutputFileExists:
        print(f'{sys.argv[0]}: error: output file already exists', file=sys.stderr)
        sys.exit(1)
