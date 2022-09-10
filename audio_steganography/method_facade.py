from .methods.method_base import MethodBase
from .methods.method import Method
from .mode import Mode
from .exceptions import OutputFileExists
import typing
import numpy as np
import scipy.io.wavfile
import os.path

class MethodFacade:
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


    def encode(self, *args, **kwargs) -> typing.Dict[str, typing.Any]:
        self.check_filename()

        self.prepare_data()

        method: MethodBase = self.method.value(self.source_data)
        method.set_secret_data(self.data_to_encode)
        output, additional_output = method.encode(*args, **kwargs)

        self.write_output(output)
        return additional_output


    def decode(self, *args, **kwargs) -> typing.Dict[str, typing.Any]:
        self.check_filename()

        self.prepare_data()

        method: MethodBase = self.method.value(self.source_data)
        output, additional_output = method.decode(*args, **kwargs)

        self.write_output(output)
        return additional_output


    def set_text_to_encode(self, text_to_encode: typing.Optional[str]):
        self.text_to_encode = text_to_encode


    def set_file_to_encode(self, file_to_encode: typing.Optional[str]):
        self.file_to_encode = file_to_encode


    def prepare_data(self):
        try:
            self.source_sr, self.source_data = scipy.io.wavfile.read(self.source)
        except FileNotFoundError:
            raise FileNotFoundError('source file not found')
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

        # -o - works only in decode mode
        if self.mode == Mode.encode:
            scipy.io.wavfile.write(fname, self.source_sr, output)
        else:
            bytes = np.packbits(output).tobytes()
            if self.output_file == '-':
                print(bytes)
            else:
                with open(fname, 'wb') as f:
                    f.write(bytes)


    def check_filename(self) -> str:
        if self.output_file == '-':
            return self.output_file

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
            raise OutputFileExists('output file already exists')

        return fname
