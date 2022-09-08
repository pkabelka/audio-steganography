import typing
from enum import Enum
from methods.method import Method
import numpy as np
import scipy.io.wavfile

class Mode(Enum):
    encode = 'encode'
    decode = 'decode'

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
        self.cover_file = cover_file
        self.file_to_encode = file_to_encode
        self.text_to_encode = text_to_encode
        self.output_file = output_file
        self.overwrite = overwrite
        self.data_to_encode = np.empty(0)

    def encode(self):
        self.prepare_data()
        self.method.value(self.cover_data, self.data_to_encode).encode()

    def decode(self):
        self.prepare_data()
        self.method.value(self.cover_data, self.data_to_encode).decode()

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
