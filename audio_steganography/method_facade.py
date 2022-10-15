# -*- coding: utf-8 -*-

# File: method_facade.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the MethodFacade which manages input, output, encoding
and decoding of files.
"""

from .methods.method_base import MethodBase
from .methods.method import Method
from .mode import Mode
from .exceptions import OutputFileExists, WavReadError
from .audio_utils import to_dtype
import typing
import numpy as np
import scipy.io.wavfile
import os.path

class MethodFacade:
    """This class prepares the input data, encodes/decodes them with the
    specified Method and writes the output to the specified file or prints it
    to STDOUT.
    """
    def __init__(
            self,
            method: Method,
            mode: Mode,
            source: str,
            output_file: typing.Optional[str] = None,
            overwrite: bool = False
        ):

        self.method = method
        self.mode = mode
        self.source = source
        self.output_file = output_file
        self.overwrite = overwrite

        self.text_to_encode = None
        self.file_to_encode = None

        self.data_to_encode = np.empty(0, dtype=np.uint8)


    def encode(self, *args, **kwargs) -> typing.Dict[str, typing.Any]:
        """This function encodes the secret data into the source using the
        specified Method.
        """
        self.check_filename()

        self.prepare_data()

        method: MethodBase = self.method.value(self.source_data, self.data_to_encode)
        output, additional_output = method.encode(*args, **kwargs)

        self.write_output(output)
        return additional_output


    def decode(self, *args, **kwargs) -> typing.Dict[str, typing.Any]:
        """This function decodes the secret data from the source using the
        specified Method.
        """
        self.check_filename()

        self.prepare_data()

        method: MethodBase = self.method.value(self.source_data)
        output, additional_output = method.decode(*args, **kwargs)

        self.write_output(output)
        return additional_output


    def set_text_to_encode(self, text_to_encode: typing.Optional[str]):
        """Setter for the text to be encoded.

        Parameters
        ----------
        text_to_encode : str | None
            String to encode. If None then `set_file_to_encode` should be
            used.
        """
        self.text_to_encode = text_to_encode


    def set_file_to_encode(self, file_to_encode: typing.Optional[str]):
        """Setter for the file to be encoded.

        Parameters
        ----------
        file_to_encode : str | None
            Path to the file to encode. If None then `set_text_to_encode`
            should be used.
        """
        self.file_to_encode = file_to_encode


    def prepare_data(self):
        """Reads the source file into self.source_data and self.source_sr.
        The source data is normalized to float64 [-1, 1]. The secret data is
        read and converted to uint8 bit array.
        """
        try:
            self.source_sr, self.source_data = scipy.io.wavfile.read(self.source)
        except FileNotFoundError:
            raise FileNotFoundError('source file not found')
        except ValueError as e:
            raise WavReadError(str(e))

        # TODO: option to use both tracks
        # TODO: output same number of tracks as input
        if self.source_data.ndim > 1:
            self.source_data = self.source_data[:, 0]

        self._source_dtype = self.source_data.dtype
        self.source_sr: int = self.source_sr

        # normalize to float64 [-1; 1]
        self.source_data: np.ndarray[
            typing.Any,
            np.dtype[np.float64]] = (self.source_data /
                np.abs(self.source_data).max()).astype(np.float64)

        if self.mode == Mode.encode:

            if self.text_to_encode is not None:
                self.data_to_encode = np.unpackbits(
                    np.frombuffer(self.text_to_encode.encode('utf8'), np.uint8))

            elif self.file_to_encode is not None:
                self.data_to_encode = np.unpackbits(
                    np.fromfile(self.file_to_encode, np.uint8))


    def write_output(self, output: np.ndarray):
        """Writes the output NumPy array to a file or STDOUT.

        If the specified file name is `-` and decode mode is used, prints the
        output to STDOUT, otherwise in encode mode, writes the output to
        a file.

        Parameters
        ----------
        output : numpy.ndarray
            Array containing the output of encode/decode methods.
        """
        fname = self.check_filename()

        if self.mode == Mode.encode:
            # center and normalize range to the original dtype
            output = output - np.mean(output)
            output = output / np.abs(output).max()

            if self._source_dtype in [np.uint8, np.int16, np.int32]:
                output = to_dtype(output, self._source_dtype)

            scipy.io.wavfile.write(fname, self.source_sr, output)
        else:
            bytes = np.packbits(output).tobytes()
            # -o - works only in decode mode
            if self.output_file == '-':
                print(bytes)
            else:
                with open(fname, 'wb') as f:
                    f.write(bytes)


    def check_filename(self) -> str:
        """Creates the output file name according to the used method or uses
        the user specified file name and checks if the file already exists.

        If the file already exists and `self.overwrite` is not used, raises
        `OutputFileExists` exception.

        Returns
        -------
        fname : str
            New file name.
        """
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
