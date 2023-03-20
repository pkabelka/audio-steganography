# -*- coding: utf-8 -*-

# File: method_facade.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the MethodFacade which manages input, output, encoding
and decoding of files.
"""

from ..methods.method_base import MethodBase, EncodeDecodeReturn
from ..methods import MethodEnum
from .mode import Mode
from ..stat_utils import snr_db, mse, rmsd, psnr_db, ber_percent
from typing import Optional, Dict, Any
import numpy as np

class MethodFacade:
    """This class prepares the input data, encodes/decodes them with the
    specified Method and writes the output to the specified file or prints it
    to STDOUT.
    """
    def __init__(
            self,
            method: MethodEnum,
            mode: Mode,
            source: np.ndarray,
        ):

        self.method = method
        self.mode = mode
        self.source_data = source

        self.data_to_encode = np.empty(0, dtype=np.uint8)

    def encode(self, *args, **kwargs) -> EncodeDecodeReturn:
        """This function encodes the secret data into the source using the
        specified Method.
        """
        self.prepare_source_data()

        method: MethodBase = self.method.value(self.source_data, self.data_to_encode)
        output, additional_output = method.encode(*args, **kwargs)

        return output, additional_output

    def decode(self, *args, **kwargs) -> EncodeDecodeReturn:
        """This function decodes the secret data from the source using the
        specified Method.
        """
        self.prepare_source_data()

        method: MethodBase = self.method.value(self.source_data)
        output, additional_output = method.decode(*args, **kwargs)

        return output, additional_output

    def prepare_source_data(self):
        """Reads the source file into self.source_data and self.source_sr.
        The source data is normalized to float64 [-1, 1]. The secret data is
        read and converted to uint8 bit array.
        """
        # TODO: option to use both tracks
        # TODO: output same number of tracks as input
        if self.source_data.ndim > 1:
            self.source_data = self.source_data[:, 0]

        self._source_dtype = self.source_data.dtype

        # normalize to float64 [-1; 1]
        # FIXME: this breaks LSB method
        # self.source_data: np.ndarray[
        #     Any,
        #     np.dtype[np.float64]] = (self.source_data /
        #         np.abs(self.source_data).max()).astype(np.float64)
        
    def get_stats(
            self,
            output: np.ndarray,
            decoded_secret,
        ) -> Dict:
        """Compute and return statistical tests on source and `encode` method
        output and also `source`, `secret` and `output` lengths.

        Parameters
        ----------
        output : numpy.ndarray
            Output of `encode` method.

        Returns
        -------
        stats : Dict
            Results of statistical functions.
        """
        # source_bits = np.unpackbits(
        #     np.array(bytearray(self.source_data.tobytes()), dtype=np.uint8),
        #     bitorder='little')

        # output_bits = np.unpackbits(
        #     np.array(bytearray(output.tobytes()), dtype=np.uint8),
        #     bitorder='little')

        stats = {
            'snr_db': snr_db(self.source_data, output),
            'mse': mse(self.source_data, output),
            'rmsd': rmsd(self.source_data, output),
            'psnr_db': psnr_db(self.source_data, output),
            # 'ber_percent_source_output': ber_percent(source_bits, output_bits),
            'ber_percent_secret_encoded': ber_percent(self.data_to_encode,
                                                      decoded_secret),
            'source_sample_len': len(self.source_data),
            'secret_bit_len': len(self.data_to_encode),
            'output_sample_len': len(output),
        }
        return stats
