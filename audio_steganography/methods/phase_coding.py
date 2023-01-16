# -*- coding: utf-8 -*-

# File: phase_coding.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the phase coding method implementation
"""

from .method_base import MethodBase, EncodeDecodeReturn, EncodeDecodeArgsReturn
from ..exceptions import SecretSizeTooLarge
from ..audio_utils import to_dtype, split_to_segments_of_len_n
from typing import Optional
import numpy as np

class PhaseCoding(MethodBase):
    """This is an implementation of phase coding method.

    Examples
    --------
    Encode "42" to source array.

    >>> import numpy as np
    >>> from audio_steganography.methods import PhaseCoding
    >>> secret = np.array([0,0,1,1,0,1,0,0,0,0,1,1,0,0,1,0], dtype=np.uint8)
    >>> source = np.random.rand(secret.size * 2)
    >>> PhaseCoding_method = PhaseCoding(source, secret)
    >>> encoded = PhaseCoding_method.encode()

    Decode

    >>> PhaseCoding_method = PhaseCoding(encoded[0])
    >>> PhaseCoding_method.decode()
    """

    def encode(self, **kwargs) -> EncodeDecodeReturn:
        """Encodes the secret data into source using phase coding.

        If the secret data is bigger than source capacity, a
        `SecretSizeTooLarge` exception is raised.

        Returns
        -------
        out : method_base.EncodeDecodeReturn
            Tuple containing NumPy array of samples with secret data encoded
            using phase coding method and additional output needed for
            decoding.
        """

        secret_len = self._secret_data.size
        if secret_len == 0:
            return self._source_data, {
                'l': 0,
            }

        # segment_len should be twice as large as secret_len for secret_angles
        # to blend in
        segment_len = int(2 * 2**np.ceil(np.log2(2 * secret_len)))

        # check if segments can fit in source
        if segment_len > self._source_data.size:
            raise SecretSizeTooLarge('secret data cannot fit in source: '+
                f'len(secret) = {self._secret_data.size} bits, '+
                f'capacity(source) = {segment_len//8} bits')

        seg, rest = split_to_segments_of_len_n(self._source_data, segment_len)
        fft = np.fft.fft(seg)
        angles = np.angle(fft)

        # get difference of angles across segments column-wise
        diff = np.diff(angles, axis=0)

        # convert secret bits to [-pi/2; pi/2] interval; 0 => pi/2; 1 => -pi/2
        secret_angles = (self._secret_data.astype(np.float64) * 2 - 1) * -np.pi/2

        # insert secret_angles to the middle of the first segment
        angles[0, segment_len//2 - secret_len : segment_len//2] = secret_angles
        angles[0, segment_len//2 + 1 : segment_len//2 + 1 + secret_len] = -np.flip(secret_angles)

        # adjust phases of subsequent segments
        for i in range(1, len(angles)):
            angles[i] = angles[i-1] + diff[i-1]

        # reconstruct the signal values from new angles
        encoded = np.abs(fft) * np.exp(1j * angles)
        encoded = np.fft.ifft(encoded).real.flatten()

        # center, normalize range and convert to the original dtype
        encoded = encoded - np.mean(encoded)
        if np.abs(encoded).max() != 0:
            encoded = encoded / np.abs(encoded).max()
        encoded = to_dtype(encoded, self._source_data.dtype)

        return np.append(encoded, rest), {
            'l': secret_len,
        }


    def decode(
            self,
            l: int,
            **kwargs,
        ) -> EncodeDecodeReturn:
        """Decode the source data by converting values of angles in the first
        >0 to 0 and <0 to 1.

        Parameters
        ----------
        l : int
            Number of bits encoded in the source.

        Returns
        -------
        out : method_base.EncodeDecodeReturn
            NumPy array of uint8 zeros and ones representing the bits decoded
            using phase coding method.
        """

        if l < 1:
            return np.zeros(0), {}

        # get the first segment
        segment_len = int(2 * 2**np.ceil(np.log2(2 * l)))
        seg = self._source_data[:segment_len]

        # get phase values
        fft = np.fft.fft(seg)
        angles = np.angle(fft)

        # convert values >0 to 0 and values <0 to 1
        decoded = np.array(angles[segment_len//2 - l : segment_len//2] < 0, dtype=np.uint8)
        return decoded, {}


    @staticmethod
    def get_decode_args() -> EncodeDecodeArgsReturn:
        args = []
        args.append((['-l', '--len'],
                     {
                         'action': 'store',
                         'type': int,
                         'required': True,
                         'help': 'encoded data length; decode only this many '+
                             'bits',
                         'default': None,
                     }))
        return args
