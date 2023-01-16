# -*- coding: utf-8 -*-

# File: phase_coding.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the phase coding method implementation
"""

from .method_base import MethodBase, EncodeDecodeReturn, EncodeDecodeArgsReturn
from ..exceptions import SecretSizeTooLarge
from ..audio_utils import to_dtype, seg_split_len_n_exact, seg_split_exact_len
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
                'l': -1,
            }

        seg, rest = seg_split_len_n_exact(self._source_data, secret_len)
        # l = len(seg[0])
        fft = np.fft.fft(seg)
        angles = np.angle(fft)
        diff = np.diff(angles, axis=0)
        diff = np.concatenate(([np.zeros(secret_len)], diff))
        # diff = np.concatenate(([np.zeros(l)], diff))
        # convert secret bits to [-pi/2; pi/2] interval; 0 => pi/2; 1 => -pi/2
        secret_angles = (self._secret_data * 2 - 1) * -np.pi/2
        angles[0] = secret_angles
        # angles[0][int(len(fft[0])/2)-secret_len:int(len(fft[0])/2)] = secret_angles
        # angles[0][int(len(fft[0])/2):int(len(fft[0])/2)+secret_len] = -np.flip(secret_angles)
        # new_angles = angles + (np.concatenate((diff[1:], [np.zeros(secret_len)])))
        # new_angles = np.concatenate(([secret_angles], new_angles[:-1]))
        new_angles = angles.copy()
        for i in range(1, len(angles)):
            new_angles[i] = new_angles[i-1] + diff[i]

        encoded = np.abs(fft) * np.exp(1j * new_angles)
        encoded = np.fft.ifft(encoded).real.flatten()

        # center, normalize range and convert to the original dtype
        encoded = encoded - np.mean(encoded)
        if np.abs(encoded).max() != 0:
            encoded = encoded / np.abs(encoded).max()
        encoded = to_dtype(encoded, self._source_data.dtype)

        return encoded, {
            'l': secret_len,
        }


    def decode(
            self,
            l: int,
            **kwargs,
        ) -> EncodeDecodeReturn:
        """Decode using plain least significant bit substitution.

        If `depth` is less than 1 or more than the number of bits the source
        dtype provides, a `ValueError` exception is raised.

        Parameters
        ----------
        depth : int
            Number of bits encoded in each sample. Default is 1.
        l : int | None
            Number of bits encoded in the source. If `l` is set to `None`, then
            decode will return all least significant bits.

        Returns
        -------
        out : method_base.EncodeDecodeReturn
            NumPy array of uint8 zeros and ones representing the bits decoded
            using least significant bit substitution method.
        """

        _len = self._source_data.size
        if l is not None:
            _len = l

        # seg = self._source_data[:int(np.floor(len(self._source_data) / _len))]
        seg = self._source_data[:_len]
        fft = np.fft.fft(seg)
        angles = np.angle(fft)
        decoded = np.array(angles < 0, dtype=np.uint8)

#         decoded = np.zeros(_len, dtype=np.uint8)
#         for i in range(_len):
#             if angles[int(l/2) - _len + i] > 0:
#                 decoded[i] = 0
#             else:
#                 decoded[i] = 1

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
