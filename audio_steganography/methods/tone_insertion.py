# -*- coding: utf-8 -*-

# File: tone_insertion.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the tone insertion method implementation.
"""

from .method_base import MethodBase, EncodeDecodeReturn, EncodeDecodeArgsReturn
from ..exceptions import SecretSizeTooLarge
from ..audio_utils import split_to_segments_of_len_n
import numpy as np

class ToneInsertion(MethodBase):
    """This is an implementation of tone insertion method.

    Examples
    --------
    Encode "42" to source array.

    >>> import numpy as np
    >>> from audio_steganography.methods import ToneInsertion
    >>> secret = np.array([0,0,1,1,0,1,0,0,0,0,1,1,0,0,1,0], dtype=np.uint8)
    >>> source = np.random.rand(secret.size * 1000)
    >>> ToneInsertion_method = ToneInsertion(source, secret)
    >>> encoded = ToneInsertion_method.encode()

    Decode

    >>> ToneInsertion_method = ToneInsertion(encoded[0])
    >>> ToneInsertion_method.decode(encoded[1]['l'])
    """

    def encode(
            self,
            f0: int = 250,
            f1: int = 350,
            **kwargs,
        ) -> EncodeDecodeReturn:
        """Encodes the secret data into source using tone insertion.

        If the secret data is bigger than source capacity, a
        `SecretSizeTooLarge` exception is raised.

        Returns
        -------
        out : method_base.EncodeDecodeReturn
            Tuple containing NumPy array of samples with secret data encoded
            using tone insertion method and additional output needed
            for decoding.
        """

        if self._secret_data.size == 0:
            return self._source_data, {
                'l': 0,
            }

        segment_len = 705
        segments, rest = split_to_segments_of_len_n(self._source_data, segment_len)
        segment_powers = np.sum(np.abs(segments)**2, axis=1) / segment_len

        tone_f0 = np.sin(2. * np.pi * f0 * np.linspace(0, 0.016, 705))
        tone_f1 = np.sin(2. * np.pi * f1 * np.linspace(0, 0.016, 705))

        tone_f0_power = np.sum(np.abs(tone_f0)**2) / len(tone_f0)
        tone_f1_power = np.sum(np.abs(tone_f1)**2) / len(tone_f1)

        return np.concatenate(segments), {
            'l': len(self._secret_data),
        }


    def decode(
            self,
            l: int,
            f0: int = 250,
            f1: int = 350,
            **kwargs,
        ) -> EncodeDecodeReturn:
        """Decode the source data by comparing ratios of values of tones at
        given frequencies.

        Parameters
        ----------
        l : int
            Number of bytes encoded in the source.

        Returns
        -------
        out : method_base.EncodeDecodeReturn
            NumPy array of uint8 zeros and ones representing the bits decoded
            using tone insertion method.
        """

        if l < 1:
            return np.empty(0, dtype=np.uint8), {}

        decoded = np.empty(0)
        return decoded, {}


    @staticmethod
    def get_encode_args() -> EncodeDecodeArgsReturn:
        args = []
        args.append((['-f0'],
                     {
                         'action': 'store',
                         'type': int,
                         'default': 250,
                         'help': 'first frequency for encoding',
                     }))
        args.append((['-f1'],
                     {
                         'action': 'store',
                         'type': int,
                         'default': 350,
                         'help': 'second frequence for encoding',
                     }))
        return args

    @staticmethod
    def get_decode_args() -> EncodeDecodeArgsReturn:
        args = []
        args.append((['-f0'],
                     {
                         'action': 'store',
                         'type': int,
                         'default': 250,
                         'help': 'first frequency for decoding',
                     }))
        args.append((['-f1'],
                     {
                         'action': 'store',
                         'type': int,
                         'default': 350,
                         'help': 'second frequence for decoding',
                     }))
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
