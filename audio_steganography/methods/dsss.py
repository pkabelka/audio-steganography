# -*- coding: utf-8 -*-

# File: dsss.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the Direct sequence spread spectrum method
implementation
"""

from .method_base import MethodBase, EncodeDecodeReturn, EncodeDecodeArgsReturn
from ..exceptions import SecretSizeTooLarge
from ..audio_utils import to_dtype, split_to_n_segments, mixer_sig
from typing import Optional
import numpy as np
import hashlib

class DSSS(MethodBase):
    """This is an implementation of Direct sequence spread spectrum method.

    Examples
    --------
    Encode "42" to source array.

    >>> import numpy as np
    >>> from audio_steganography.methods import DSSS
    >>> secret = np.array([0,0,1,1,0,1,0,0,0,0,1,1,0,0,1,0], dtype=np.uint8)
    >>> source = np.random.rand(secret.size * 2)
    >>> DSSS_method = DSSS(source, secret)
    >>> encoded = DSSS_method.encode()

    Decode

    >>> DSSS_method = DSSS(encoded[0])
    >>> DSSS_method.decode()
    """

    def encode(
            self,
            password: str = '',
            alpha: float = 0.005,
            **kwargs
        ) -> EncodeDecodeReturn:
        """Encodes the secret data into source using direct sequence spread
        spectrum method.

        If the secret data is bigger than source capacity, a
        `SecretSizeTooLarge` exception is raised.

        Parameters
        ----------
        password : str
            Password for seeding the PSRNG to generate the pseudo-random
            sequence from.

        Returns
        -------
        out : method_base.EncodeDecodeReturn
            Tuple containing NumPy array of samples with secret data encoded
            using direct sequance spread spectrum method and additional output
            needed for decoding.
        """

        mixer = mixer_sig(self._secret_data, self._source_data.size)
        mixer = mixer.astype(np.float64) * 2 - 1

        # center and normalize source to [-1; 1]
        source = self._source_data - np.mean(self._source_data)
        if np.abs(source).max() != 0:
            source = source / np.abs(source).max()

        hash = hashlib.sha256()
        hash.update(password.encode('utf-8'))

        # using `numpy.random` module because `secrets` module does not allow seeding
        # TODO: allow generating PN-sequence file
        pn_generator = np.random.RandomState(seed=np.frombuffer(hash.digest(), dtype=np.uint32))
        pn_sequence = pn_generator.choice([-1, 1], size=len(mixer))

        encoded = source + mixer * alpha * pn_sequence

        # center, normalize range and convert to the original dtype
        encoded = encoded - np.mean(encoded)
        if np.abs(encoded).max() != 0:
            encoded = encoded / np.abs(encoded).max()
        encoded = to_dtype(encoded, self._source_data.dtype)

        return encoded, {
            'l': len(self._secret_data),
            'password': password,
        }


    def decode(
            self,
            l: int,
            password: str = '',
            **kwargs,
        ) -> EncodeDecodeReturn:
        """Decode using direct sequence spread spectrum.

        Parameters
        ----------
        password : str
            Password for seeding the PSRNG to generate the pseudo-random
            sequence from.
        l : int | None
            Number of bits encoded in the source. If `l` is set to `None`, then
            decode will use all source samples.

        Returns
        -------
        out : method_base.EncodeDecodeReturn
            NumPy array of uint8 zeros and ones representing the bits decoded
            using least significant bit substitution method.
        """

        if l < 1:
            return np.zeros(0), {}

        hash = hashlib.sha256()
        hash.update(password.encode('utf-8'))
        pn_generator = np.random.RandomState(seed=np.frombuffer(hash.digest(), dtype=np.uint32))
        pn_sequence = pn_generator.choice([-1, 1], size=len(self._source_data))

        source_segments, _ = split_to_n_segments(self._source_data, l)
        pn_sequence_segments, _ = split_to_n_segments(pn_sequence, l)

        decoded = np.zeros(l, dtype=np.uint8)
        for i in range(l):
            corr = np.sum(source_segments[i] * pn_sequence_segments[i])

            if corr > 0:
                decoded[i] = 1
            else:
                decoded[i] = 0

        # decoded = np.array(
        #     np.sum(
        #         source_segments * pn_sequence_segments,
        #         axis=1
        #     ) > 0, dtype=np.uint8
        # )

        return decoded, {}


    @staticmethod
    def get_encode_args() -> EncodeDecodeArgsReturn:
        args = []
        args.append((['-p', '--password'],
                     {
                         'action': 'store',
                         'type': str,
                         'required': True,
                         'default': '',
                         'help': 'number of bits to encode in a sample',
                     }))
        args.append((['-a', '--alpha'],
                     {
                         'action': 'store',
                         'type': float,
                         'required': False,
                         'default': 0.005,
                         'help': 'encoding sequence amplitude multiplier',
                     }))
        return args

    @staticmethod
    def get_decode_args() -> EncodeDecodeArgsReturn:
        args = []
        args.append((['-p', '--password'],
                     {
                         'action': 'store',
                         'type': str,
                         'required': True,
                         'default': '',
                         'help': 'number of bits to encode in a sample',
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
