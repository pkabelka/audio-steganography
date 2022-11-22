# -*- coding: utf-8 -*-

# File: lsb.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the LSB method implementation
"""

from .method_base import MethodBase, EncodeDecodeReturn, EncodeDecodeArgsReturn
from ..exceptions import SecretSizeTooLarge
from ..audio_utils import to_dtype
from typing import Optional
import numpy as np

class LSB(MethodBase):
    """This is an implementation of least significant bit substitution method.

    Examples
    --------
    Encode "42" to source array.

    >>> import numpy as np
    >>> secret = np.array([0,0,1,1,0,1,0,0,0,0,1,1,0,0,1,0], dtype=np.uint8)
    >>> source = np.random.rand(len(secret) * 32)
    >>> LSB_method = LSB(source, secret)
    >>> encoded = LSB_method.encode()

    Decode

    >>> LSB_method = LSB(encoded[0])
    >>> LSB_method.decode()
    """

    def encode(self) -> EncodeDecodeReturn:
        """Encodes the secret data into source using least significant bit
        substitution.

        If the secret data is bigger than source capacity, a
        `SecretSizeTooLarge` exception is raised.
        """

        if len(self._secret_data) > len(self._source_data):
            raise SecretSizeTooLarge('secret data cannot fit in source: '+
                f'len(secret) = {len(self._secret_data)}, capacity(source) = '+
                f'{len(self._source_data)}')

        # convert float dtypes to int64
        source = self._source_data
        source_dtype = source.dtype
        if source_dtype in [np.float16, np.float32, np.float64]:
            source = to_dtype(source, np.int32)

        # zero out LSB
        encoded = np.bitwise_and(source, np.bitwise_not(1))

        # encode secret data to LSB
        encoded = np.bitwise_or(
            encoded,
            np.pad(
                self._secret_data,
                (0, len(self._source_data) - len(self._secret_data))
            )
        )
        if source_dtype in [np.float16, np.float32, np.float64]:
            encoded = to_dtype(encoded, source_dtype)
        return encoded, {
            'l': len(self._secret_data),
        }


    def decode(self, l: Optional[int] = None) -> EncodeDecodeReturn:
        """Decode using plain least significant bit substitution.
        """

        _len = len(self._source_data)
        if l is not None:
            _len = l

        # convert float dtypes to int64
        source = self._source_data
        if source.dtype in [np.float16, np.float32, np.float64]:
            source = to_dtype(source, np.int32)

        # read last significant bit
        decoded = np.bitwise_and(source[:_len], 1)
        return decoded, {}


    @staticmethod
    def get_decode_args() -> EncodeDecodeArgsReturn:
        args = []
        args.append((['-l', '--len'],
                     {
                         'action': 'store',
                         'type': int,
                         'required': False,
                         'help': 'encoded data length; decode only this many '+
                             'bits',
                         'default': None,
                     }))
        return args
