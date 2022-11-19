# -*- coding: utf-8 -*-

# File: lsb.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the LSB method implementation
"""

from .method_base import MethodBase, EncodeDecodeReturn
from ..exceptions import SecretSizeTooLarge
from ..audio_utils import to_dtype
import numpy as np

class LSB(MethodBase):
    """This is an implementation of least significant bit substitution method.

    Examples
    --------
    Encode "42" to source array.

    >>> import numpy as np
    >>> secret = np.array([0,0,1,1,0,1,0,0,0,0,1,1,0,0,1,0], dtype=np.uint8)
    >>> source = np.random.rand(len(secret) * 8192, 1)
    >>> LSB_method = LSB(source, secret)
    >>> encoded = LSB_method.encode()

    Decode
    >>> LSB_method = LSB(encoded)
    >>> LSB_method.decode()
    """

    def encode(self) -> EncodeDecodeReturn:
        """Encodes the secret data into source using least significant bit
        substitution.
        """

        if len(self._secret_data) > len(self._source_data):
            raise SecretSizeTooLarge('secret data cannot fit in source: ' +
                f'len(secret) = {len(self._secret_data)}, capacity(source) = '+
                f'{len(self._source_data)}')

        encoded = to_dtype(self._source_data, np.int16)
        # zero out LSB
        encoded = np.bitwise_and(encoded, np.bitwise_not(1))

        # encode secret data to LSB
        encoded = np.bitwise_or(
            encoded,
            np.pad(
                self._secret_data,
                (0, len(self._source_data) - len(self._secret_data))
            )
        )
        encoded = to_dtype(encoded, np.float64)
        return encoded, {}


    def decode(self) -> EncodeDecodeReturn:
        """Decode using plain least significant bit substitution.
        """

        decoded = to_dtype(self._source_data, np.int16)
        decoded = np.bitwise_and(decoded, 1)
        return decoded, {}
