# -*- coding: utf-8 -*-

# File: lsb.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the LSB method implementation
"""

from .method_base import MethodBase, EncodeDecodeReturn, EncodeDecodeArgsReturn
from ..exceptions import SecretSizeTooLarge
from ..audio_utils import to_dtype, seg_split_len_n
from typing import Optional
import numpy as np

class LSB(MethodBase):
    """This is an implementation of least significant bit substitution method.

    Examples
    --------
    Encode "42" to source array.

    >>> import numpy as np
    >>> from audio_steganography.methods.lsb import LSB
    >>> secret = np.array([0,0,1,1,0,1,0,0,0,0,1,1,0,0,1,0], dtype=np.uint8)
    >>> source = np.random.rand(len(secret) * 2)
    >>> LSB_method = LSB(source, secret)
    >>> encoded = LSB_method.encode()

    Decode

    >>> LSB_method = LSB(encoded[0])
    >>> LSB_method.decode()
    """

    def encode(self, depth: int = 1) -> EncodeDecodeReturn:
        """Encodes the secret data into source using least significant bit
        substitution.

        If the secret data is bigger than source capacity, a
        `SecretSizeTooLarge` exception is raised.

        Parameters
        ----------
        depth : int
            Number of bits to encode in each sample. Default is 1.

        Returns
        -------
        out : method_base.EncodeDecodeReturn
            Tuple containing NumPy array of samples with secret data encoded
            least significant bit substitution method and additional output
            needed for decoding.
        """

        # TODO: support more than 8
        depth = int(depth)
        if depth < 1 or depth > 8:
            raise ValueError('bit depth must be between 1 and 8')

        # pad secret data to nearest bit depth multiple
        secret_padded_to_bit_depth = np.pad(
            self._secret_data,
            (0, np.ceil(len(self._secret_data) / depth).
                 astype(np.uint32) * depth - len(self._secret_data)))

        # split to bit depth long arrays
        secret_split_by_depth = seg_split_len_n(secret_padded_to_bit_depth,
                                                depth)

        if len(secret_split_by_depth) > len(self._source_data):
            raise SecretSizeTooLarge('secret data cannot fit in source: '+
                f'len(secret) = {len(secret_split_by_depth)}, '+
                f'capacity(source) = {len(self._source_data)}')

        # convert float dtypes to int64
        source = self._source_data
        source_dtype = source.dtype
        if source_dtype in [np.float16, np.float32, np.float64]:
            source = to_dtype(source, np.int32)

        # convert bits to uint8 numbers
        secret_packed = np.packbits(
            secret_split_by_depth,
            axis=-1,
            bitorder='little').flatten()

        # zero out LSB
        encoded = np.bitwise_and(source, np.bitwise_not(2**depth - 1))

        # encode secret data to LSB
        encoded = np.bitwise_or(
            encoded,
            np.pad(
                secret_packed,
                (0, len(self._source_data) - len(secret_packed))
            )
        )

        if source_dtype in [np.float16, np.float32, np.float64]:
            encoded = to_dtype(encoded, source_dtype)

        return encoded, {
            'l': len(self._secret_data),
        }


    def decode(
            self,
            depth: int = 1,
            l: Optional[int] = None
        ) -> EncodeDecodeReturn:
        """Decode using plain least significant bit substitution.

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

        # TODO: support more than 8
        depth = int(depth)
        if depth < 1 or depth > 8:
            raise ValueError('bit depth must be between 1 and 8')

        _len = len(self._source_data)
        if l is not None:
            _len = l

        # convert float dtypes to int64
        source = self._source_data
        if source.dtype in [np.float16, np.float32, np.float64]:
            source = to_dtype(source, np.int32)

        # get least significant bits up to bit depth
        lsb = np.bitwise_and(source[:_len], 2**depth - 1).astype(np.uint8)

        # unpack bytes to bits
        unpacked = np.unpackbits(lsb, bitorder='little')

        # split to byte size arrays and extract up to bit depth in each array
        unpacked_split = np.array(seg_split_len_n(unpacked, 8))[:, :depth]

        return unpacked_split.flatten()[:_len], {}


    @staticmethod
    def get_encode_args() -> EncodeDecodeArgsReturn:
        args = []
        args.append((['-d', '--depth'],
                     {
                         'action': 'store',
                         'type': int,
                         'default': 1,
                         'help': 'number of bits to encode in a sample',
                     }))
        return args

    @staticmethod
    def get_decode_args() -> EncodeDecodeArgsReturn:
        args = []
        args.append((['-d', '--depth'],
                     {
                         'action': 'store',
                         'type': int,
                         'default': 1,
                         'help': 'number of bits encoded in a sample',
                     }))
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
