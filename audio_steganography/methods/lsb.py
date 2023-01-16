# -*- coding: utf-8 -*-

# File: lsb.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the LSB method implementation
"""

from .method_base import MethodBase, EncodeDecodeReturn, EncodeDecodeArgsReturn
from ..exceptions import SecretSizeTooLarge
from ..audio_utils import to_dtype, split_to_segments_of_approx_len_n
from typing import Optional
import numpy as np

dtype_conv = {
    np.dtype(np.float64): np.int64,
    np.dtype(np.float32): np.int32,
    np.dtype(np.float16): np.int16,
}

class LSB(MethodBase):
    """This is an implementation of least significant bit substitution method.

    Examples
    --------
    Encode "42" to source array.

    >>> import numpy as np
    >>> from audio_steganography.methods import LSB
    >>> secret = np.array([0,0,1,1,0,1,0,0,0,0,1,1,0,0,1,0], dtype=np.uint8)
    >>> source = np.random.rand(secret.size * 2)
    >>> LSB_method = LSB(source, secret)
    >>> encoded = LSB_method.encode()

    Decode

    >>> LSB_method = LSB(encoded[0])
    >>> LSB_method.decode()
    """

    def encode(self, depth: int = 1, **kwargs) -> EncodeDecodeReturn:
        """Encodes the secret data into source using least significant bit
        substitution.

        If `depth` is less than 1 or more than the number of bits the source
        dtype provides, a `ValueError` exception is raised.

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

        # convert float dtypes to int dtypes
        source = self._source_data
        source_dtype = source.dtype
        if source.dtype in dtype_conv:
            source = to_dtype(source, dtype_conv[source.dtype])

        depth = int(depth)
        if depth < 1 or depth > np.iinfo(source.dtype).bits:
            raise ValueError(f'bit depth must be between 1 and '+
                             f'{np.iinfo(source.dtype).bits}')

        secret = self._secret_data
        if depth > 1:
            # pad secret data to nearest bit depth multiple
            secret_padded_to_bit_depth = np.pad(
                self._secret_data,
                (0, np.ceil(len(self._secret_data) / depth).
                     astype(np.uint32) * depth - len(self._secret_data)))

            # split to bit depth long arrays
            secret = np.array(
                split_to_segments_of_approx_len_n(secret_padded_to_bit_depth, depth),
                dtype=np.uint8)

        if secret.size > source.size:
            raise SecretSizeTooLarge('secret data cannot fit in source: '+
                f'secret.size = {secret.size}, '+
                f'capacity(source) = {source.size}')

        if depth > 1:
            # convert bits to uint8 numbers
            secret = np.packbits(
                secret,
                axis=-1,
                bitorder='little').flatten()

        # zero out LSB
        encoded = np.bitwise_and(source, np.bitwise_not(2**depth - 1))

        # encode secret data to LSB
        encoded = np.bitwise_or(
            encoded,
            np.pad(
                secret,
                (0, source.size - secret.size)
            )
        )

        if source_dtype in dtype_conv:
            encoded = to_dtype(encoded, source_dtype)

        return encoded, {
            'l': len(self._secret_data),
            'depth': depth,
        }


    def decode(
            self,
            depth: int = 1,
            l: Optional[int] = None,
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

        depth = int(depth)
        if depth < 1 or depth > np.iinfo(self._source_data.dtype).bits:
            raise ValueError(f'bit depth must be between 1 and '+
                             f'{np.iinfo(self._source_data.dtype).bits}')

        _len = self._source_data.size
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

        # when encoding, bits representing larger numbers than 255 will get
        # split to multiple bytes
        # chunk_size stores number of bits of those bytes
        chunk_size = int(np.ceil(depth / 8) * 8)
        padded = np.pad(
            unpacked,
            (0, np.ceil(len(unpacked) / chunk_size).
                 astype(np.uint32) * chunk_size - len(unpacked)))

        # split to chunk size arrays and extract up to bit depth in each array
        unpacked_split = np.array(
            split_to_segments_of_approx_len_n(padded, chunk_size))[:, :depth]

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
