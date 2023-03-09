# -*- coding: utf-8 -*-

# File: silence_interval.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the silence interval coding method implementation.
"""

from .method_base import MethodBase, EncodeDecodeReturn, EncodeDecodeArgsReturn
from ..exceptions import SecretSizeTooLarge
from ..audio_utils import consecutive_values, split_to_segments_of_len_n
import numpy as np

class SilenceInterval(MethodBase):
    """This is an implementation of silence interval coding method.

    Examples
    --------
    Encode "42" to source array.

    >>> import numpy as np
    >>> from audio_steganography.methods import SilenceInterval
    >>> secret = np.array([0,0,1,1,0,1,0,0,0,0,1,1,0,0,1,0], dtype=np.uint8)
    >>> source = np.cumsum(np.random.rand(secret.size * 10000))
    >>> SilenceInterval_method = SilenceInterval(source, secret)
    >>> encoded = SilenceInterval_method.encode()

    Decode

    >>> SilenceInterval_method = SilenceInterval(encoded[0])
    >>> SilenceInterval_method.decode(encoded[1]['l'])
    """

    def encode(self, min_silence_len=400, **kwargs) -> EncodeDecodeReturn:
        """Encodes the secret data into source using silence interval coding.

        If the secret data is bigger than source capacity, a
        `SecretSizeTooLarge` exception is raised.

        Returns
        -------
        out : method_base.EncodeDecodeReturn
            Tuple containing NumPy array of samples with secret data encoded
            using silence interval coding method and additional output needed
            for decoding.
        """

        if self._secret_data.size == 0:
            return self._source_data, {
                'l': 0,
            }

        silence_starts, silence_lens = consecutive_values(
            np.abs(self._source_data) <= 0.15 * np.abs(self._source_data).max())

        # padding in case secret is not multiple of 4
        secret_padded_to_multiple4 = np.pad(
            self._secret_data,
            (0, np.ceil(len(self._secret_data) / 4).
                 astype(np.uint32) * 4 - len(self._secret_data)))

        secret, _ = split_to_segments_of_len_n(secret_padded_to_multiple4, 4)
        secret = np.packbits(
            secret.astype(np.uint8),
            axis=-1,
            bitorder='little').ravel()

        # trivial check if segments can fit in source
        if silence_lens[silence_lens > min_silence_len].size < len(secret):
            raise SecretSizeTooLarge('secret data cannot fit in source: '+
                f'len(secret) = {secret.size} bytes, '+
                f'very inaccurate approx capacity(source) = '+
                f'{silence_lens[silence_lens > min_silence_len].size} bytes')

        segments = np.split(self._source_data, silence_starts[1:])

        secret_idx = 0
        for i, segment in enumerate(segments):
            if secret_idx == len(secret):
                break

            new_len = len(segment) - ((len(segment) - secret[secret_idx]) % 16)
            if len(segment) < min_silence_len or new_len < min_silence_len:
                continue

            # shorten the segment
            segments[i] = segments[i][:new_len]

            secret_idx += 1

        # actual check if segments can fit in source
        if secret_idx != len(secret):
            raise SecretSizeTooLarge('secret data cannot fit in source: '+
                f'len(secret) = {secret.size} bytes, '+
                f'very inaccurate approx capacity(source) = '+
                f'{silence_lens[silence_lens > min_silence_len].size} bytes')

        return np.concatenate(segments), {
            'l': len(self._secret_data),
        }


    def decode(
            self,
            l: int,
            min_silence_len=400,
            **kwargs,
        ) -> EncodeDecodeReturn:
        """Decode the source data by finding lengths of silence intervals and
        calculating the byte value with modulo operator.

        Parameters
        ----------
        l : int
            Number of bytes encoded in the source.

        Returns
        -------
        out : method_base.EncodeDecodeReturn
            NumPy array of uint8 zeros and ones representing the bits decoded
            using silence interval coding method.
        """

        if l < 1:
            return np.zeros(0), {}

        _, silence_lens = consecutive_values(
            np.abs(self._source_data) <= 0.15 * np.abs(self._source_data).max())

        decoded = np.array(
            silence_lens[silence_lens >= min_silence_len][
                :int(np.ceil(len(self._source_data) / 4))] % 16,
            dtype=np.uint8)

        # take every other 4 bits
        decoded = np.unpackbits(
            decoded, axis=-1,
            bitorder='little').reshape(-1, 4)[::2, :].reshape(-1)[:l]
        return decoded, {}


    @staticmethod
    def get_encode_args() -> EncodeDecodeArgsReturn:
        args = []
        args.append((['-m', '--min_silence_len'],
                     {
                         'action': 'store',
                         'type': int,
                         'required': False,
                         'help': 'minimum length of a silence interval; '+
                             'default: 400',
                         'default': 400,
                     }))
        return args

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
        args.append((['-m', '--min_silence_len'],
                     {
                         'action': 'store',
                         'type': int,
                         'required': False,
                         'help': 'minimum length of a silence interval; '+
                             'default: 400',
                         'default': 400,
                     }))
        return args
