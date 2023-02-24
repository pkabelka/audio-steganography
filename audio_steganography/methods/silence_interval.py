# -*- coding: utf-8 -*-

# File: silence_interval.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the silence interval coding method implementation.
"""

from .method_base import MethodBase, EncodeDecodeReturn, EncodeDecodeArgsReturn
from ..exceptions import SecretSizeTooLarge
from ..audio_utils import consecutive_values
import numpy as np

class SilenceInterval(MethodBase):
    """This is an implementation of silence interval coding method.

    Examples
    --------
    Encode "42" to source array.

    >>> import numpy as np
    >>> from audio_steganography.methods import SilenceInterval
    >>> secret = np.array([0,0,1,1,0,1,0,0,0,0,1,1,0,0,1,0], dtype=np.uint8)
    >>> source = np.random.rand(secret.size * 2)
    >>> SilenceInterval_method = SilenceInterval(source, secret)
    >>> encoded = SilenceInterval_method.encode()

    Decode

    >>> SilenceInterval_method = SilenceInterval(encoded[0])
    >>> SilenceInterval_method.decode()
    """

    min_silence_len = 600

    def encode(self, **kwargs) -> EncodeDecodeReturn:
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

        secret = np.packbits(
            self._secret_data,
            axis=-1,
            bitorder='big')

        # trivial check if segments can fit in source
        if silence_lens[silence_lens > self.min_silence_len].size < len(secret):
            raise SecretSizeTooLarge('secret data cannot fit in source: '+
                f'len(secret) = {secret.size} bytes, '+
                f'very inaccurate approx capacity(source) = '+
                f'{silence_lens[silence_lens > self.min_silence_len + 256].size} bytes')

        segments = np.split(self._source_data, silence_starts[1:])

        secret_idx = 0
        for i, segment in enumerate(segments):
            if secret_idx == len(secret):
                break

            new_len = len(segment) - ((len(segment) - secret[secret_idx]) % 256)
            if len(segment) < self.min_silence_len + 256 or new_len < self.min_silence_len + 256:
                continue

            # shorten the segment
            segments[i] = segments[i][:new_len]

            secret_idx += 1

        # actual check if segments can fit in source
        if secret_idx != len(secret):
            raise SecretSizeTooLarge('secret data cannot fit in source: '+
                f'len(secret) = {secret.size} bytes, '+
                f'very inaccurate approx capacity(source) = '+
                f'{silence_lens[silence_lens > self.min_silence_len + 256].size} bytes')

        return np.concatenate(segments), {
            'l': len(secret),
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
            using silence interval coding method.
        """

        if l < 1:
            return np.zeros(0), {}

        _, silence_lens = consecutive_values(
            np.abs(self._source_data) <= 0.15 * np.abs(self._source_data).max())

        decoded = np.array(
            silence_lens[silence_lens >= self.min_silence_len + 256][:l] % 256,
            dtype=np.uint8)

        decoded = np.unpackbits(decoded, axis=-1, bitorder='big')
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
