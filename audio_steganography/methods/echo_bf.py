# -*- coding: utf-8 -*-

# File: echo_bf.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the EchoBF class
"""

from .echo_base import EchoBase
from .method_base import EncodeDecodeReturn
from ..audio_utils import (
    split_to_n_segments,
    spread_bits,
    to_dtype,
    autocorr_scipy_correlate,
    center,
    normalize,
)
from typing import Optional
import numpy as np

class EchoBF(EchoBase):
    """This is an implementation of backward forward echo hiding method using a
    kernel with two echos for each of the hidden bits. One echo is moved back
    and second one is moved forward.

    Kernel visualization; middle represents original sample.

         |
    |    |    |
    ...........

    Examples
    --------
    Encode "42" to source array.

    >>> import numpy as np
    >>> from audio_steganography.methods import EchoBF
    >>> secret = np.array([0,0,1,1,0,1,0,0,0,0,1,1,0,0,1,0], dtype=np.uint8)
    >>> source = np.random.rand(len(secret) * 8192, 1)
    >>> echo_method = EchoBF(source, secret)
    >>> encoded = echo_method.encode(250, 350)

    Decode

    >>> echo_method = EchoBF(encoded[0])
    >>> echo_method.decode(250, 350)
    """

    def _encode(
            self,
            d0: int,
            d1: int,
            alpha = 0.5,
            decay_rate = 0.85,
        ) -> EncodeDecodeReturn:

        secret_len = self._secret_data.size
        if secret_len == 0 or d0 >= d1:
            return self._source_data, {
                'd0': -1,
                'd1': -1,
                'l': 0,
            }

        mixer = spread_bits(self._secret_data, self._source_data.size)
        source_pad = np.pad(self._source_data, d1) * alpha/2

        # left side: d0 zeros
        echo_0_fwd = source_pad[d1-d0:]
        # right side: d0 zeros
        echo_0_bwd = source_pad[d1+d0:]

        # left side: d1 zeros
        echo_1_fwd = source_pad
        # right side: d1 zeros
        echo_1_bwd = source_pad[d1+d1:]

        encoded = (
            self._source_data[:len(mixer)] +
            echo_1_fwd[:len(mixer)] * mixer +
            echo_1_bwd[:len(mixer)] * mixer +
            echo_0_fwd[:len(mixer)] * np.abs(1-mixer) +
            echo_0_bwd[:len(mixer)] * np.abs(1-mixer)
        )

        encoded = center(encoded)
        encoded = normalize(encoded)
        encoded = to_dtype(encoded, self._source_data.dtype)

        return encoded, {
            'd0': d0,
            'd1': d1,
            'l': secret_len,
        }

    def encode(
            self,
            d0: Optional[int] = None,
            d1: Optional[int] = None,
            delay_search = '',
            alpha = 0.5,
            decay_rate = 0.85,
            **kwargs,
        ) -> EncodeDecodeReturn:
        """Encodes the secret data into source. If the returned d0, d1 and l
        are all -1, then the method failed to encode correctly.

        If the secret data is bigger than source capacity, a
        `SecretSizeTooLarge` exception is raised.

        Parameters
        ----------
        d0 : int | None
            Echo delay for binary 0. Default value is 150.
        d1 : int | None
            Echo delay for binary 1. Default value is d0 + 50
        delay_search : str
            Method for searching optimal d0 and d1. Valid options are:
            - '' : do not search, encode with the supplied values
            - 'basinhopping' : search using scipy.optimize.basinhopping method,
              stops at 50 iterations and returns the best values found
            - 'bruteforce' : loop through 5000 possible values
            This searching can take a very long time depending on the length of
            `source`.
        alpha : float
           Echo amplitude multiplier.
        decay_rate : float
            Decay rate of echo amplitude.

        Returns
        -------
        out : MethodBase.EncodeDecodeReturn
            Tuple containing NumPy array of samples with secret data encoded
            using echo single kernel method and additional output needed for
            decoding.
        """

        return super().encode_wrapper(
            EchoBF,
            d0,
            d1,
            delay_search,
            alpha,
            decay_rate,
            **kwargs)


    def decode(self, d0: int, d1: int, l: int, **kwargs) -> EncodeDecodeReturn:
        """Decode with the supplied d0, d1 and l values.

        Decoding of this method involves checking the height of the peaks in
        autoceptrum of the segement. The autocepstrum is just an autorrelated
        power cepstrum.

        Parameters
        ----------
        d0 : int
            Echo delay for binary 0.
        d1 : int
            Echo delay for binary 1.
        l : int
            Number of bits encoded in the source.

        Returns
        -------
        out : MethodBase.EncodeDecodeReturn
            NumPy array of uint8 zeros and ones representing the bits decoded
            using echo backward-forward kernel method.
        """

        if l < 1:
            return np.empty(0, dtype=np.uint8), {}

        split, _ = split_to_n_segments(self._source_data, l)
        decoded = np.zeros(len(split), dtype=np.uint8)

        for i, segment in enumerate(split):
            cn = autocorr_scipy_correlate(
                # power cepstrum
                np.fft.irfft(np.log(np.abs(np.fft.rfft(segment)))**2)
            )
            if cn[d0] > cn[d1]:
                decoded[i] = 0
            else:
                decoded[i] = 1

        return decoded, {}
