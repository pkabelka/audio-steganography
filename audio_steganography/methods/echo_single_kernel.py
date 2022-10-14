# -*- coding: utf-8 -*-

# File: echo_single_kernel.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the Echo_single_kernel class
"""

from .method_base import MethodBase
from ..audio_utils import seg_split, mixer_sig
from typing import Tuple, Dict, List, Any, Optional
import numpy as np
import scipy.signal

from ..stat_utils import ber_percent
import scipy.optimize

class Echo_single_kernel(MethodBase):
    def __init__(self, source_data: np.ndarray):
        super().__init__(source_data)
        self._alpha = 0.5
        self._decay_rate = 0.85

    def _encode(
            self,
            d0: int,
            d1: int,
        ) -> Tuple[np.ndarray, Dict[str, Any]]:

        secret_len = len(self._secret_data)
        mixer = mixer_sig(self._secret_data, len(self._source_data))

        # echo kernel for binary 0
        k0 = np.append(np.zeros(d0), [1]) * self._alpha
        # echo kernel for binary 1
        k1 = np.append(np.zeros(d1), [1]) * self._alpha * self._decay_rate

        h0 = scipy.signal.fftconvolve(k0, self._source_data)
        h1 = scipy.signal.fftconvolve(k1, self._source_data)

        sp = np.pad(np.array(self._source_data), (0, len(h1)-len(self._source_data)))
        x = sp[:len(mixer)] + h1[:len(mixer)] * mixer + h0[:len(mixer)] * np.abs(1-mixer)

        return x, {
            'd0': d0,
            'd1': d1,
            'l': secret_len,
        }

    def encode(
            self,
            d0: Optional[int] = None,
            d1: Optional[int] = None,
            delay_search = '',
        ) -> Tuple[np.ndarray, Dict[str, Any]]:

        if d0 is None:
            d0 = 150
        if d1 is None:
            d1 = d0 + 50

        if delay_search == 'bruteforce':
            return self._encode_bruteforce(d0, d1)

        if delay_search == 'basinhopping':
            return self._encode_basinhopping(d0, d1)

        return self._encode(d0, d1)

    def _encode_bruteforce(
            self,
            d0: int,
            d1: int,
        ) -> Tuple[np.ndarray, Dict[str, Any]]:

        encoded = np.empty(0)
        for d0 in range(d0, d0+100):
            for d1 in range(d0, d0+50):

                encoded, params = self._encode(d0, d1)

                if np.abs(encoded).max() == 0:
                    continue

                # Decode to verify delay pair
                test_decoder = Echo_single_kernel(encoded)
                if np.all(test_decoder.decode(d0, d1, params['l'])[0] == self._secret_data):
                    return encoded, {
                        'd0': d0,
                        'd1': d1,
                        'l': params['l'],
                    }

        return encoded, {
            'd0': -1,
            'd1': -1,
            'l': -1,
        }

    def _optimize_encode(self, x):
        encoded, params = self._encode(int(x[0]), int(x[1]))
        test_decoder = Echo_single_kernel(encoded)
        ber = ber_percent(test_decoder.decode(**params)[0], self._secret_data)
        return ber

    def _encode_basinhopping(
            self,
            d0: int,
            d1: int,
        ) -> Tuple[np.ndarray, Dict[str, Any]]:

        def bounds(**kwargs):
            x = kwargs['x_new']
            return x[0] < x[1]

        res = scipy.optimize.basinhopping(
            func=self._optimize_encode,
            x0=[d0, d1],
            niter=100,
            T=3.0,
            stepsize=8.0,
            accept_test=bounds,
            callback=lambda x, f, accept: True if f == 0.0 else None,
        )

        secret_len = len(self._secret_data)

        x, _ = self._encode(int(res.x[0]), int(res.x[1]))
        return x, {
            'd0': int(res.x[0]),
            'd1': int(res.x[1]),
            'l': secret_len,
        }

    def set_alpha(self, alpha):
        self._alpha = alpha

    def set_decay_rate(self, decay_rate):
        self._decay_rate = decay_rate


    def decode(self, d0: int, d1: int, l: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        split = seg_split(self._source_data, l+1)[:-1]
        decoded = np.zeros(len(split), dtype=int)
        i = 0
        for seg in split:
            cn = np.fft.ifft(np.log(np.abs(np.fft.fft(seg))))
            if cn[d0+1] > cn[d1+1]:
                decoded[i] = 0
            else:
                decoded[i] = 1
            i += 1

        return decoded, {}

    @staticmethod
    def get_encode_args() -> List[Tuple[List, Dict]]:
        args = []
        args.append((['-d0'],
                     {
                         'action': 'store',
                         'type': int,
                         'default': None,
                     }))
        args.append((['-d1'],
                     {
                         'action': 'store',
                         'type': int,
                         'default': None,
                     }))
        args.append((['--delay_search'],
                     {
                         'action': 'store',
                         'type': str,
                         'default': '',
                         'choices': ['bruteforce', 'basinhopping'],
                     }))
        return args

    @staticmethod
    def get_decode_args() -> List[Tuple[List, Dict]]:
        args = []
        args.append((['-d0'],
                     {
                         'action': 'store',
                         'type': int,
                         'required': True
                     }))
        args.append((['-d1'],
                     {
                         'action': 'store',
                         'type': int,
                         'required': True
                     }))
        args.append((['-l', '--len'],
                     {
                         'action': 'store',
                         'type': int,
                         'required': True,
                         'help': 'encoded data length'
                     }))
        return args
