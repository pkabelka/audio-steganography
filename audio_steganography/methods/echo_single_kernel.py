# -*- coding: utf-8 -*-

# File: echo_single_kernel.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the Echo_single_kernel class
"""

from .method_base import MethodBase
from ..audio_utils import seg_split
from typing import Tuple, Dict, List, Any
import numpy as np
import scipy.signal

class Echo_single_kernel(MethodBase):
    def encode(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        secret_len = len(self._secret_data)
        mixer = seg_split(np.ones(len(self._source_data)), secret_len + 1)
        # print('Mixer len:', len(mixer))
        # print('Seg sample len:', len(mixer[0]))

        for i in range(len(self._secret_data)):
            mixer[i] = mixer[i] * self._secret_data[i]

        mixer = np.hstack(mixer)
        # print(mixer)
        # print('Mixer len:', len(mixer))

        delay_pairs = []
        end = False
        x = np.empty(0)
        alpha = 0.5
        decay_rate = 0.85
        for d0 in range(150, 250):
            # echo kernel for binary 0
            k0 = np.append(np.zeros(d0), [1]) * alpha
            for d1 in range(d0, d0+50):

                # echo kernel for binary 1
                k1 = np.append(np.zeros(d1), [1]) * alpha * decay_rate

                h0 = scipy.signal.fftconvolve(k0, self._source_data)
                h1 = scipy.signal.fftconvolve(k1, self._source_data)

                sp = np.pad(np.array(self._source_data), (0, len(h1)-len(self._source_data)))
                x = sp[:len(mixer)] + h1[:len(mixer)] * mixer + h0[:len(mixer)] * np.abs(1-mixer)

                if np.abs(x).max() == 0:
                    continue

                # Decode to verify delay pair
                test_decoder = Echo_single_kernel(x)
                if np.all(test_decoder.decode(d0, d1, secret_len)[0] == self._secret_data):
                    delay_pairs.append((d0, d1))
                    end = True
                    break

            if end:
                break

        return x, {
            'd0': delay_pairs[0][0],
            'd1': delay_pairs[0][1],
            'l': secret_len,
        }


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
