# -*- coding: utf-8 -*-

# File: echo_base.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the EchoBase class
"""

from .method_base import MethodBase, EncodeDecodeReturn, EncodeDecodeArgsReturn
from ..exceptions import SecretSizeTooLarge
from ..audio_utils import split_to_n_segments, spread_bits, to_dtype
from typing import Optional
import numpy as np
import abc

from ..stat_utils import ber_percent
import scipy.optimize

class EchoBase(MethodBase, abc.ABC):
    """This is an abstract base class of echo hiding methods.
    """

    @abc.abstractmethod
    def _encode(
            self,
            d0: int,
            d1: int,
            alpha = 0.5,
            decay_rate = 0.85,
        ) -> EncodeDecodeReturn:
        """Inherited steganography method must implement this function"""

    def encode_wrapper(
            self,
            echo_class,
            d0: Optional[int] = None,
            d1: Optional[int] = None,
            delay_search = '',
            alpha = 0.5,
            decay_rate = 0.85,
            **kwargs,
        ) -> EncodeDecodeReturn:
        """This method wraps the concrete encode methods of the `echo_class` to
        simplify reuse of the `_encode_buteforce` and `_encode_basinhopping`.

        Encodes the secret data into source. If the returned d0, d1 and l
        are all -1, then the method failed to encode correctly.

        If the secret data is bigger than source capacity, a
        `SecretSizeTooLarge` exception is raised.

        Parameters
        ----------
        echo_class : EchoBase
            Concrete echo method class. It must implement the `_encode` method.
        d0 : int | None
            Echo delay for binary 0. Default value is 150. Must be larger than
            0 otherwise raises ValueError exception.
        d1 : int | None
            Echo delay for binary 1. Default value is d0 + 50. Must be larger
            than 0 and d0 otherwise raises ValueError exception.
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

        if d0 is None:
            d0 = 150
        if d1 is None:
            d1 = d0 + 50

        if d0 >= d1:
            raise ValueError('d0 must be smaller than d1')
        if d0 <= 0 or d1 <= 0:
            raise ValueError('d0 and d1 must be larger than 0')

        # require at least 1024 samples per encoded bit
        if self._secret_data.size * 1024 > self._source_data.size:
            raise SecretSizeTooLarge('secret data cannot fit in source: '+
                f'len(secret) = {self._secret_data.size}, capacity(source) = '+
                f'{self._source_data.size//1024}')

        if delay_search == 'bruteforce':
            return self._encode_bruteforce(echo_class, d0, d1, alpha, decay_rate)

        elif delay_search == 'basinhopping':
            return self._encode_basinhopping(echo_class, d0, d1, alpha, decay_rate)

        return self._encode(d0, d1, alpha, decay_rate)

    def _encode_bruteforce(
            self,
            echo_class,
            d0: int,
            d1: int,
            alpha: float,
            decay_rate: float,
        ) -> EncodeDecodeReturn:

        encoded = np.empty(0)
        params_hist = {}

        _d0 = d0
        _d1 = d1
        for d0 in range(_d0, _d0+10):
            for d1 in range(_d1, _d1+30):
                if d0 >= d1:
                    continue

                encoded, params = self._encode(d0, d1, alpha, decay_rate)

                if np.abs(encoded).max() == 0:
                    continue

                # Decode to verify delay pair
                test_decoder = echo_class(encoded)
                ber = ber_percent(test_decoder.decode(**params)[0], self._secret_data)
                if ber == 0.0:
                    return encoded, {
                        'd0': d0,
                        'd1': d1,
                        'l': params['l'],
                    }
                else:
                    params_hist[(d0, d1, params['l'])] = ber

        # Encode using best delays found
        best_params = min(params_hist, key=params_hist.get)
        encoded, params = self._encode(best_params[0], best_params[1])

        return encoded, {
            'd0': params['d0'],
            'd1': params['d1'],
            'l': params['l'],
        }


    def _encode_basinhopping(
            self,
            echo_class,
            d0: int,
            d1: int,
            alpha: float,
            decay_rate: float,
        ) -> EncodeDecodeReturn:

        def optimize_encode(x):
            encoded, params = self._encode(int(x[0]), int(x[1]), alpha, decay_rate)
            test_decoder = echo_class(encoded)
            ber = ber_percent(test_decoder.decode(**params)[0], self._secret_data)
            return ber

        def take_step(x):
            x[0] = np.random.randint(max(1, x[0]-5), x[0]+10)
            x[1] = np.random.randint(max(x[0]+1, x[1]-5), max(x[0]+2, x[1]+10))

            return x

        res = scipy.optimize.basinhopping(
            func=optimize_encode,
            x0=[d0, d1],
            niter=100,
            take_step=take_step,
            callback=lambda x, f, accept: True if f == 0.0 else None,
        )

        secret_len = self._secret_data.size

        x, _ = self._encode(int(res.x[0]), int(res.x[1]), alpha, decay_rate)
        return x, {
            'd0': int(res.x[0]),
            'd1': int(res.x[1]),
            'l': secret_len,
        }


    @staticmethod
    def get_encode_args() -> EncodeDecodeArgsReturn:
        args = []
        args.append((['-d0'],
                     {
                         'action': 'store',
                         'type': int,
                         'default': None,
                         'help': 'offset for binary 0; default 150 samples',
                     }))
        args.append((['-d1'],
                     {
                         'action': 'store',
                         'type': int,
                         'default': None,
                         'help': 'offset for binary 1; default d0 + 50 samples',
                     }))
        args.append((['-a', '--alpha'],
                     {
                         'action': 'store',
                         'type': float,
                         'required': False,
                         'default': 0.5,
                         'help': 'echo amplitude multiplier; default: 0.5',
                     }))
        args.append((['-dr', '--decay_rate'],
                     {
                         'action': 'store',
                         'type': float,
                         'required': False,
                         'default': 0.85,
                         'help': 'second echo amplitude multiplier; default: 0.85',
                     }))
        args.append((['--delay_search'],
                     {
                         'action': 'store',
                         'type': str,
                         'default': '',
                         'choices': ['bruteforce', 'basinhopping'],
                         'help': 'method for searching offsets resulting in '+
                             '0 bit error rate',
                     }))
        return args

    @staticmethod
    def get_decode_args() -> EncodeDecodeArgsReturn:
        args = []
        args.append((['-d0'],
                     {
                         'action': 'store',
                         'type': int,
                         'required': True,
                     }))
        args.append((['-d1'],
                     {
                         'action': 'store',
                         'type': int,
                         'required': True,
                     }))
        args.append((['-l', '--len'],
                     {
                         'action': 'store',
                         'type': int,
                         'required': True,
                         'help': 'number of encoded bits',
                     }))
        return args
