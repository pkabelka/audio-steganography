# -*- coding: utf-8 -*-

# File: method_base.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the abstract class MethodBase which all methods inherit
"""

import typing
import abc
import numpy as np

class MethodBase(abc.ABC):
    """All method classes must inherit this class and implement the encode and
    decode method.

    Custom arguments for encoding and decoding can be specified by overriding
    `get_encode_args` and `get_decode_args` methods.
    """
    def __init__(self, source_data: np.ndarray):
        self._source_data = source_data
        self._secret_data = np.empty(0, dtype=np.uint8)

    @abc.abstractmethod
    def encode(self) -> typing.Tuple[np.ndarray, typing.Dict[str, typing.Any]]:
        raise NotImplementedError('Inherited steganography method must implement this function')

    @abc.abstractmethod
    def decode(self) -> typing.Tuple[np.ndarray, typing.Dict[str, typing.Any]]:
        raise NotImplementedError('Inherited steganography method must implement this function')

    @staticmethod
    def get_encode_args() -> typing.List[typing.Tuple[typing.List, typing.Dict]]:
        return []

    @staticmethod
    def get_decode_args() -> typing.List[typing.Tuple[typing.List, typing.Dict]]:
        return []

    def set_secret_data(self, secret_data: np.ndarray[typing.Any, np.dtype[np.uint8]]):
        self._secret_data = secret_data
