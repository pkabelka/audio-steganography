# -*- coding: utf-8 -*-

# File: method_base.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the abstract class MethodBase which all methods inherit
"""

import abc
from typing import Tuple, Dict, List, Any
import numpy as np

class MethodBase(abc.ABC):
    """All method classes must inherit this class and implement the encode and
    decode method.

    For encoding, the called must use `set_secret_data` method to set the data
    to encode into the source.

    Custom arguments for encoding and decoding can be specified by overriding
    `get_encode_args` and `get_decode_args` methods.
    """
    def __init__(self, source_data: np.ndarray):
        self._source_data = source_data
        self._secret_data = np.empty(0, dtype=np.uint8)

    @abc.abstractmethod
    def encode(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError('Inherited steganography method must implement this function')

    @abc.abstractmethod
    def decode(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError('Inherited steganography method must implement this function')

    @staticmethod
    def get_encode_args() -> List[Tuple[List, Dict]]:
        return []

    @staticmethod
    def get_decode_args() -> List[Tuple[List, Dict]]:
        return []

    def set_secret_data(self, secret_data: np.ndarray[Any, np.dtype[np.uint8]]):
        """Setter for secret_data array.

        If not used, secret_data will be an empty NumPy array with dtype uint8.

        Parameters
        ----------
        secret_data : numpy.ndarray
            The data to encode into the source.
        """
        self._secret_data = secret_data
