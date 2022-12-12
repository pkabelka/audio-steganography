# -*- coding: utf-8 -*-

# File: method_base.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the abstract class MethodBase which all methods inherit
"""

import abc
from typing import Tuple, Dict, List, Any
import numpy as np

EncodeDecodeReturn = Tuple[np.ndarray, Dict[str, Any]]
EncodeDecodeArgsReturn = List[Tuple[List, Dict]]

class MethodBase(abc.ABC):
    """All method classes must inherit this class and implement the encode and
    decode method. For correct function in CLI mode, the method needs to be
    added in the `Method` enum.

    Custom arguments for encoding and decoding can be specified by overriding
    `get_encode_args` and `get_decode_args` methods.
    """
    def __init__(
            self,
            source_data: np.ndarray,
            secret_data = np.empty(0, dtype=np.uint8)
        ):

        if secret_data.dtype != np.uint8:
            raise TypeError("secret_data must be of type numpy.uint8")

        self._source_data = source_data
        self._secret_data = secret_data

    @abc.abstractmethod
    def encode(self, **kwargs) -> EncodeDecodeReturn:
        raise NotImplementedError('Inherited steganography method must implement this function')

    @abc.abstractmethod
    def decode(self, **kwargs) -> EncodeDecodeReturn:
        raise NotImplementedError('Inherited steganography method must implement this function')

    @staticmethod
    def get_encode_args() -> EncodeDecodeArgsReturn:
        return []

    @staticmethod
    def get_decode_args() -> EncodeDecodeArgsReturn:
        return []
