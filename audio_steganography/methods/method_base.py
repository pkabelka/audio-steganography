import typing
import abc
import numpy as np

class MethodBase(abc.ABC):
    def __init__(self, source_data: np.ndarray):
        self._source_data = source_data
        self._secret_data = np.empty(0)

    @abc.abstractmethod
    def encode(self) -> np.ndarray:
        raise NotImplementedError('Inherited steganography method must implement this function')

    @abc.abstractmethod
    def decode(self) -> np.ndarray:
        raise NotImplementedError('Inherited steganography method must implement this function')

    @staticmethod
    def get_encode_args() -> typing.List[typing.Tuple[typing.List, typing.Dict]]:
        return []

    @staticmethod
    def get_decode_args() -> typing.List[typing.Tuple[typing.List, typing.Dict]]:
        return []

    def set_secret_data(self, secret_data: np.ndarray):
        self._secret_data = secret_data
