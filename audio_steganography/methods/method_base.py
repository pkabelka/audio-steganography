from ..mode import Mode
import typing
import abc
import numpy as np

class MethodBase(abc.ABC):
    def __init__(self, data: np.ndarray, mode: Mode):
        self.cover_data = np.empty(0)
        if mode == Mode.encode:
            self.data_to_encode = data
        else:
            self.data_to_decode = data

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

    def set_cover_data(self, cover_data: np.ndarray):
        self.cover_data = cover_data
