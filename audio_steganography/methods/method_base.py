from ..mode import Mode
import numpy as np

class MethodBase:
    def __init__(self, cover: np.ndarray, data: np.ndarray, mode: Mode):
        self.cover = cover
        if mode == Mode.encode:
            self.data_to_encode = data
        else:
            self.data_to_decode = data

    def encode(self) -> np.ndarray:
        raise NotImplementedError('Inherited steganography method must implement this function')

    def decode(self) -> np.ndarray:
        raise NotImplementedError('Inherited steganography method must implement this function')
