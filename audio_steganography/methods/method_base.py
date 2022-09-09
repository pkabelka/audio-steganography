from ..mode import Mode
import numpy as np

class MethodBase:
    def __init__(self, data: np.ndarray, mode: Mode):
        self.cover_data = np.empty(0)
        if mode == Mode.encode:
            self.data_to_encode = data
        else:
            self.data_to_decode = data

    def encode(self) -> np.ndarray:
        raise NotImplementedError('Inherited steganography method must implement this function')

    def decode(self) -> np.ndarray:
        raise NotImplementedError('Inherited steganography method must implement this function')

    def set_cover_data(self, cover_data: np.ndarray):
        self.cover_data = cover_data
