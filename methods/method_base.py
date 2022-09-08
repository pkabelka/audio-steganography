import numpy as np

class MethodBase:
    def __init__(self, cover: np.ndarray, data_to_encode: np.ndarray):
        self.cover = cover
        self.data_to_encode = data_to_encode

    def encode(self) -> np.ndarray:
        raise NotImplementedError('Inherited steganography method must implement this function')

    def decode(self) -> np.ndarray:
        raise NotImplementedError('Inherited steganography method must implement this function')
