# -*- coding: utf-8 -*-

# File: audio_utils.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains audio utility functions.
"""

import numpy as np
from typing import List, Any

def seg_split(input: np.ndarray, n: int) -> List[np.ndarray]:
    """Splits the input array into N segments

    Parameters
    ----------
    input : numpy.ndarray
        The array to split into N segments.
    n : int
        Number of segments to split input into.

    Returns
    -------
    out : List
        A list containing input split into N NumPy arrays.
    """
    return np.array_split(input, n)[:-1] + [input[-int(round(len(input)/n)):]]

def mixer_sig(secret_data: np.ndarray[Any, np.dtype[np.uint8]], signal_length: int) -> np.ndarray:
    """Creates a mixer signal by spliting the input array into segments of
    secret_data length + 1.

    Parameters
    ----------
    secret_data : numpy.ndarray
        Secret data bits array.
    signal_length : int
        Length of signal.

    Returns
    -------
    out : numpy.ndarray
        Mixer signal of `signal_length` length.
    """
    secret_len = len(secret_data)
    mixer = seg_split(np.ones(signal_length), secret_len + 1)

    for i in range(len(secret_data)):
        mixer[i] = mixer[i] * secret_data[i]

    return np.hstack(mixer)
