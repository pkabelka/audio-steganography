# -*- coding: utf-8 -*-

# File: audio_utils.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains audio utility functions.
"""

import numpy as np
from numpy.typing import DTypeLike
from typing import List

def seg_split(input: np.ndarray, n: int) -> List[np.ndarray]:
    """Splits the input array into N segments of approximately same length.

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

def seg_split_len_n(input: np.ndarray, n: int) -> List[np.ndarray]:
    """Splits the input array into segments of approximately length N.

    Parameters
    ----------
    input : numpy.ndarray
        The array to split into N segments.
    n : int
        Segment length.

    Returns
    -------
    out : List
        A list containing input split into NumPy arrays of length N.
    """
    if len(input) == 0:
        return [np.empty(0)]
    return np.array_split(input, np.ceil(len(input) / n))

def seg_split_len_n_except_last(input: np.ndarray, n: int) -> List[np.ndarray]:
    """Splits the input array into segments of length N. Last segment contains
    the remainder of the input array.

    Parameters
    ----------
    input : numpy.ndarray
        The array to split into N segments.
    n : int
        Segment length.

    Returns
    -------
    out : List
        A list containing input split into NumPy arrays of length N.
    """
    if len(input) == 0:
        return [np.empty(0)]
    return np.split(input, np.arange(n, len(input), n))

def seg_split_same_len_except_last(input: np.ndarray, n: int) -> List[np.ndarray]:
    """Splits the input array into N segments of same length except last
    segment. Last segment contains the remainder of the input array.

    Parameters
    ----------
    input : numpy.ndarray
        The array to split into N segments.
    n : int
        Number of segments to split input into.

    Returns
    -------
    out : List
        A list containing input split into N NumPy arrays of same length. Last
        array contains the remainder of input array.
    """
    if len(input) == 0:
        return [np.empty(0)]
    return np.array_split(
            input[:int(np.floor(len(input)/(n-1))) * (n-1)],
            n-1) + [input[int(np.floor(len(input)/(n-1))) * (n-1):]]

def mixer_sig(secret_data: np.ndarray, signal_length: int) -> np.ndarray:
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
    mixer = seg_split_same_len_except_last(np.ones(signal_length), secret_len + 1)

    for i in range(len(secret_data)):
        mixer[i] = mixer[i] * secret_data[i]

    return np.hstack(mixer)

def to_dtype(input: np.ndarray, dtype: DTypeLike) -> np.ndarray:
    """Converts values in input array to specified dtype values.

    The values in `input` are converted to `numpy.float64` dtype and then
    converted to the dtype specified in `dtype`.

    Parameters
    ----------
    input : numpy.ndarray
        Input array to convert.
    dtype : numpy.dtype
        Output array values dtype.

    Returns
    -------
    out : numpy.ndarray
        Array with values converted to specified dtype.
    """

    input = np.asanyarray(input)
    max = 1.0
    try:
        max = np.iinfo(input.dtype).max
    except ValueError:
        # already float type
        pass
    out = input / max
    out = out.astype(np.float64)

    max = 1.0
    try:
        max = np.iinfo(dtype).max
    except ValueError:
        # already float type
        pass
    out = out * max
    out = out.astype(dtype)
    return out
