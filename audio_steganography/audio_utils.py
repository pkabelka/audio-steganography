# -*- coding: utf-8 -*-

# File: audio_utils.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains audio utility functions.
"""

import numpy as np
from numpy.typing import DTypeLike
from typing import List, Tuple

def split_to_n_approx_same(input: np.ndarray, n: int) -> List[np.ndarray]:
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

def split_to_n_same_except_last(input: np.ndarray, n: int) -> List[np.ndarray]:
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

def split_to_n_segments(
        input: np.ndarray, n: int
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Splits the input array into N segments of same length and also returns
    the unused part of array.

    Parameters
    ----------
    input : numpy.ndarray
        The array to split into N segments.
    n : int
        Number of segments to split input into.

    Returns
    -------
    out : NDArray[...]
        A NumPy array with input split into N NumPy arrays of same length.
    rest : NDArray
        Rest of the input which had to be sliced off for alignment.
    """
    return np.asanyarray(
        np.array_split(input[:int(np.floor(len(input)/n) * n)], n)
    ), input[int(np.floor(len(input)/n) * n):]

def split_to_segments_of_approx_len_n(input: np.ndarray, n: int) -> List[np.ndarray]:
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

def split_to_segments_of_len_n_except_last(input: np.ndarray, n: int) -> List[np.ndarray]:
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

def split_to_segments_of_len_n(input: np.ndarray, n: int):
    """Splits the input array into segments of exactly length N and also
    returns the unused part of array.

    Parameters
    ----------
    input : numpy.ndarray
        The array to split into segments of length N.
    n : int
        Segment length.

    Returns
    -------
    out : NDArray
        A list containing input split into NumPy arrays of exactly length N.
    rest : NDArray
        Rest of the input which had to be sliced off for alignment.
    """
    if len(input) == 0:
        return [np.empty(0)]
    return np.asanyarray(
        np.array_split(input[:int(np.floor(len(input)/n) * n)], int(len(input)/n))
    ), input[int(np.floor(len(input)/n) * n):]

def mixer_sig(secret_data: np.ndarray, signal_length: int) -> np.ndarray:
    """Creates a mixer signal by spliting the input array into segments of
    secret_data length and multiplying each array with each bit in secret_data.

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
    mixer, rest = split_to_n_segments(np.ones(signal_length), len(secret_data))
    mixer = mixer * secret_data[:, None]
    return np.append(mixer, rest)

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
