# -*- coding: utf-8 -*-

# File: audio_utils.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains audio utility functions.
"""

import numpy as np
import scipy.signal
import scipy.interpolate
from numpy.typing import DTypeLike
from typing import List, Tuple, Union

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
        return np.empty(0), np.empty(0)
    return np.asanyarray(
        np.array_split(input[:int(np.floor(len(input)/n) * n)], int(len(input)/n))
    ), input[int(np.floor(len(input)/n) * n):]

def spread_bits(secret_data: np.ndarray, signal_length: int) -> np.ndarray:
    """Spreads the `secret_data` over the size of `signal_length`.

    Parameters
    ----------
    secret_data : numpy.ndarray
        Secret data bits array.
    signal_length : int
        Length of signal.

    Returns
    -------
    out : numpy.ndarray
        Array `secret_data` spread to `signal_length` size.
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

def autocorr_scipy_correlate(x: Union[List, np.ndarray]):
    """Computes autocorrelation of signal x.

    This function should be faster than numpy.correlate because it
    automatically chooses between direct and Fourier method.

    Parameters
    ----------
    x : List | ArrayLike
        A 1-D signal to autocorrelate.
    """
    result = scipy.signal.correlate(x, x, mode='full')
    return result[len(result)//2:]

def center(input: np.ndarray) -> np.ndarray:
    """Centers input array i.e. removes the DC component.

    Parameters
    ----------
    input : numpy.ndarray
        Input array to center.

    Returns
    -------
    out : numpy.ndarray
        Centered array.
    """
    return input - np.mean(input)

def normalize(input: np.ndarray) -> np.ndarray:
    """Normalizes input array to [-1; 1] range.

    Parameters
    ----------
    input : numpy.ndarray
        Input array to normalize.

    Returns
    -------
    out : numpy.ndarray
        Normalized array.
    """
    if len(input) > 0 and np.abs(input).max() != 0:
        input = input / np.abs(input).max()
    return input

def consecutive_values(input: Union[List, np.ndarray]):
    """Returns starting indices and lengths of consecutive values.

    Parameters
    ----------
    input : List | NDArray
        1D array of values.

    Returns
    -------
    indices : NDArray
        NumPy array of indices where consecutive values start
    lengths : NDArray
        NumPy array of lengths of consecutive values

    ---
    This function is a modified version from a StackOverflow answer:
    https://stackoverflow.com/a/4652265
    by Paul (https://stackoverflow.com/users/31676/paul)
    edited by Peter Mortensen (https://stackoverflow.com/users/63550/peter-mortensen)

    This function is under CC BY-SA 3.0 License:
    https://creativecommons.org/licenses/by-sa/3.0

    THE WORK (AS DEFINED BELOW) IS PROVIDED UNDER THE TERMS OF THIS CREATIVE
    COMMONS PUBLIC LICENSE ("CCPL" OR "LICENSE"). THE WORK IS PROTECTED BY
    COPYRIGHT AND/OR OTHER APPLICABLE LAW. ANY USE OF THE WORK OTHER THAN AS
    AUTHORIZED UNDER THIS LICENSE OR COPYRIGHT LAW IS PROHIBITED.

    BY EXERCISING ANY RIGHTS TO THE WORK PROVIDED HERE, YOU ACCEPT AND AGREE TO
    BE BOUND BY THE TERMS OF THIS LICENSE. TO THE EXTENT THIS LICENSE MAY BE
    CONSIDERED TO BE A CONTRACT, THE LICENSOR GRANTS YOU THE RIGHTS CONTAINED
    HERE IN CONSIDERATION OF YOUR ACCEPTANCE OF SUCH TERMS AND CONDITIONS.

    UNLESS OTHERWISE MUTUALLY AGREED TO BY THE PARTIES IN WRITING, LICENSOR
    OFFERS THE WORK AS-IS AND MAKES NO REPRESENTATIONS OR WARRANTIES OF ANY
    KIND CONCERNING THE WORK, EXPRESS, IMPLIED, STATUTORY OR OTHERWISE,
    INCLUDING, WITHOUT LIMITATION, WARRANTIES OF TITLE, MERCHANTIBILITY,
    FITNESS FOR A PARTICULAR PURPOSE, NONINFRINGEMENT, OR THE ABSENCE OF LATENT
    OR OTHER DEFECTS, ACCURACY, OR THE PRESENCE OF ABSENCE OF ERRORS, WHETHER
    OR NOT DISCOVERABLE. SOME JURISDICTIONS DO NOT ALLOW THE EXCLUSION OF
    IMPLIED WARRANTIES, SO SUCH EXCLUSION MAY NOT APPLY TO YOU.

    Citation date: 2023-02-23
    """
    input = np.asanyarray(input)

    if input.size == 0:
        return np.array([]), np.array([])

    diff = np.ones(input.shape, dtype=bool)
    diff[1:] = np.diff(input)
    return np.nonzero(diff)[0], np.diff(np.append(np.nonzero(diff)[0], input.size))

def resample(
        input: Union[List, np.ndarray],
        factor: float,
        kind='linear',
    ) -> np.ndarray:
    """Resamples the input array with the given factor.

    Parameters
    ----------
    input : List | NDArray
        1D array of values.
    factor : List | NDArray
        Resampling factor.
    kind : List | NDArray
        Interpolation kind.

    Returns
    -------
    out : NDArray
        NumPy array with the input resampled with the given factor.

    ---
    This function is a modified version from a StackOverflow answer:
    https://stackoverflow.com/a/55747293
    by fabda01 (https://stackoverflow.com/users/6327658/fabda01)
    edited 2019-04-19 13:08

    This function is under CC BY-SA 4.0 License:
    https://creativecommons.org/licenses/by-sa/4.0

    By exercising the Licensed Rights (defined below), You accept and agree to
    be bound by the terms and conditions of this Creative Commons
    Attribution-ShareAlike 4.0 International Public License ("Public License").
    To the extent this Public License may be interpreted as a contract, You are
    granted the Licensed Rights in consideration of Your acceptance of these
    terms and conditions, and the Licensor grants You such rights in
    consideration of benefits the Licensor receives from making the Licensed
    Material available under these terms and conditions.

    Unless otherwise separately undertaken by the Licensor, to the extent
    possible, the Licensor offers the Licensed Material as-is and as-available,
    and makes no representations or warranties of any kind concerning the
    Licensed Material, whether express, implied, statutory, or other. This
    includes, without limitation, warranties of title, merchantability, fitness
    for a particular purpose, non-infringement, absence of latent or other
    defects, accuracy, or the presence or absence of errors, whether or not
    known or discoverable. Where disclaimers of warranties are not allowed in
    full or in part, this disclaimer may not apply to You.

    To the extent possible, in no event will the Licensor be liable to You on
    any legal theory (including, without limitation, negligence) or otherwise
    for any direct, special, indirect, incidental, consequential, punitive,
    exemplary, or other losses, costs, expenses, or damages arising out of this
    Public License or use of the Licensed Material, even if the Licensor has
    been advised of the possibility of such losses, costs, expenses, or
    damages. Where a limitation of liability is not allowed in full or in part,
    this limitation may not apply to You.

    Citation date: 2023-03-20
    """
    input = np.asanyarray(input)

    if input.size == 0:
        return np.empty(0)

    n = int(np.ceil(input.size / factor))
    f = scipy.interpolate.interp1d(np.linspace(0, 1, input.size), input, kind)
    return f(np.linspace(0, 1, n))

def add_normalized_noise(input: np.ndarray, wanted_snr_db: float):
    """Adds noise with requested dB and normalizes the signal to [-1; 1] range.

    Parameters
    ----------
    input : numpy.ndarray
        Input array to add noise to.
    wanted_snr_db : float
        Signal to noise ratio of the noise to add in decibells.

    Returns
    -------
    out : numpy.ndarray
        Input array with added noise.

    ---
    This function is a modified version from a StackOverflow answer:
    https://stackoverflow.com/a/53688043
    by tmcdevitt (https://stackoverflow.com/users/6321443/tmcdevitt)
    edited 2019-09-24 15:14

    This function is under CC BY-SA 4.0 License:
    https://creativecommons.org/licenses/by-sa/4.0

    By exercising the Licensed Rights (defined below), You accept and agree to
    be bound by the terms and conditions of this Creative Commons
    Attribution-ShareAlike 4.0 International Public License ("Public License").
    To the extent this Public License may be interpreted as a contract, You are
    granted the Licensed Rights in consideration of Your acceptance of these
    terms and conditions, and the Licensor grants You such rights in
    consideration of benefits the Licensor receives from making the Licensed
    Material available under these terms and conditions.

    Unless otherwise separately undertaken by the Licensor, to the extent
    possible, the Licensor offers the Licensed Material as-is and as-available,
    and makes no representations or warranties of any kind concerning the
    Licensed Material, whether express, implied, statutory, or other. This
    includes, without limitation, warranties of title, merchantability, fitness
    for a particular purpose, non-infringement, absence of latent or other
    defects, accuracy, or the presence or absence of errors, whether or not
    known or discoverable. Where disclaimers of warranties are not allowed in
    full or in part, this disclaimer may not apply to You.

    To the extent possible, in no event will the Licensor be liable to You on
    any legal theory (including, without limitation, negligence) or otherwise
    for any direct, special, indirect, incidental, consequential, punitive,
    exemplary, or other losses, costs, expenses, or damages arising out of this
    Public License or use of the Licensed Material, even if the Licensor has
    been advised of the possibility of such losses, costs, expenses, or
    damages. Where a limitation of liability is not allowed in full or in part,
    this limitation may not apply to You.

    Citation date: 2023-03-21
    """
    signal = center(input)
    signal = normalize(signal)
    avg_db = 10 * np.log10(np.mean(signal**2))
    noise_power = 10 ** ((avg_db - wanted_snr_db) / 10)
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise
