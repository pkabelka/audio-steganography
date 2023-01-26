# -*- coding: utf-8 -*-

# File: stat_utils.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains statistic utility functions
"""

import numpy as np
from typing import List, Union

def mse(x: Union[np.ndarray, List], y: Union[np.ndarray, List]) -> float:
    """Calculates the mean square error of clean signal and noisy signal.

    Parameters
    ----------
    x : np.ndarray | List
        Clean signal array.
    y : np.ndarray | List
        Noisy signal array.

    Returns
    -------
    mse : float
        Mean square error value.
    """
    x = np.asanyarray(x)
    y = np.asanyarray(y)

    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]

    if np.abs(x).max() != 0:
        x = x / np.abs(x).max()

    if np.abs(y).max() != 0:
        y = y / np.abs(y).max()

    return float(1/n * np.sum((x - y)**2))

def rmsd(x: Union[np.ndarray, List], y: Union[np.ndarray, List]) -> float:
    """Calculates the root mean square deviation of clean signal and noisy
    signal.

    Parameters
    ----------
    x : np.ndarray | List
        Clean signal array.
    y : np.ndarray | List
        Noisy signal array.

    Returns
    -------
    rmsd : float
        Root mean square deviation value.
    """
    x = np.asanyarray(x)
    y = np.asanyarray(y)

    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]

    return np.sqrt(mse(x, y))

def rms_db(x: Union[np.ndarray, List]) -> float:
    """Calculates the decibell root mean square sum of the input signal.

    Parameters
    ----------
    x : np.ndarray | List
        Signal array.

    Returns
    -------
    rms_db : float
        Root mean square sum in decibells.
    """
    x = np.asanyarray(x)
    return 10 * np.log10(mse(x, np.zeros_like(x)))

def snr_db(x: Union[np.ndarray, List], y: Union[np.ndarray, List]) -> float:
    """Calculates the decibell signal-to-noise ratio of clean signal and noisy
    signal.

    Parameters
    ----------
    x : np.ndarray | List
        Clean signal array.
    y : np.ndarray | List
        Noisy signal array.

    Returns
    -------
    snr : float
        Signal-to-noise ratio value in decibells.
    """
    x = np.asanyarray(x)
    y = np.asanyarray(y)

    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]

    if np.abs(x).max() != 0:
        x = x / np.abs(x).max()

    if np.abs(y).max() != 0:
        y = y / np.abs(y).max()

    return 10 * np.log10((np.sum(x**2)) / (np.sum((x - y)**2)))

def psnr_db(x: Union[np.ndarray, List], y: Union[np.ndarray, List]) -> float:
    """Calculates the peak signal-to-noise ratio of clean signal and noisy
    signal in decibells.

    Parameters
    ----------
    x : np.ndarray | List
        Clean signal array.
    y : np.ndarray | List
        Noisy signal array.

    Returns
    -------
    psnr : float
        Peak signal-to-noise ratio value in decibells.
    """
    x = np.asanyarray(x)
    y = np.asanyarray(y)

    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]

    return 10 * np.log10(1.0 / mse(x, y))

def ber_percent(
        x: Union[np.ndarray, List],
        y: Union[np.ndarray, List],
    ) -> float:
    """Calculates the bit error rate of between the two input arrays.

    Parameters
    ----------
    x : np.ndarray | List
        First input array.
    y : np.ndarray | List
        Second input array.

    Returns
    -------
    ber : int
        Bit error rate between the two input arrays.
    """
    x = np.asanyarray(x)
    y = np.asanyarray(y)

    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]

    return float((x != y).sum() / n) * 100
