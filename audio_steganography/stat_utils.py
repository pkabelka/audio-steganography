# -*- coding: utf-8 -*-

# File: stat_utils.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains statistic utility functions
"""

import numpy as np
from typing import List, Any

def snr_db(x: np.ndarray | List, y: np.ndarray | List) -> float:
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
    n = len(x)
    if n != len(y):
        raise
    return 10 * np.log10((np.sum(x**2)) / (np.sum((x - y)**2)))

def mse(x: np.ndarray | List, y: np.ndarray | List) -> float:
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
    n = len(x)
    if n != len(y):
        raise
    return 1/n * np.sum((x - y)**2)

def rmsd(x: np.ndarray | List, y: np.ndarray | List) -> float:
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
    n = len(x)
    if n != len(y):
        raise
    return np.sqrt(mse(x, y))

def psnr_db(x: np.ndarray | List, y: np.ndarray | List) -> float:
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
    n = len(x)
    if n != len(y):
        raise
    return 10 * np.log10(1.0 / mse(x, y))

def ber_percent(
        x: np.ndarray | List[List],
        y: np.ndarray | List[List],
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
    n = len(x)
    if n != len(y):
        raise
    return float((x != y).sum() / n) * 100
