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
