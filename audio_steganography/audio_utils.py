# -*- coding: utf-8 -*-

# File: audio_utils.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains audio utility functions.
"""

import numpy as np
from typing import List

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

