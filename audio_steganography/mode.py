# -*- coding: utf-8 -*-

# File: mode.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains Mode enum.
"""

from enum import Enum

class Mode(Enum):
    encode = 'encode'
    decode = 'decode'

