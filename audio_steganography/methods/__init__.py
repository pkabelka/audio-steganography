# -*- coding: utf-8 -*-

# File: __init__.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the MethodEnum class which maps the individual method
classes to a name in the enum.
"""

from enum import Enum as _Enum
from .echo_single_kernel import Echo_single_kernel
from .lsb import LSB
from .phase_coding import PhaseCoding
from .dsss import DSSS

class MethodEnum(_Enum):
    """This enum contains implemented steganography methods.
    """
    echo_single_kernel = Echo_single_kernel
    lsb = LSB
    phase = PhaseCoding
    dsss = DSSS
