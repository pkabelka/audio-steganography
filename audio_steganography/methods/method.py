# -*- coding: utf-8 -*-

# File: method.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the Method enum which maps the individual method classes
to a name in the enum.
"""

from enum import Enum
from .echo_single_kernel import Echo_single_kernel

class Method(Enum):
    """This enum contains implemented steganography methods.
    """
    echo_single_kernel = Echo_single_kernel
