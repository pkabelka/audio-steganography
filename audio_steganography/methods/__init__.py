# -*- coding: utf-8 -*-

# File: __init__.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains the MethodEnum class which maps the individual method
classes to a name in the enum.
"""

from enum import Enum as _Enum
from .echo_single import EchoSingle
from .echo_bipolar import EchoBipolar
from .echo_bf import EchoBF
from .echo_bipolar_bf import EchoBipolarBF
from .lsb import LSB
from .phase_coding import PhaseCoding
from .dsss import DSSS
from .silence_interval import SilenceInterval
from .dsss_dft import DSSSDFT
from .tone_insertion import ToneInsertion

class MethodEnum(_Enum):
    """This enum contains implemented steganography methods.
    """
    echo_single = EchoSingle
    echo_bipolar = EchoBipolar
    echo_bf = EchoBF
    echo_bipolar_bf = EchoBipolarBF
    lsb = LSB
    phase = PhaseCoding
    dsss = DSSS
    silence_interval = SilenceInterval
    dsss_dft = DSSSDFT
    tone_insertion = ToneInsertion
