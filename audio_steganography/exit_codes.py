# -*- coding: utf-8 -*-

# File: exit_codes.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains all program's exit codes.
"""

from enum import Enum

class ExitCode(Enum):
    Ok = 0
    InvalidMethod = 1
    InvalidMode = 2
    OutputFileExists = 3
    FileNotFound = 4
    WavReadError = 5
    SecretSizeTooLarge = 6
