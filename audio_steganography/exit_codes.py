# -*- coding: utf-8 -*-

from enum import Enum

class ExitCode(Enum):
    Ok = 0
    InvalidMethod = 1
    InvalidMode = 2
    OutputFileExists = 3
    FileNotFound = 4
    WavReadError = 5
