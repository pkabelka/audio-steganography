# -*- coding: utf-8 -*-

# File: utils.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains various utility functions.
"""

from .exit_codes import ExitCode
import sys

def error_exit(message: str, exit_code: ExitCode):
    print(f'{sys.argv[0]}: error: {message}', file=sys.stderr)
    sys.exit(exit_code.value)
