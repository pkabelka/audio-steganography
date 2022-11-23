# -*- coding: utf-8 -*-

# File: utils.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This module contains various utility functions.
"""

from .exit_codes import ExitCode
import sys
from typing import Any, Optional

def error_exit(message: str, exit_code: ExitCode):
    """Exits the program with the supplied error message and exit code.

    Parameters
    ----------
    message : str
        Message to print.
    exit_code : ExitCode
        Exit code to use.
    """
    print(f'{sys.argv[0]}: error: {message}', file=sys.stderr)
    sys.exit(exit_code.value)

def get_attr(namespace: object, attribute: str) -> Optional[Any]:
    """Checks if the namespace object contains the attribute and returns its
    value or `None`.

    Parameters
    ----------
    namespace : object
        Any namespace object.
    attribute : str
        Attribute name.

    Returns
    -------
    out : Unknown | None
        The value of the attribute or `None` if the namespace does not contain
        the attribute.
    """
    if hasattr(namespace, attribute):
        return getattr(namespace, attribute)
    else:
        return None
