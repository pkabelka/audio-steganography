#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File: __main__.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This file loads and executes the main function which is inside __init__.py

This file is a modification of __main__.py in the yt-dlp project which is in
public domain.

URL: https://github.com/yt-dlp/yt-dlp
"""

import sys

if __package__ is None and not hasattr(sys, 'frozen'):
    import os.path
    path = os.path.realpath(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(path)))

import audio_steganography

if __name__ == '__main__':
    audio_steganography.main()
