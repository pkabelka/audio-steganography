#!/bin/sh
exec python3 "$(dirname "$(realpath "$0")")/audio_steganography/__main__.py" "$@"
