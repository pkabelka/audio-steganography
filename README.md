# Audio Steganography

This repository contains a Python library for coding and decoding arbitrary
data into cover WAV audio files using the implemented steganography methods.

Czech translation of this README can be found in [README-CZ.md](README-CZ.md).

## Prerequisites

- Python 3.8 or *later*
- NumPy
- SciPy
- pandas (for evaluation)
- matplotlib (for evaluation)

These libraries can be installed with:

```
python -m pip install -r requirements.txt
```

## Program usage

The embedded CLI program can be used like this:

```
python -m audio_steganography
```

or with:

```
./audio-steganography.sh
```

The program expects a steganography method as first argument followed by
`encode` or `decode` depending on the wanted operation.

### Encoding

When encoding, you must use `-s` to specify the cover audio file and either
`-f` for encoding a file or `-t` for encoding text from the argument. Some
methods may ask you to enter a value for their required parameters. Also, most
methods have optional parameters which can influence the encoding results.
These parameters can be printed out with the `-h` switch after the method name
argument.

### Decoding

When decoding, you must use `-s` to specify the stego audio file. Some methods
also require you to use the parameters that were used when encoding.

## Library usage

Import the library with:

```
import audio_steganography
```

and use the methods from [methods](audio_steganography/methods) directory.

If you want add a method, the [method_base
module](audio_steganography/methods/method_base.py) contains the abstract base
class `MethodBase` which all methods must inherit. Then the new method class
must then be added into the [MethoEnum
class](audio_steganography/methods/__init__.py) and all of the method's
parameters must be added to the `options` dictionary in [`main`
function](audio_steganography/cli/__init__.py).

## LICENSE

This project is under the Apache-2.0 license. See [LICENSE](LICENSE) and
[NOTICE](NOTICE) for more information.
