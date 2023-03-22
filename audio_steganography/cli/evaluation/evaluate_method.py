# -*- coding: utf-8 -*-

# File: evaluate_method.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This file contains the function for evaluating a steganograpghy method.
"""

from ...methods import MethodEnum
from ..method_facade import MethodFacade
from ..mode import Mode
from ...exceptions import SecretSizeTooLarge
from .. import prepare_secret_data
from ...decorators import perf
from ...audio_utils import resample, to_dtype, add_normalized_noise
import numpy as np
import pandas as pd
import logging
import json
from typing import List

def half_sampling(input: np.ndarray):
    x = resample(input, 2, 'nearest')
    x = resample(x, 0.5, 'nearest')

    if input.dtype in [
        np.dtype(np.int64),
        np.dtype(np.int32),
        np.dtype(np.int16),
    ]:
        x = x.astype(input.dtype)
    return x

def half_quantization(input: np.ndarray):
    dtype_conv = {
        np.dtype(np.int64): np.int32,
        np.dtype(np.int32): np.int16,
        np.dtype(np.int16): np.int8,
        np.dtype(np.float64): np.float32,
        np.dtype(np.float32): np.float16,
    }

    x = to_dtype(input, dtype_conv.get(input.dtype, np.float64))
    return to_dtype(x, input.dtype)

def evaluate_method(
        method: MethodEnum,
        source_data: np.ndarray,
        columns: List[str],
        extended=False,
    ) -> pd.DataFrame:
    """Evaluates the quality of the given method by encoding data of varying
    length. The stego signal is then filtered, resampled, requantized and
    converted to MP3 and back. The modified signals are used to calculate
    bit error rate to see how robust is the method.

    Parameters
    ----------
    method : MethodEnum
        Chosen steganography method from MethodEnum.
    source_data : NDArray
        Source signal array.
    columns : MethodEnum
        Columns of the statistical DataFrame defined in __main__.
    extended : bool
        Enables extended testing. This includes basinhopping and bruteforce in
        echo methods.

    Returns
    -------
    out : method_base.EncodeDecodeReturn
        Tuple containing NumPy array of samples with secret data encoded
        using direct sequance spread spectrum method and additional output
        needed for decoding.
    """

    facade = MethodFacade(
        method,
        Mode.encode,
        source_data,
    )

    options = {
        MethodEnum.lsb: [
            {'depth': i, 'only_needed': needed}
            for i in [1, 2, 4, 8]
            for needed in [True, False]
        ],
        **dict.fromkeys(
            [
                MethodEnum.echo_single,
                MethodEnum.echo_bipolar,
                MethodEnum.echo_bf,
                MethodEnum.echo_bipolar_bf,
            ], [
            {
                'd0': d0,
                'd1': d0 + 50,
                'alpha': alpha,
                'decay_rate': 0.85,
                'delay_search': delay_search,
            }
                for d0 in [50, 100, 150, 200]
                for alpha in [0.5, 0.25, 0.05] + ([0.1] if extended else [])
                for delay_search in [''] +
                    (['basinhopping', 'bruteforce'] if extended else [])
            ]
        ),
        MethodEnum.phase: [{}],
        MethodEnum.dsss: [
            {'alpha': alpha}
            for alpha in [0.05, 0.005, 0.0025]
        ],
        MethodEnum.silence_interval: [
            {'min_silence_len': l}
            for l in [400, 600] + ([800] if extended else [])
        ],
        MethodEnum.dsss_dft:[
            {'alpha': alpha}
            for alpha in [0.05, 0.005, 0.0025]
        ],
        MethodEnum.tone_insertion: [
            {'f0': f0, 'f1': f1}
            for f0, f1 in list(zip([433, 10685, 18757], [511, 13277, 21703])) +
                ([(5215, 13629), (6331, 15755)] if extended else [])
        ],
    }

    # DataFrame for stats of all runs of the method
    all_stats_df = pd.DataFrame(columns=columns)

    # encode messages of various lengths
    for secret_data in [
        'Bike',
        'Hyperventilation',
        'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer',
    ] + (['Boundary', 'In rhoncus, ligula id dictum sit'] if extended else []):

        logging.info(secret_data)
        facade.data_to_encode = prepare_secret_data(secret_data, None)

        # use various method options for encoding
        for opt in options.get(method, []):
            logging.info(f'method name: {method.name}')
            logging.info(f'parameters: {opt}')
            try:
                # encode
                (stego, additional_output), time_to_encode = perf(facade.encode)(**opt)
            except SecretSizeTooLarge:
                stats_df = pd.DataFrame(
                    [[
                        '',
                        '',
                        '',
                        method.name,
                        json.dumps(opt),
                        len(secret_data) * 8,
                        '',
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.Inf,
                        np.Inf,
                    ]], columns=columns)
                all_stats_df = pd.concat([all_stats_df, stats_df], ignore_index=True)
                continue

            logging.info(f'encoding took: {time_to_encode}')

            # modify the stego signal to test robustness
            modifications = {
                '': lambda x: x,
                'half sampling': half_sampling,
                'half quantization': half_quantization,
                'noise: SNR 20 dB': lambda x: to_dtype(add_normalized_noise(x, 20), x.dtype),
                'noise: SNR 10 dB': lambda x: to_dtype(add_normalized_noise(x, 10), x.dtype),
            }
            modifications.update(
                {
                    'noise: SNR 15 dB': lambda x: to_dtype(add_normalized_noise(x, 15), x.dtype),
                } if extended else {})

            for modification_name, modification_func in modifications.items():
                logging.info(f'modification name: {modification_name}')
                # modify stego signal
                stego = modification_func(stego)

                # decode
                (decoded_secret, _), time_to_decode = (
                    perf(method.value(stego).decode)(**additional_output)
                )
                logging.info(f'decoding took: {time_to_decode}')

                # calculate statistical functions
                stats, time_to_get_stats = perf(facade.get_stats)(stego, decoded_secret)
                logging.info(f'get_stats took: {time_to_get_stats}')

                # append stats to the DataFrame
                stats_df = pd.DataFrame(
                    [[
                        '',
                        '',
                        '',
                        method.name,
                        json.dumps(opt),
                        len(secret_data) * 8,
                        modification_name,
                        stats['ber_percent_secret_encoded'],
                        stats['snr_db'],
                        stats['psnr_db'],
                        stats['mse'],
                        stats['rmsd'],
                        time_to_encode,
                        time_to_decode,
                    ]], columns=columns)
                all_stats_df = pd.concat([all_stats_df, stats_df], ignore_index=True)

    return all_stats_df
