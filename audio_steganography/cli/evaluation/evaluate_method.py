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
from ...audio_utils import resample
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
            for i in [1, 2, 4, 8] for needed in [True, False]
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
                'decay_rate': decay_rate,
                'delay_search': delay_search,
            }
                for d0 in [50, 100, 150, 200]
                for alpha in [0.5, 0.25, 0.1, 0.05]
                for decay_rate in [0.85, 0.5]
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
            for f0, f1 in zip([3685, 5215, 13277, 18757], [4629, 6331, 15755, 21703])
        ],
    }

    # DataFrame for stats of all runs of the method
    all_stats_df = pd.DataFrame(columns=columns)

    for secret_data in [
        'Bike',
        'Hyperventilation',
        'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer',
    ] + (['Boundary', 'In rhoncus, ligula id dictum sit'] if extended else []):

        logging.info(secret_data)
        facade.data_to_encode = prepare_secret_data(secret_data, None)

        for opt in options.get(method, []):
            logging.info(method.name)
            logging.info(opt)
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

            # modifications
            for modification_name, modification_func in {
                '': lambda x: x,
                'half sampling': half_sampling,
            }.items():
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
