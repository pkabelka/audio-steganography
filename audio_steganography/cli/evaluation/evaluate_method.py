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
import numpy as np
import pandas as pd
import logging
from typing import List

def evaluate_method(
        method: MethodEnum, 
        source_data: np.ndarray, 
        columns: List[str],
        extended=False,
    ) -> pd.DataFrame:

    facade = MethodFacade(
        method,
        Mode.encode,
        source_data,
    )

    options = {
        MethodEnum.lsb: [
            {
                'depth': i,
                'only_needed': needed,
            } for i in range(1, 9) for needed in [True, False]
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
                for delay_search in [''] + (['basinhopping', 'bruteforce'] if extended else [])
            ]
        ),
        MethodEnum.phase: [{}],
        MethodEnum.dsss: [{'alpha': alpha} for alpha in [0.05, 0.005, 0.001]],
        MethodEnum.silence_interval: [
            {'min_silence_len': l} for l in [400, 600, 800]
        ],
        MethodEnum.dsss_dft:[{'alpha': alpha} for alpha in [0.05, 0.005, 0.001]], 
        MethodEnum.tone_insertion: [
            {
                'f0': f0,
                'f1': f1,
            } for f0, f1 in zip([3685, 5215, 13277, 18757], [4629, 6331, 15755, 21703]) 
        ],
    }

    # DataFrame for stats of all runs of the method
    all_stats_df = pd.DataFrame(columns=columns)

    facade.data_to_encode = prepare_secret_data('Lorem ipsum', None)

    for opt in options.get(method, []):
        logging.info(method.name)
        logging.info(opt)
        try:
            # encode
            (output, additional_output), time_to_encode = perf(facade.encode)(**opt)
        except SecretSizeTooLarge:
            stats_df = pd.DataFrame(
                [[
                    '', 
                    '', 
                    '', 
                    method.name, 
                    opt,
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

        # decode
        (decoded_secret, _), time_to_decode = perf(method.value(output).decode)(**additional_output)
        logging.info(f'decoding took: {time_to_decode}')

        # calculate statistical functions
        stats, time_to_get_stats = perf(facade.get_stats)(output, decoded_secret)
        logging.info(f'get_stats took: {time_to_get_stats}')

        # append stats to the DataFrame
        stats_df = pd.DataFrame(
            [[
                '', 
                '', 
                '', 
                method.name, 
                opt, 
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
