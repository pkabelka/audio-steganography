#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File: evaluation.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This file contains the main() function for the audio steganography method
evaluation program.
"""

from ..methods import MethodEnum
from .method_facade import MethodFacade
from .mode import Mode
from ..exceptions import SecretSizeTooLarge
from .cli_utils import error_exit
from .exit_codes import ExitCode
from . import prepare_secret_data
import argparse
from typing import Tuple, List, Any
from pathlib import Path
import scipy.io.wavfile
import numpy as np
import pandas as pd
import time
import logging

def parse_args() -> Tuple[Any, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-w',
        '--workdir', metavar='PATH',
        action='store',
        help='path to a directory containing datasets',
        required=True)

    parser.add_argument(
        '-o',
        '--output',
        metavar='OUTPUT_FILE',
        action='store',
        help='path to an output file; - outputs to STDOUT',
        default=None)

    parser.add_argument(
        '-y',
        '--overwrite',
        action='store_true',
        help='overwrite existing OUTPUT_FILE',
        default=False)

    parser.add_argument(
        '-l',
        '--log',
        action='store_true',
        help='enables information logging',
        default=False)

    parser.add_argument(
        'methods',
        action='store',
        choices=['ALL'] + [method.name for method in MethodEnum],
        nargs='*',
        help='method(s) to evaluate; default: ALL',
        default='ALL')

    # Parse all args
    args = parser.parse_args()
    return args, parser

def evaluate_method(
        method: MethodEnum, 
        source_data: np.ndarray, 
        columns,
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
                for delay_search in ['', 'basinhopping', 'bruteforce']
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

    all_stats_df = pd.DataFrame(columns=columns)

    facade.data_to_encode = prepare_secret_data('Lorem ipsum dolor sit amet consectetur', None)

    for opt in options.get(method, []):
        logging.info(method.name)
        logging.info(opt)
        try:
            start_time = time.perf_counter()
            output, additional_output = facade.encode(**opt)
            end_time = time.perf_counter()
            time_to_encode = end_time - start_time
        except SecretSizeTooLarge:
            continue

        stats = facade.get_stats(output, additional_output)
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
            ]], columns=columns)
        all_stats_df = pd.concat([all_stats_df, stats_df], ignore_index=True)
        # additional_output = {**additional_output, **stats}

    return all_stats_df

def main():
    """The main function of the evaluation program.

    The function first parses user arguments and checks if they are valid.
    Output is written to a file or printed to STDOUT.
    """
    args, _ = parse_args()
    logging.basicConfig(level=logging.INFO)
    if not args.log:
        logging.disable(logging.INFO) 

    # check the output file
    output_file = Path(args.output)
    if args.output != '-':
        if output_file.exists() and not args.overwrite:
            error_exit('output file already exists', ExitCode.OutputFileExists)

    # get the requested methods from args
    methods: List[MethodEnum] = [method for method in MethodEnum]
    if isinstance(args.methods, list) and 'ALL' not in args.methods:
        methods = []
        for method in args.methods:
            # Check if the method is valid
            try:
                methods.append(MethodEnum[method])
            except KeyError:
                error_exit('invalid method specified', ExitCode.InvalidMethod)

    dataset_root = Path(args.workdir)
    datasets = [x for x in dataset_root.iterdir() if x.is_dir()]

    columns = [
        'dataset', 
        'category', 
        'file', 
        'method', 
        'params', 
        'ber_percent',
        'snr_db',
        'psnr_db',
        'mse',
        'rmsd',
        'time_to_encode',
    ]
    stats = pd.DataFrame( columns=columns)

    # evaluate chosen methods on dataset files
    for dataset in datasets:
        logging.debug(dataset.name)
        categories = [x for x in dataset.iterdir() if x.is_dir()]
        logging.debug(categories)
        for category in categories:
            files = [x for x in category.iterdir() if
                x.is_file() and x.suffix.lower() == '.wav']
            logging.debug(files)
            for file in files:
                for method in methods:
                    # Read source WAV data
                    try:
                        source_sr, source_data = scipy.io.wavfile.read(file)
                    except FileNotFoundError:
                        error_exit('source file not found', ExitCode.FileNotFound)
                    except ValueError as e:
                        error_exit(str(e), ExitCode.WavReadError)

                    logging.info(file)
                    method_res = evaluate_method(method, source_data, columns)
                    method_res['dataset'] = dataset.name
                    method_res['category'] = category.name
                    method_res['file'] = file.name
                    stats = pd.concat([stats, method_res], ignore_index=True)

    # output stats to STDOUT or CSV file
    if args.output == '-':
        print(stats)
    else:
        stats.to_csv(output_file, sep=';')

if __name__ == '__main__':
    main()
