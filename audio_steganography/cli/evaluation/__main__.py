#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File: __main__.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This file contains the main() function for the audio steganography method
evaluation program.
"""

from ..cli_utils import error_exit
from ..exit_codes import ExitCode
from .argument_parsing import parse_args
from ...methods import MethodEnum
from .evaluate_method import evaluate_method
from typing import List
from pathlib import Path
import scipy.io.wavfile
import logging
import time
import multiprocessing as mp

def evaluation_process(
        dataset,
        category,
        file,
        methods,
        columns,
        extended,
        output_dir,
    ):
    logging.info(f'file: {file}')
    for method in methods:
        # Read source WAV data
        try:
            source_sr, source_data = scipy.io.wavfile.read(file)
        except FileNotFoundError:
            error_exit('source file not found', ExitCode.FileNotFound)
        except ValueError as e:
            error_exit(str(e), ExitCode.WavReadError)

        method_res = evaluate_method(
            method,
            source_data,
            source_sr,
            columns,
            extended,
        )
        method_res['dataset'] = dataset.name
        method_res['category'] = category.name
        method_res['file'] = file.name

        output_path = output_dir / f'{dataset.name}' / f'{category.name}'
        output_path.mkdir(parents=True, exist_ok=True)

        method_res.to_csv(
            output_path / f'{file.name}.csv',
            sep=';',
        )

def main():
    """The main function of the evaluation program.

    The function first parses user arguments and checks if they are valid.
    Output is written to a file or printed to STDOUT.
    """
    args, _ = parse_args()
    logging.basicConfig(level=logging.INFO)
    if not args.log:
        logging.disable(logging.INFO)

    output_dir = Path(args.output)

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

    columns = [
        'dataset',
        'category',
        'file',
        'method',
        'params',
        'secret_bits',
        'modification',
        'ber_percent',
        'snr_db',
        'psnr_db',
        'mse',
        'rmsd',
        'time_to_encode',
        'time_to_decode',
    ]

    # evaluate chosen methods on dataset files
    dataset_root = Path(args.workdir)
    # ignore datasets starting with "."
    datasets = [
        x for x in dataset_root.iterdir()
        if x.is_dir() and not x.name.startswith('.')
    ]

    # multiprocess pool
    pool = mp.Pool(mp.cpu_count())

    start_time = time.perf_counter()
    for dataset in datasets:
        logging.debug(dataset.name)
        categories = [
            x for x in dataset.iterdir()
            if x.is_dir() and not x.name.startswith('.')
        ]
        logging.debug(categories)
        for category in categories:
            files = [x for x in category.iterdir() if
                x.is_file() and x.suffix.lower() == '.wav']
            logging.debug(files)
            for file in files:
                # add files to pool
                pool.apply_async(
                    evaluation_process,
                    args=(
                        dataset,
                        category,
                        file,
                        methods,
                        columns,
                        args.extended,
                        output_dir,
                    ),
                )

    # run processes
    pool.close()
    pool.join()

    logging.info(f'evaluation took: {time.perf_counter() - start_time}')

    # stats = stats.sort_values(
    #     [
    #         'method',
    #         'dataset',
    #         'category',
    #         'file',
    #         'ber_percent',
    #         'snr_db',
    #     ],
    # )

if __name__ == '__main__':
    main()
