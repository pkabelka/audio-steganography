#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File: process.py
# Author: Petr Kabelka <xkabel09 at stud.fit.vutbr.cz>

"""This file contains the main() function for the program to process audio
steganography methods evaluation results.
"""

from ..cli_utils import get_attr
import argparse
import uuid
from typing import Tuple, Any, List
from pathlib import Path
import pandas as pd

def parse_args() -> Tuple[Any, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d',
        '--datasets', metavar='PATH',
        action='store',
        help='path to a directory containing dataset evaluation results; '+
            'expected structure is: <dataset root>/<datasets>/<categories>/<files>; '+
            'directories starting with "." will be ignored',
        required=True)

    parser.add_argument(
        '-o',
        '--output',
        metavar='OUTPUT_DIR',
        action='store',
        help='path to an output directory; EXISTING FILES WILL BE OVERWRITTEN',
        default=None)

    # Parse all args
    args = parser.parse_args()
    return args, parser

def set_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df['dataset'] = df['dataset'].astype('category')
    df['category'] = df['category'].astype('category')
    df['file'] = df['file'].astype('category')
    df['method'] = df['method'].astype('category')
    df['params'] = df['params'].astype('category')
    df['secret_bits'] = df['secret_bits'].astype('category')
    df['modification'] = df['modification'].astype('category')
    return df

def process_data(df: pd.DataFrame) -> List[pd.DataFrame]:
    dfs = []
    return dfs

def main():
    """The main function of the evaluation data processing program.
    """
    args, _ = parse_args()
    output_dir = Path(args.output)

    file_dfs = []

    dataset_root = Path(args.datasets)
    datasets = [
        x for x in dataset_root.iterdir()
        if x.is_dir() and not x.name.startswith('.')
    ]
    for dataset in datasets:
        categories = [
            x for x in dataset.iterdir()
            if x.is_dir() and not x.name.startswith('.')
        ]
        for category in categories:
            files = [
                x for x in category.iterdir() if
                x.is_file() and not x.name.startswith('.')
                            and x.suffix.lower() == '.csv'
            ]
            for file in files:
                file_dfs.append(
                    pd.read_csv(
                        file,
                        sep=';',
                    )
                )

    df_all = pd.concat(file_dfs, ignore_index=True)
    df_all = set_dtypes(df_all)

    dataframes = process_data(df_all)

    # write all DataFrames to CSVs
    for df in dataframes:
        df_name = get_attr(df, 'name')
        df_name = df_name if df_name is not None else str(uuid.uuid4())
        df.to_csv(
            output_dir / f'{df_name}.csv',
            sep=';',
            index=False,
        )

if __name__ == '__main__':
    main()
