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
from matplotlib import pyplot as plt

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

    df_useful_cols = df.query('time_to_encode != inf | time_to_decode != inf')
    df_useful_cols = df_useful_cols.drop(['mse', 'rmsd', 'time_to_encode', 'time_to_decode'], axis=1)

    # No method modifications, all params, min, max and mean values
    df_no_mod_all_param = df_useful_cols[df_useful_cols['modification'].isna()]
    df_no_mod_all_param_group = df_no_mod_all_param.groupby('method')
    df_no_mod_all_param_group_min = df_no_mod_all_param_group.min(numeric_only=True).reset_index()
    df_no_mod_all_param_group_max = df_no_mod_all_param_group.max(numeric_only=True).reset_index()
    df_no_mod_all_param_group_mean = df_no_mod_all_param_group.mean(numeric_only=True).reset_index()
    df_no_mod_all_param_group_min.name = 'no_modifications_all_params_min'
    df_no_mod_all_param_group_max.name = 'no_modifications_all_params_max'
    df_no_mod_all_param_group_mean.name = 'no_modifications_all_params_mean'

    # df_no_mod_all_param_group_min.plot.bar(x='method', rot=45)
    # plt.show()
    # df_no_mod_all_param_group_max.plot.bar(x='method', rot=45)
    # plt.show()
    # df_no_mod_all_param_group_mean.plot.bar(x='method', rot=45)
    # plt.show()

    dfs.append(df_no_mod_all_param_group_min)
    dfs.append(df_no_mod_all_param_group_max)
    dfs.append(df_no_mod_all_param_group_mean)

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
