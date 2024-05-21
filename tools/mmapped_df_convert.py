#!/usr/bin/env python3
import argparse
from pathlib import Path



parser = argparse.ArgumentParser(
    description="Convert pandas dataset to mmapped_df format",
)
parser.add_argument(
    "-i",
    "--input",
    required=True,
    type=Path,
    help="Dataset to convert",
)

parser.add_argument(
    "-o",
    "--output",
    required=True,
    type=Path,
    help="Output path",
)

args = parser.parse_args()

from pandas_ops.io import read_df
from mmapped_df import DatasetWriter
T = read_df(args.input)
with DatasetWriter(args.output) as DW:
    DW.append_df(T)