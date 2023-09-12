#!/usr/bin/env python3
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Unpack a Bruker TDF dataset into mmapped_df-compatible format.",
)


parser.add_argument(
    "-i",
    "--input",
    help="Input TDF dataset (usually a .d directory)",
    required=True,
    type=Path,
)

parser.add_argument("-o", "--output", help="Output path", required=True, type=Path)

parser.add_argument(
    "-f",
    "--force",
    action="store_true",
    help="Delete the target directory if it exists",
)

parser.add_argument(
    "--columns",
    help="Comma-separated list of columns to extract and store. Default: frame,scan,tof,intensity,mz,inv_ion_mobility,retention_time (all of them)",
    default="frame,scan,tof,intensity,mz,inv_ion_mobility,retention_time",
)

parser.add_argument(
    "-s", "--silent", help="Do not show progress bar", action="store_true"
)

args = parser.parse_args()

import shutil
from opentimspy import OpenTIMS
import pandas as pd
from mmapped_df import DatasetWriter
import sys

if not args.silent:
    from tqdm import tqdm
else:

    def tqdm(x, *args, **kwargs):
        return x


with OpenTIMS(args.input) as OT:
    if args.force:
        shutil.rmtree(args.output, ignore_errors=True)

    with DatasetWriter(args.output) as DW:
        for frame in tqdm(
            OT, total=len(OT.frames["Id"]), desc=sys.argv[0] + " (processing frames)"
        ):
            df = pd.DataFrame(frame, copy=False)
            DW.append_df(df)
