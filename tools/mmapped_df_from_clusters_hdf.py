#!/usr/bin/env python3
import argparse
import typing
from pathlib import Path

import h5py
import pandas as pd
from mmapped_df import DatasetWriter
from MSclusterparser.raw_peaks_4DFF_parser import Clusters_4DFF_HDF
from tqdm import tqdm


class args:
    hdfs = [Path("partial/G8027/fd2b37bd4e1/default/ms2_clusters.hdf")]
    output_files = [Path("/tmp/test_ms2.startrek")]
    output_folder = None
    progressbar = True
    chunk_size = 100_000_000
    progressbar_message = "There is no future for you!!!"


parser = argparse.ArgumentParser(
    description="Translate HDF with clusters from tims-tof (or anything else using that format) to mmapped_df `.startrek` format."
)
parser.add_argument(
    "hdfs",
    help="Input file(s) containing clusters from tims clustering tool in HDF format.",
    nargs="+",
    type=Path,
)
parser.add_argument(
    "--output_files",
    help="Path(s) to the final output: if provided, then in the same number as input HDF5 files.",
    nargs="+",
    type=Path,
)
parser.add_argument(
    "--output_folder",
    help="Path to the output folder: alternative to --output_files. Will contain all folders named after the input hdfs but with `.startrek` extension.",
    default=None,
    type=Path,
)
parser.add_argument(
    "--chunk_size",
    help="How big should one chunk be. Likely should be automated, but it's difficult to find any info on that.",
    default=100_000_000,
    type=int,
)

parser.add_argument(
    "--progressbar_message",
    help="Show progressbar message.",
    default="",
)

args = parser.parse_args()


def iter_pairs_of_indices(
    start: int, end: int, step: int
) -> typing.Iterator[tuple[int]]:
    i_prev = start
    i = i_prev + step
    while i <= end:
        yield i_prev, i
        i_prev = i
        i += step
    if i_prev < end:
        yield i_prev, end


if __name__ == "__main__":
    assert (args.output_folder is None) ^ (len(args.output_files) == 0)
    if len(args.output_files) > 0:
        assert len(args.output_files) == len(args.hdfs)
        output_paths = args.output_files
    else:
        output_paths = [
            args.output_folder / hdf_path.with_suffix(".startrek").name
            for hdf_path in args.hdfs
        ]

    for i, (hdf_path, output_path) in enumerate(zip(args.hdfs, output_paths)):
        if args.progressbar_message:
            print(
                f"File {i+1}/{len(args.hdfs)}\n:\tdumping `{hdf_path}` to `{output_path}`\n"
            )
        hdf = h5py.File(hdf_path, mode="r", swmr=True)
        rawdata = hdf["/raw/data"]
        dataset_writer = DatasetWriter(output_path, append_ok=True)

        start_end_tuples = list(
            iter_pairs_of_indices(
                start=0, end=len(rawdata["ClusterID"]), step=args.chunk_size
            )
        )
        if args.progressbar_message:
            start_end_tuples = tqdm(start_end_tuples, desc=args.progressbar_message)

        for start, end in start_end_tuples:
            df = pd.DataFrame(
                {column: rawdata[column][start:end] for column in rawdata}, copy=False
            )
            dataset_writer.append_df(df)
