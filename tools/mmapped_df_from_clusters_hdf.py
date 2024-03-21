#!/usr/bin/env python3
import argparse
import sqlite3
import sys
from pathlib import Path

import tqdm
from dia_common import midia_schemes

import h5py
import numba
import numpy as np
import pandas as pd
from mmapped_df import DatasetWriter, open_dataset_dct


class args:
    hdf = Path(
        "P/clusters/tims/G8EAIGRwXkwjnl_fIX5jY39ZEoth6jvbm6xpPnZDKpvB9nfEKOIIKfe7fVB8H239XCUKNaYKNim6yp85nozd6Mxq3Hp0vL3o5RjbK3jG1aL5P8mXdA1TrSHC0ACsK8GKIXkaQ13QM643ygZpAEWCem06oxQ=/precursor_clusters.hdf"
    )
    analysis_tdf_path = Path(
        "P/dataset/CwiAW3NwZWN0cmEvRzgwMjcuZF0D/raw.d/analysis.tdf"
    )
    output = Path("test/bigdump.startrek")
    progressbar_message = "There is no future for you!!!"


parser = argparse.ArgumentParser(
    description="Translate HDF with clusters from tims-tof (or anything else using that format) to mmapped_df `.startrek` format."
)
parser.add_argument(
    "hdf",
    help="Input HDF5 containing clusters from tims clustering tool in HDF format.",
    type=Path,
)
parser.add_argument(
    "analysis_tdf_path",
    help="Path to 'folder.d/analysis.tdf', directly the file.",
    type=Path,
)
parser.add_argument(
    "output",
    help="Path to the final output in the .startrek format.",
    type=Path,
)
parser.add_argument(
    "--progressbar_message",
    help="Show progressbar message.",
    default="",
)
args = parser.parse_args()


@numba.njit
def apply_step_translation(
    bruker_steps,
    midia_steps,
    bruker_steps_to_midia_steps,
) -> None:
    assert len(midia_steps) == len(bruker_steps)
    for i, bruker_step in enumerate(bruker_steps):
        midia_steps[i] = bruker_steps_to_midia_steps[bruker_step]


@numba.njit
def overwrite(xx, yy) -> None:
    assert len(xx) == len(yy)
    for i in range(len(xx)):
        yy[i] = xx[i]


if __name__ == "__main__":
    with h5py.File(args.hdf, mode="r") as hdf:
        clusters = hdf["raw/data"]

        if "step" in clusters:
            assert args.analysis_tdf_path.exists()

            with sqlite3.connect(args.analysis_tdf_path) as conn:
                scheme = midia_schemes.MIDIA_scheme(conn)
            translation = scheme.scheme.query("not is_ms1")
            bruker_step_to_midia_step = dict(
                zip(translation.bruker_step, translation.midia_step)
            )
            bruker_steps = translation.bruker_step.to_numpy()
            midia_steps = translation.midia_step.to_numpy()

            assert bruker_steps.min() >= 0
            bruker_steps_to_midia_steps = np.full(bruker_steps.max() + 1, 1000)
            bruker_steps_to_midia_steps[bruker_steps] = midia_steps

        renaming = lambda x: "midia_step" if x == "step" else x
        dataframe_scheme = pd.DataFrame(
            {renaming(c): pd.Series(dtype=clusters[c].dtype) for c in clusters}
        )
        size = len(clusters["ClusterID"])

        DatasetWriter.preallocate_dataset(
            args.output,
            dataframe_scheme,
            nrows=size,
        )
        datasets = open_dataset_dct(args.output, read_write=True)

        columns = list(clusters)
        if args.progressbar_message:
            print(" ".join(sys.argv))
            columns = tqdm.tqdm(columns, desc=args.progressbar_message)

        for c in columns:
            if c == "step":
                apply_step_translation(
                    clusters[c][()],  # RAM copy
                    datasets[renaming(c)],
                    bruker_steps_to_midia_steps,
                )
            else:
                overwrite(clusters[c][:], datasets[renaming(c)])
