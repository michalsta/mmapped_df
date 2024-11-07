import pathlib

import click

import h5py


@click.command(context_settings={"show_default": True})
@click.option(
    "--in_hdf",
    help="Path to the hdf file.",
    required=True,
    type=pathlib.Path,
)
@click.option(
    "--out_startrek",
    help="Path where to save the .startrek folder.",
    required=True,
    type=pathlib.Path,
)
@click.option(
    "--root",
    help="Path inside of the hdf that includes the columns to be extracted. This group contains all the datasets that share the shape.",
    default="/",
)
def hdf_to_startrek(
    in_hdf: pathlib.Path, out_startrek: pathlib.Path, root: str = "/"
) -> None:
    """
    Translate an HDF-stored table into .startrek format.
    """
    if False:
        in_hdf = pathlib.Path("tmp/clusters/tims/1fd37e91592/precursor/25/clusters.hdf")
        out_startrek = pathlib.Path("/tmp/clusters.startrek")
        root = "raw/data"
        hdf = h5py.File(in_hdf, mode="r")

    with h5py.File(in_hdf, mode="r") as hdf:
        rootgroup = hdf[root]
