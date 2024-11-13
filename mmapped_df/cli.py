# %load_ext autoreload
# %autoreload 2
from pathlib import Path

import click

from mmapped_df.main import _read_schema_tbl, write_schema
from mmapped_df.misc import shell

# path = Path("/home/matteo/clusters.startrek")
# old_name = "ClusterID"
# new_name = "OriginalClusterID"


@click.command(context_settings={"show_default": True})
@click.argument("path", type=Path)
@click.argument("old_name", type=str)
@click.argument("new_name", type=str)
@click.option("--ignore_errors", is_flag=True)
def copy_column(
    path: Path,
    old_name: str,
    new_name: str,
    ignore_errors: bool = False,
) -> None:
    """Copy a column in an existing memmapped data frame.

    Remark: on CoW file systems this will not make any data copies.

    Arguments:\n
        path (pathlib.Path): Path to the memmapped folder.\n
        old_name (str): Name of the column to copy.\n
        naw_name (str): Name for the newly copied data.\n
        ignore_errors (bool): Finish script successfully whatever assumption not met.\n
    """
    schema = _read_schema_tbl(path.expanduser())

    if not old_name in schema:
        if not ignore_errors:
            raise KeyError(
                f"Missig column {old_name} in the table. Present are: {list(schema)}."
            )
        return

    assert (
        not new_name in schema
    ), f"Column `{old_name}` is already present in table saved under `{path}`."

    schema[new_name] = schema[old_name]
    cols = list(schema.columns)
    old_file = path / f"{cols.index(old_name)}.bin"
    new_file = path / f"{cols.index(new_name)}.bin"

    shell(f"cp --reflink=auto {old_file} {new_file}")
    write_schema(schema, path)


# path = Path("/home/matteo/fragments.startrek")
# old_name = "step"
# new_name = "midia_cycle"
# ignore_errors = False
@click.command(context_settings={"show_default": True})
@click.argument("path", type=Path)
@click.argument("old_name", type=str)
@click.argument("new_name", type=str)
@click.option("--ignore_errors", is_flag=True)
def rename_column(
    path: Path,
    old_name: str,
    new_name: str,
    ignore_errors: bool = False,
) -> None:
    """Copy a column in an existing memmapped data frame.

    Remark: on CoW file systems this will not make any data copies.

    Arguments:\n
        path (pathlib.Path): Path to the memmapped folder.\n
        old_name (str): Name of the column to copy.\n
        naw_name (str): Name for the newly copied data.\n
        ignore_errors (bool): Finish script successfully whatever assumption not met.\n
    """
    schema = _read_schema_tbl(path.expanduser())

    if not old_name in schema:
        if not ignore_errors:
            raise KeyError(
                f"Missig column {old_name} in the table. Present are: {list(schema)}."
            )
        return

    assert (
        not new_name in schema
    ), f"Column `{old_name}` is already present in table saved under `{path}`."

    new_schema = schema.rename(columns={old_name: new_name})
    write_schema(new_schema, path)
