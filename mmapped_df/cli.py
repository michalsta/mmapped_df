import subprocess
from pathlib import Path

import click

from mmapped_df.main import _read_schema_tbl, write_schema


def _load_schema(path: Path, old_name: str, new_name: str):
    """Read the schema and check the column names.

    Returns None if `old_name` is missing and `ignore_errors` is set, so the
    caller can exit successfully without doing anything.
    """
    schema = _read_schema_tbl(path)
    if old_name not in schema:
        raise click.UsageError(
            f"Missing column `{old_name}` in the table saved under `{path}`. Present are: {list(schema)}."
        )
    if new_name in schema:
        raise click.UsageError(
            f"Column `{new_name}` is already present in the table saved under `{path}`."
        )
    return schema


def _run(path: Path, old_name: str, new_name: str, ignore_errors: bool, action):
    path = path.expanduser()
    try:
        schema = _load_schema(path, old_name, new_name)
    except click.UsageError:
        if ignore_errors:
            return
        raise
    action(path, schema)


@click.command(context_settings={"show_default": True})
@click.argument("path", type=Path)
@click.argument("old_name", type=str)
@click.argument("new_name", type=str)
@click.option(
    "--ignore_errors",
    is_flag=True,
    help="Exit successfully, doing nothing, if OLD_NAME is missing or NEW_NAME already exists.",
)
def copy_column(
    path: Path,
    old_name: str,
    new_name: str,
    ignore_errors: bool = False,
) -> None:
    """Copy column OLD_NAME to a new column NEW_NAME in the dataset at PATH.

    On copy-on-write file systems (btrfs, XFS, ...) the data is reflinked, not copied.
    """

    def action(path: Path, schema):
        schema[new_name] = schema[old_name]
        cols = list(schema.columns)
        old_file = path / f"{cols.index(old_name)}.bin"
        new_file = path / f"{cols.index(new_name)}.bin"
        # Argument list, no shell: paths may contain spaces or metacharacters.
        subprocess.run(
            ["cp", "--reflink=auto", str(old_file), str(new_file)], check=True
        )
        write_schema(schema, path)

    _run(path, old_name, new_name, ignore_errors, action)


@click.command(context_settings={"show_default": True})
@click.argument("path", type=Path)
@click.argument("old_name", type=str)
@click.argument("new_name", type=str)
@click.option(
    "--ignore_errors",
    is_flag=True,
    help="Exit successfully, doing nothing, if OLD_NAME is missing or NEW_NAME already exists.",
)
def rename_column(
    path: Path,
    old_name: str,
    new_name: str,
    ignore_errors: bool = False,
) -> None:
    """Rename column OLD_NAME to NEW_NAME in the dataset at PATH.

    Only schema.txt is rewritten; no data is touched.
    """

    def action(path: Path, schema):
        write_schema(schema.rename(columns={old_name: new_name}), path)

    _run(path, old_name, new_name, ignore_errors, action)
