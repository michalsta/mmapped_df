import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

import mmapped_df
from mmapped_df.cli import copy_column, rename_column


@pytest.fixture
def awkward(tmp_path):
    """A dataset whose path would break (or exploit) a shell command line."""
    path = tmp_path / "my data; $(touch PWNED)"
    with mmapped_df.DatasetWriter(path) as w:
        w.append_df(
            pd.DataFrame({"a": np.arange(3, dtype=np.uint32), "b": np.arange(3.0)})
        )
    return path


def run(command, *args):
    return CliRunner().invoke(command, [str(a) for a in args])


def test_copy_column(awkward, tmp_path):
    result = run(copy_column, awkward, "a", "a_copy")
    assert result.exit_code == 0, result.output
    df = mmapped_df.open_dataset(awkward)
    assert df.columns.tolist() == ["a", "b", "a_copy"]
    assert df["a_copy"].dtype == np.uint32
    assert df["a_copy"].tolist() == [0, 1, 2]
    assert not list(tmp_path.rglob("PWNED"))  # nothing went through a shell


def test_rename_column(awkward):
    result = run(rename_column, awkward, "b", "bee")
    assert result.exit_code == 0, result.output
    assert (awkward / "schema.txt").read_text().splitlines() == ["uint32 a", "float64 bee"]


@pytest.mark.parametrize("command", [copy_column, rename_column])
@pytest.mark.parametrize(
    "old, new, message",
    [("nope", "x", "Missing column `nope`"), ("a", "b", "`b` is already present")],
    ids=["missing old", "existing new"],
)
def test_errors(awkward, command, old, new, message):
    schema_before = (awkward / "schema.txt").read_text()
    result = run(command, awkward, old, new)
    assert result.exit_code == 2
    assert message in result.output
    assert (awkward / "schema.txt").read_text() == schema_before

    result = run(command, awkward, old, new, "--ignore_errors")
    assert result.exit_code == 0
    assert (awkward / "schema.txt").read_text() == schema_before


def test_tilde_is_expanded(awkward, monkeypatch):
    monkeypatch.setenv("HOME", str(awkward.parent))
    result = run(copy_column, f"~/{awkward.name}", "a", "a_copy")
    assert result.exit_code == 0, result.output
    assert "a_copy" in mmapped_df.open_dataset(awkward)
