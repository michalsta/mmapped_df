import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import mmapped_df
from mmapped_df import (
    open_dataset,
    open_dataset_dct,
    open_dataset_pa,
    open_dataset_simple_namespace,
)


def test_open_dataset(written, small_df):
    pd.testing.assert_frame_equal(open_dataset(written), small_df)


def test_open_dataset_dct_is_read_only_by_default(written):
    cols = open_dataset_dct(written)
    assert list(cols) == ["id", "value", "with space"]
    assert not any(c.flags.writeable for c in cols.values())
    with pytest.raises(ValueError, match="read-only"):
        cols["id"][0] = 1


def test_read_write_persists(written):
    cols = open_dataset_dct(written, read_write=True)
    cols["id"][0] = 42
    cols["value"][-1] = -1.5
    del cols
    back = open_dataset(written)
    assert back["id"][0] == 42
    assert back["value"].iloc[-1] == -1.5


def test_simple_namespace(written, small_df):
    ns = open_dataset_simple_namespace(written)
    np.testing.assert_array_equal(ns.value, small_df["value"].values)


@pytest.mark.parametrize(
    "opener", [open_dataset, open_dataset_dct, open_dataset_pa, open_dataset_simple_namespace]
)
def test_unknown_keyword_rejected(written, opener):
    # Regression: a typo used to be swallowed by **kwargs, silently opening read-only.
    with pytest.raises(TypeError, match="readwrite"):
        opener(written, readwrite=True)


def test_open_empty_dataset(tmp_path):
    # Regression: mmap refuses empty files.
    path = tmp_path / "ds"
    mmapped_df.DatasetWriter.preallocate_dataset(
        path, mmapped_df.get_scheme(a=np.uint32, b=np.float64), nrows=0
    )
    df = open_dataset(path)
    assert df.shape == (0, 2)
    assert df.dtypes.tolist() == [np.uint32, np.float64]
    assert not open_dataset_dct(path)["a"].flags.writeable
    assert open_dataset_dct(path, read_write=True)["a"].flags.writeable
    assert len(open_dataset_pa(path)["b"]) == 0


def test_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_dataset(tmp_path / "nope")


# --- Arrow --------------------------------------------------------------------

ZERO_COPY = {
    "int8": np.arange(5, dtype=np.int8),
    "uint64": np.arange(5, dtype=np.uint64),
    "float16": np.arange(5, dtype=np.float16),
    "float64": np.arange(5, dtype=np.float64),
    "datetime64[ns]": np.arange(5).astype("datetime64[ns]"),
    "datetime64[s]": np.arange(5).astype("datetime64[s]"),
    "timedelta64[us]": np.arange(5).astype("timedelta64[us]"),
}


@pytest.mark.parametrize("name", ZERO_COPY)
def test_np_to_pa_zero_copy(name):
    arr = ZERO_COPY[name]
    out = mmapped_df.np_to_pa(arr)
    assert out.buffers()[1].address == arr.ctypes.data  # shared, not copied
    np.testing.assert_array_equal(out.to_numpy(zero_copy_only=False), arr)


INCOMPATIBLE = {
    # Regression: bool used to be misread (Arrow bit-packs it).
    "bool": np.array([True, False, True, True, False]),
    # Regression: Arrow reads datetime64[D] as 32-bit date32.
    "datetime64[D]": np.arange(5).astype("datetime64[D]"),
    "strided": np.arange(10)[::2],
}


@pytest.mark.parametrize("name", INCOMPATIBLE)
def test_np_to_pa_raises_instead_of_copying(name):
    with pytest.raises(TypeError, match="zero-copy"):
        mmapped_df.np_to_pa(INCOMPATIBLE[name])


def test_open_dataset_pa(written, small_df):
    arrays = open_dataset_pa(written)
    assert arrays["id"].type == pa.uint32()
    for col in small_df:
        assert arrays[col].to_pylist() == small_df[col].tolist()


def test_open_dataset_pa_bool_column_raises(tmp_path):
    with mmapped_df.DatasetWriter(tmp_path / "ds") as w:
        w.append_df(pd.DataFrame({"b": np.array([True, False])}))
    with pytest.raises(TypeError, match="zero-copy"):
        open_dataset_pa(tmp_path / "ds")


def _polars_usable() -> bool:
    # Importing polars on a CPU it was not built for can kill the process
    # (SIGILL), so probe it in a subprocess.
    probe = "import polars; polars.DataFrame({'a': [1]}).sum()"
    return subprocess.run([sys.executable, "-c", probe], capture_output=True).returncode == 0


@pytest.mark.skipif(not _polars_usable(), reason="polars not installed or not usable on this CPU")
def test_open_dataset_pl(written, small_df):
    df = mmapped_df.open_dataset_pl(written)
    assert df.columns == list(small_df.columns)
    assert df["value"].to_list() == small_df["value"].tolist()
