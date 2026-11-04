import os

import numpy as np
import pandas as pd
import pytest

import mmapped_df
from mmapped_df import DatasetWriter, open_dataset, open_dataset_dct


def assert_frames_equal(path, expected: pd.DataFrame):
    pd.testing.assert_frame_equal(open_dataset(path), expected.reset_index(drop=True))


# --- on-disk format -----------------------------------------------------------


def test_on_disk_format(written, small_df):
    assert sorted(os.listdir(written)) == ["0.bin", "1.bin", "2.bin", "schema.txt"]
    assert (written / "schema.txt").read_text().splitlines() == [
        "uint32 id",
        "float64 value",
        "int16 with space",
    ]
    for idx, col in enumerate(small_df):
        raw = np.fromfile(written / f"{idx}.bin", dtype=small_df[col].dtype)
        np.testing.assert_array_equal(raw, small_df[col].values)


# --- append_df ----------------------------------------------------------------


def test_append_df_chunks(tmp_path, small_df):
    path = tmp_path / "ds"
    with DatasetWriter(path) as w:
        for _ in range(3):
            w.append_df(small_df)
        assert len(w) == 15
        assert type(len(w)) is int
    assert_frames_equal(path, pd.concat([small_df] * 3))


def test_append_empty_df_is_noop(tmp_path, small_df):
    path = tmp_path / "ds"
    with DatasetWriter(path) as w:
        w.append_df(small_df)
        w.append_df(small_df.iloc[:0])
        assert len(w) == 5
    assert_frames_equal(path, small_df)


@pytest.mark.parametrize(
    "bad",
    [
        lambda df: df[["id", "value"]],  # missing column
        lambda df: df[["value", "id", "with space"]],  # reordered
        lambda df: df.assign(extra=1),  # extra column
    ],
    ids=["missing", "reordered", "extra"],
)
def test_append_df_rejects_column_mismatch(tmp_path, small_df, bad):
    path = tmp_path / "ds"
    with DatasetWriter(path) as w:
        w.append_df(small_df)
        with pytest.raises(ValueError, match="do not match the dataset schema"):
            w.append_df(bad(small_df))
        assert len(w) == 5
    assert_frames_equal(path, small_df)


def test_append_df_rejects_dtype_mismatch_before_writing(tmp_path, small_df):
    path = tmp_path / "ds"
    with DatasetWriter(path) as w:
        w.append_df(small_df)
        # Only the last column is wrong: the first two must not be written.
        with pytest.raises(TypeError, match="with space"):
            w.append_df(small_df.astype({"with space": np.int64}))
        assert len(w) == 5
    assert_frames_equal(path, small_df)


# --- opening modes ------------------------------------------------------------


def test_existing_dir_without_flags_raises(written):
    with pytest.raises(FileExistsError):
        DatasetWriter(written)


def test_append_ok_appends(written, small_df):
    with DatasetWriter(written, append_ok=True) as w:
        assert len(w) == 5
        w.append_df(small_df)
        assert len(w) == 10
    assert_frames_equal(written, pd.concat([small_df, small_df]))


def test_overwrite_dir_starts_fresh(written):
    other = pd.DataFrame({"x": np.arange(3, dtype=np.int8)})
    with DatasetWriter(written, overwrite_dir=True) as w:
        w.append_df(other)
    assert sorted(os.listdir(written)) == ["0.bin", "schema.txt"]
    assert_frames_equal(written, other)


# --- other ways of writing ----------------------------------------------------


def test_append_row_and_new(tmp_path):
    path = tmp_path / "ds"
    w = DatasetWriter.new(path, a=np.uint16, b=np.float32)
    w.append_row(a=1, b=2.5)
    w.append_row(b=4, a=3)  # keyword order does not matter
    assert len(w) == 2
    w.close()
    assert_frames_equal(
        path,
        pd.DataFrame(
            {"a": np.array([1, 3], np.uint16), "b": np.array([2.5, 4], np.float32)}
        ),
    )


def test_append_row_infers_schema(tmp_path):
    path = tmp_path / "ds"
    with DatasetWriter(path) as w:
        w.append_row(a=np.int32(1), b=np.float64(0.5))
        w.append_row(a=2, b=1.5)
    assert open_dataset(path).dtypes.tolist() == [np.int32, np.float64]
    assert open_dataset(path).values.tolist() == [[1, 0.5], [2, 1.5]]


def test_new_requires_columns(tmp_path):
    with pytest.raises(AssertionError):
        DatasetWriter.new(tmp_path / "ds")


def test_get_scheme():
    scheme = mmapped_df.get_scheme(a=np.uint8, b=np.float64)
    assert len(scheme) == 0
    assert scheme.dtypes.tolist() == [np.uint8, np.float64]


# --- append_column(s) ---------------------------------------------------------


def test_append_column_keeps_existing_dtypes(written, small_df):
    # Regression: adding a column used to turn existing columns into float64
    # in schema.txt.
    with DatasetWriter(written, append_ok=True) as w:
        w.append_column("new", np.arange(5, dtype=np.int8))
        w.append_columns(
            flag=pd.Series([True, False, True, False, True]),
            big=np.arange(5, dtype=np.uint64),
        )
    expected = small_df.assign(
        new=np.arange(5, dtype=np.int8),
        flag=[True, False, True, False, True],
        big=np.arange(5, dtype=np.uint64),
    )
    assert_frames_equal(written, expected)


def test_append_column_same_size_dtype_regression(tmp_path):
    # With 8-byte columns the old float64 corruption was silent.
    path = tmp_path / "ds"
    with DatasetWriter(path) as w:
        w.append_df(pd.DataFrame({"a": np.arange(3, dtype=np.uint64)}))
        w.append_column("b", np.arange(3, dtype=np.uint64) * 10)
    assert (path / "schema.txt").read_text().splitlines() == ["uint64 a", "uint64 b"]


def test_append_column_rejects_existing_name(written, small_df):
    with DatasetWriter(written, append_ok=True) as w:
        with pytest.raises(ValueError, match="already present"):
            w.append_column("id", np.zeros(5, dtype=np.uint32))
    assert_frames_equal(written, small_df)
    assert len(os.listdir(written)) == 4


def test_append_columns_checks_all_names_first(written, small_df):
    with DatasetWriter(written, append_ok=True) as w:
        with pytest.raises(ValueError, match="already present"):
            w.append_columns(fresh=np.zeros(5), id=np.zeros(5, dtype=np.uint32))
    assert_frames_equal(written, small_df)  # `fresh` was not added either


def test_append_column_rejects_wrong_length(written):
    with DatasetWriter(written, append_ok=True) as w:
        with pytest.raises(AssertionError):
            w.append_columns(short=np.zeros(4))


# --- preallocation ------------------------------------------------------------


def test_preallocate_and_fill(tmp_path):
    path = tmp_path / "ds"
    cols = mmapped_df.open_new_dataset_dct(
        path, mmapped_df.get_scheme(x=np.float64, n=np.uint8), nrows=1000
    )
    assert all(len(c) == 1000 and c.flags.writeable for c in cols.values())
    assert not cols["x"].any()  # zero-filled
    cols["x"][:] = np.arange(1000)
    cols["n"][::2] = 7
    del cols
    back = open_dataset_dct(path)
    np.testing.assert_array_equal(back["x"], np.arange(1000))
    assert back["n"].tolist()[:4] == [7, 0, 7, 0]


def test_preallocate_existing_dir_raises(written):
    with pytest.raises(FileExistsError):
        DatasetWriter.preallocate_dataset(
            written, mmapped_df.get_scheme(a=np.int8), nrows=1
        )


# --- dtypes -------------------------------------------------------------------

STORABLE = {
    "u8": np.arange(4, dtype=np.uint8),
    "i64": np.arange(4, dtype=np.int64),
    "f16": np.arange(4, dtype=np.float16),
    "f32": np.arange(4, dtype=np.float32),
    "c128": np.arange(4) + 1j,
    "bool": np.array([True, False, False, True]),
    "big-endian": np.arange(4, dtype=">i4"),
    "ns": np.arange(4).astype("datetime64[ns]"),
    "s": np.arange(4).astype("datetime64[s]"),
    "td": np.arange(4).astype("timedelta64[ms]"),
}


@pytest.mark.parametrize("name", STORABLE)
def test_storable_dtypes_round_trip(tmp_path, name):
    path = tmp_path / "ds"
    df = pd.DataFrame({name: STORABLE[name]})
    with DatasetWriter(path) as w:
        w.append_df(df)
    with DatasetWriter(path, append_ok=True) as w:
        w.append_df(df)
    back = open_dataset(path)
    assert back[name].dtype == df[name].dtype
    np.testing.assert_array_equal(back[name].values, np.concatenate([df[name].values] * 2))


UNSTORABLE = {
    "object": pd.Series([1, "x", None, 2.0], dtype=object),
    "str": pd.Series(["a", "b", "c", "d"]),
    "bytes": pd.Series(np.array([b"a", b"bc", b"", b"d"], dtype="S2")),
    "category": pd.Series(["a", "b", "a", "b"], dtype="category"),
    "nullable Int64": pd.Series([1, None, 3, 4], dtype="Int64"),
    "tz-aware": pd.Series(pd.date_range("2026-01-01", periods=4, tz="UTC")),
}


@pytest.mark.parametrize("name", UNSTORABLE)
def test_unstorable_dtypes_rejected_before_writing(tmp_path, name):
    path = tmp_path / "ds"
    with DatasetWriter(path) as w:
        with pytest.raises(TypeError, match="cannot be stored"):
            w.append_df(pd.DataFrame({"ok": np.arange(4), name: UNSTORABLE[name]}))
    assert os.listdir(path) == []


def test_unstorable_append_row_string(tmp_path):
    with DatasetWriter(tmp_path / "ds") as w:
        with pytest.raises(TypeError, match="cannot be stored"):
            w.append_row(a=1, s="text")


def test_writer_usable_after_rejected_column(written, small_df):
    with DatasetWriter(written, append_ok=True) as w:
        with pytest.raises(TypeError):
            w.append_column("s", np.array([1, "x", None, 2, 3], dtype=object))
        w.append_df(small_df)
    assert_frames_equal(written, pd.concat([small_df, small_df]))


def test_tz_aware_into_naive_column_rejected(tmp_path):
    with DatasetWriter(tmp_path / "ds") as w:
        w.append_df(pd.DataFrame({"t": pd.date_range("2026", periods=2).values}))
        with pytest.raises(TypeError):
            w.append_df(pd.DataFrame({"t": pd.date_range("2026", periods=2, tz="UTC")}))


# --- corruption detection -----------------------------------------------------


def test_reopen_detects_partial_value(written):
    os.truncate(written / "0.bin", 5 * 4 - 1)  # uint32 column
    with pytest.raises(RuntimeError, match="not a multiple"):
        DatasetWriter(written, append_ok=True)


def test_reopen_detects_unequal_lengths(written):
    os.truncate(written / "0.bin", 4 * 4)  # 4 rows of uint32 instead of 5
    with pytest.raises(RuntimeError, match="unequal lengths"):
        DatasetWriter(written, append_ok=True)


# --- open bugs ----------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="bug: DatasetWriter.length is unset before a schema exists")
def test_len_of_fresh_writer(tmp_path):
    assert len(DatasetWriter(tmp_path / "ds")) == 0


@pytest.mark.xfail(strict=True, reason="bug: DatasetWriter.append calls len() on scalars")
def test_append_scalars(tmp_path):
    with DatasetWriter.new(tmp_path / "ds", a=np.uint32) as w:
        w.append(a=3)
        assert len(w) == 1
