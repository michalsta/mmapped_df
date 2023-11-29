import mmap
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import pyarrow as pa

from .numba_helper import mkindex


def schema_to_str(schema: pd.DataFrame):
    ret = []
    for colname in schema:
        ret.append(f"{schema[colname].values.dtype} {colname}")
    return "\n".join(ret)


def str_to_schema(s: str):
    ret = {}
    for line in s.splitlines():
        dtype_str, colname = line.split(maxsplit=1)
        ret[colname] = np.empty(dtype=np.dtype(dtype_str), shape=0)
    return pd.DataFrame(ret)


def _read_schema_tbl(path: Path):
    try:
        with open(path / "schema.txt", "rt") as f:
            return str_to_schema(f.read())
    except FileNotFoundError:
        pickle_path = path / "scheme.pickle"
        scheme = pd.read_pickle(pickle_path)
        return scheme


class DatasetWriter:
    def __init__(self, path: Path | str, append_ok: bool = False, overwrite_dir=False):
        self.files = None
        self.colnames = None
        self.dtypes = None
        self.schema = None
        self.path = Path(path)
        if self.path.exists() and append_ok:
            tbl = _read_schema_tbl(self.path)
            self._reset_schema(tbl)
        else:
            self.path.mkdir(parents=True, exist_ok=overwrite_dir)

    def _reset_schema(self, like: pd.DataFrame):
        self.close()
        self.files = []
        self.colnames = []
        self.dtypes = []
        schema_str = schema_to_str(like)
        self.schema = str_to_schema(schema_str)
        lengths = []
        for idx, colname in enumerate(self.schema):
            file_path = self.path / f"{idx}.bin"
            self.files.append(open(file_path, "ab", buffering=10240))
            self.colnames.append(colname)
            self.dtypes.append(self.schema[colname].values.dtype)
            lengths.append(file_path.stat().st_size / self.dtypes[-1].itemsize)

        if not (lengths == [] or all(l == lengths[0] for l in lengths)):
            raise RuntimeError(
                f"Corrupted dataset: columns of unequal lengths: {self.path}"
            )

        self.length = 0 if lengths == [] else int(lengths[0])

        with open(self.path / "schema.txt", "wt") as f:
            f.write(schema_str)

    def close(self):
        if self.files is not None:
            for file in self.files:
                file.close()
        self.files = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __len__(self):
        return self.length

    def append_df(self, df):
        if len(df) == 0:
            return
        if self.files is None:
            self._reset_schema(like=df)
        self.length += len(df)
        for idx, colname in enumerate(df):
            column = df[colname]
            assert colname == self.colnames[idx]
            assert (
                column.values.dtype == self.dtypes[idx] or len(df) == 0
            ), f"Types don't match: {column.values.dtype} vs {self.dtypes[idx]}"
            self.files[idx].write(column.values.tobytes())

    def append_column(self, colname: str, column: npt.NDArray | pd.Series):
        if isinstance(column, pd.Series):
            column = column.values
        assert self.length == len(column)
        self.schema[colname] = np.empty_like(column)
        with open(self.path / f"{len(self.files)}.bin", "xb") as f:
            f.write(column.tobytes())
        self._reset_schema(like=self.schema)

    def flush(self):
        for file in self.files:
            file.flush()

    def append(self, **kwargs):
        lengths = []
        for file, dtype, colname in zip(self.files, self.dtypes, self.colnames):
            dat = dtype.type(kwargs[colname])
            file.write(dat.tobytes())
            lengths.append(len(dat))
        self.length += lengths[0]

    def append_row(self, **kwargs):
        for file, dtype, colname in zip(self.files, self.dtypes, self.colnames):
            dat = dtype.type(kwargs[colname])
            file.write(dat.tobytes())
        self.length += 1

    def append_dct(self, D):
        return self.append(**D)

    def append_list(self, L):
        assert len(L) == len(self.files)
        self.length += len(L[0])
        for data, file, dtype in zip(L, self.files, self.dtypes):
            dat = dtype.type(data)
            file.write(dat.tobytes())


def open_dataset_dct(path: Path | str, **kwargs):
    path = Path(path)
    df = _read_schema_tbl(path)

    new_data = {}
    for idx, column_name in enumerate(df):
        col_dtype = df[column_name].values.dtype
        fd = os.open(path / f"{idx}.bin", os.O_RDONLY)
        mmap_obj = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        new_data[column_name] = np.frombuffer(mmap_obj, dtype=col_dtype)

    return new_data


def open_dataset_simple_namespace(path: Path | str, **kwargs) -> SimpleNamespace:
    return SimpleNamespace(**open_dataset_dct(path, **kwargs))


def open_dataset(path: Path | str, **kwargs):
    return pd.DataFrame(open_dataset_dct(path, **kwargs), copy=False)


def np_to_pa(np_arr):
    """Convert Numpy array to Pyarrow one, sharing the same backing buffer"""
    pyarrow_buf = pa.py_buffer(np_arr)
    dtype = pa.from_numpy_dtype(np_arr.dtype)
    return pa.Array.from_buffers(
        type=dtype, length=len(np_arr), buffers=[None, pyarrow_buf], null_count=0
    )


def open_dataset_pa(path: Path | str, **kwargs):
    """Return dataset as dict of colname -> mmapped pyarrow array"""
    return {key: np_to_pa(val) for key, val in open_dataset_dct(path, **kwargs).items()}


def open_dataset_pl(path: Path | str, **kwargs):
    """Return dataset as mmapped Polars dataframe"""
    return pl.from_arrow(pa.table(open_dataset_pa(path, **kwargs)))


class IndexedReader:
    def __init__(self, path: Path | str, index_col: str, **kwargs):
        self.dataset = open_dataset_dct(path, **kwargs)
        self.indexed = self.dataset[index_col]
        self.index = mkindex(self.indexed)

    def __getitem__(self, idx):
        return {
            k: v[self.index[idx] : self.index[idx + 1]] for k, v in self.dataset.items()
        }

    def __len__(self):
        return len(self.index) - 1
