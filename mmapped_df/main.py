import mmap
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import polars as pl


class DatasetWriter:
    def __init__(self, path: Path | str, append_ok: bool = False):
        self.files = None
        self.colnames = None
        self.dtypes = None
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=append_ok)
        if append_ok and (self.path / "scheme.pickle").exists:
            df = pd.read_pickle(path / "scheme.pickle")
            self.files = []
            self.colnames = []
            self.dtypes = []
            for idx, colname in enumerate(df):
                self.files.append(open(self.path / f"{idx}.bin", "ab"))
                self.colnames.append(self.colname)
                self.dtypes.append(df[colname].values.dtype)

    def _set_schema(self, like: pd.DataFrame):
        assert self.files is None
        self.files = []
        self.colnames = []
        self.dtypes = []
        types = {}
        for idx, colname in enumerate(like):
            file = open(self.path / f"{idx}.bin", "wb")
            self.files.append(file)
            self.colnames.append(colname)
            dtype = like[colname].values.dtype
            types[colname] = np.empty(dtype=dtype, shape=0)
            self.dtypes.append(dtype)
        pd.DataFrame(types).to_pickle(self.path / "scheme.pickle")

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

    def append_df(self, df):
        if self.files is None:
            self._set_schema(like=df)
        for idx, colname in enumerate(df):
            column = df[colname]
            assert colname == self.colnames[idx]
            assert column.values.dtype == self.dtypes[idx]
            self.files[idx].write(column.values.tobytes())

    def flush(self):
        for file in self.files:
            file.flush()

    def append(self, **kwargs):
        for file, dtype, colname in zip(self.files, self.dtypes, self.colnames):
            dat = dtype.type(kwargs[colname])
            file.write(dat.tobytes())

    def append_dct(self, D):
        return self.append(**D)

    def append_list(self, L):
        assert len(L) == len(self.files)
        for data, file, dtype in zip(L, self.files, self.dtypes):
            dat = dtype.type(data)
            file.write(dat.tobytes())


def open_dataset_dct(path: Path | str, **kwargs):
    path = Path(path)
    df = pd.read_pickle(path / "scheme.pickle")

    new_data = {}
    for idx, column_name in enumerate(df):
        col_dtype = df[column_name].values.dtype
        fd = os.open(path / f"{idx}.bin", os.O_RDONLY)
        mmap_obj = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        new_data[column_name] = np.frombuffer(mmap_obj, dtype=col_dtype)

    return new_data

def open_dataset(path: Path | str, **kwargs):
    return pd.DataFrame(open_dataset_dct(path, **kwargs), copy=False)

def np_to_pa(np_arr):
    '''Convert Numpy array to Pyarrow one, sharing the same backing buffer'''
    pyarrow_buf = pa.py_buffer(np_arr)
    dtype = pa.from_numpy_dtype(np_arr.dtype)
    return pa.Array.from_buffers(type=dtype, length=len(np_arr), buffers=[None, pyarrow_buf], null_count=0)

def open_dataset_pa(path: Path | str, **kwargs):
    '''Return dataset as dict of colname -> mmapped pyarrow array'''
    return {key: np_to_pa(val) for key, val in open_dataset_dct(path, **kwargs).items()}

def open_dataset_pl(path: Path | str, **kwargs):
    '''Return dataset as mmapped Polars dataframe'''
    return pl.from_arrow(pa.table(open_dataset_pa(path, **kwargs)))
