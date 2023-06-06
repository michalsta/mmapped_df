from pathlib import Path

import numpy as np
import pandas as pd
import mmap
import os

class DatasetWriter:
    def __init__(self, path : Path | str):
        self.files = None
        self.colnames = None
        self.dtypes = None
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)

    def _set_schema(self, like : pd.DataFrame):
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
        pd.DataFrame(types).to_pickle(self.path / 'scheme.pickle')

    def __del__(self):
        if self.files is not None:
            for file in self.files:
                file.close()

    def append_df(self, df):
        if self.files is None:
            self._set_schema(like=df)
        for idx, colname in enumerate(df):
            column = df[colname]
            assert colname == self.colnames[idx]
            assert column.values.dtype == self.dtypes[idx]
            self.files[idx].write(column.values.tobytes())


def open_dataset(path : Path | str):
    path = Path(path)
    df = pd.read_pickle(path / 'scheme.pickle')

    new_data = {}
    for idx, column_name in enumerate(df):
        col_dtype = df[column_name].values.dtype
        fd = os.open(path / f"{idx}.bin", os.O_RDONLY)
        mmap_obj = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        new_data[column_name] = np.frombuffer(mmap_obj, dtype=col_dtype)

    return pd.DataFrame(new_data)