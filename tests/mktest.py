import pandas as pd
import numpy as np
from mmapped_df import DatasetWriter, open_dataset
import os
from tqdm import tqdm


def gr():
    return np.random.randint(low=0, high=10000, size=100000, dtype=np.uint64)


df = pd.DataFrame({"aaa": gr(), "bb ba": gr()})
"""
df.to_pickle('test_data/scheme.pickle')

with open('test_data/1.dat', 'wb') as f:
    f.write(gr().tobytes())
"""
with DatasetWriter("test_data2") as DW:
    for _ in tqdm(range(30)):
        DW.append_df(df)
    df = open_dataset("test_data2")
    ncol = df["aaa"] + df["bb ba"]
    DW.append_column("sum", ncol)

# print(df.dtypes)
