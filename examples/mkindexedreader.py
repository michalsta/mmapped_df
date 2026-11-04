import pandas as pd
import numpy as np
from mmapped_df import DatasetWriter, open_dataset
import os
from tqdm import tqdm


example_size = 1000000


def gr():
    return np.random.randint(low=0, high=10000, size=example_size, dtype=np.uint64)


def group_col():
    X = np.random.choice(
        [0, 1, 2, 10], size=example_size, replace=True, p=[0.9, 0.05, 0.04, 0.01]
    )
    return np.cumsum(X)


df = pd.DataFrame({"group_column": group_col(), "aaa": gr(), "bb ba": gr()})
"""
df.to_pickle('test_data/scheme.pickle')

with open('test_data/1.dat', 'wb') as f:
    f.write(gr().tobytes())
"""
with DatasetWriter("index_test.raw") as DW:
    DW.append_df(df)

# print(df.dtypes)
