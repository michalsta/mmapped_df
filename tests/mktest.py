import pandas as pd
import numpy as np
from mmapped_df import DatasetWriter
import os

def gr():
 return np.random.randint(low=0, high=10000, size=100000, dtype=np.uint64)


df = pd.DataFrame({'aaa' : gr(),
                   'bbba' : gr()})
'''
df.to_pickle('test_data/scheme.pickle')

with open('test_data/1.dat', 'wb') as f:
    f.write(gr().tobytes())
'''
with DatasetWriter("test_data2") as DW:
    for _ in range(30000):
        DW.append_df(df)

#print(df.dtypes)

