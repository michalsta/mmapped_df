import pandas as pd
import numpy as np
from mmapped_df import DatasetWriter
import os

def gr():
 return np.random.randint(low=0, high=100, size=100, dtype=np.uint64)


df = pd.DataFrame({'aaa' : gr(),
                   'bbba' : gr()})
'''
df.to_pickle('test_data/scheme.pickle')

with open('test_data/1.dat', 'wb') as f:
    f.write(gr().tobytes())
'''
DatasetWriter("test_data2").append_df(df)

#print(df.dtypes)

