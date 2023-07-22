import pandas as pd
import numpy as np
from mmapped_df import DatasetWriter
import os
from tqdm import tqdm

def gr():
 return np.random.randint(low=0, high=9_00_000_000, size=100000, dtype=np.uint64)


#with DatasetWriter("join_data_1") as DW:
#    for _ in tqdm(range(30000)):
#        df = pd.DataFrame({'col1' : gr(),
#                           'col2' : gr()})
#        DW.append_df(df)

with DatasetWriter("join_data_2") as DW:
    DW._set_schema(like=pd.DataFrame({'idx':gr(), 'modulo':gr()}))
    start = 0
    for idx in tqdm(range(2_000)):
        DW.append(idx=idx, modulo=idx%100)
        df = pd.DataFrame({'idx':np.arange(start, start+1_000_000, dtype=np.uint64)})
        df['modulo'] = df.idx % 100
        DW.append_df(df)
        start += 1_000_000

#print(df.dtypes)

