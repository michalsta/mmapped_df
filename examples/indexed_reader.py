from pprint import pprint

from tqdm import tqdm

import pandas as pd
from mmapped_df import IndexedReader

IR = IndexedReader("index_test.raw", "group_column")
print(pd.DataFrame(IR.dataset, copy=False).head(50))
print(IR[3])
L = list(tqdm(IR))
# pprint(L)
# for x in IR:
#    print(x)
