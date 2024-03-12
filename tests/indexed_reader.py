from mmapped_df import IndexedReader
from tqdm import tqdm
from pprint import pprint
import pandas as pd

IR = IndexedReader("index_test.raw", "group_column")
print(pd.DataFrame(IR.dataset).head(50))
print(IR[3])
L = list(tqdm(IR))
# pprint(L)
#for x in IR:
#    print(x)
