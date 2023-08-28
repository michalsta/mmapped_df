from mmapped_df import IndexedReader
from tqdm import tqdm
from pprint import pprint
IR = IndexedReader('aaa.raw', 'idx')
print(IR[99999])
L=list(tqdm(IR))
#pprint(L)
for x in IR: print(x)