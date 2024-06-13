import os
import pickle
import lmdb
from tqdm import tqdm

import os
import pandas as pd

def read_lmdb(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )  
    txn = env.begin()
    keys = list(txn.cursor().iternext(values=False))
    out_list = []

    for idx in tqdm(keys):
        datapoint_pickled = txn.get(idx)
        data = pickle.loads(datapoint_pickled)
        out_list.append(data)
    env.close()
    return out_list 



def read_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data



if __name__ == '__main__':
    # Path to the lmdb file
    lmdb_path = '/path/to/lmdb'
    data = read_lmdb(lmdb_path)
    print(data[0])

    # Path to the pickle file
    pickle_path = '/path/to/pkl/file'
    data = read_pickle(pickle_path)
    print(data[0])