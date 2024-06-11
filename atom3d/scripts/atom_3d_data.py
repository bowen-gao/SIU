

import os

import lmdb

import os
import pickle
import lmdb

from tqdm import tqdm, trange


import os
import pandas as pd
from atom3d.util.voxelize import dotdict, get_center, gen_rot_matrix, get_grid

import re


def remove_subscripts(formula):
    # Use a regular expression to remove digits
    cleaned_formula = re.sub(r'\d+', '', formula)
    return cleaned_formula

def voxelize(atoms_pocket, atoms_ligand):

    grid_config = dotdict({
            # Mapping from elements to position in channel dimension.
            'element_mapping': {
                'H': 0,
                'C': 1,
                'O': 2,
                'N': 3,
                'F': 4,
            },
            # Radius of the grids to generate, in angstroms.
            'radius': 20.0,
            # Resolution of each voxel, in angstroms.
            'resolution': 1.0,
            # Number of directions to apply for data augmentation.
            'num_directions': 20,
            # Number of rolls to apply for data augmentation.
            'num_rolls': 20,
        })

    # Use center of ligand as subgrid center
    ligand_pos = atoms_ligand[['x', 'y', 'z']].astype(np.float32)
    #print(ligand_pos.shape)
    #print(atoms_ligand)
    ligand_center = get_center(ligand_pos)
    # Generate random rotation matrix
    rot_mat = gen_rot_matrix(grid_config, random_seed=1)
    # Transform protein/ligand into voxel grids and rotate
    grid = get_grid(pd.concat([atoms_pocket, atoms_ligand]),
                    ligand_center, config=grid_config, rot_mat=rot_mat)
    
    # Last dimension is atom channel, so we need to move it to the front
    # per pytroch style
    grid = np.moveaxis(grid, -1, 0)
    return grid

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
    pocket_list = []
    count=0
    for idx in tqdm(keys):
        datapoint_pickled = txn.get(idx)
        
        data = pickle.loads(datapoint_pickled)
        #print(data["coordinates"].shape)
        out_list.append(data)
        # #print(len(data))
        #print(data.keys())
        # print(data["subset"])
        # print(data["IDs"])
        


    env.close()
    return out_list 



def write_lmdb(out_list, save_path):
    
    env = lmdb.open(
        save_path,
        subdir=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=64,
        map_size=1099511627776
    )

    with env.begin(write=True) as lmdb_txn:
        for i in tqdm(range(len(out_list))):
            #print('{:0>10d}'.format(i), pickle.dumps(out_list[i]))
            lmdb_txn.put(str(i).encode('ascii'), pickle.dumps(out_list[i]))





import atom3d.datasets as da

from atom3d.datasets import make_lmdb_dataset














    






root = "/data/protein/ssmopi_train_data/split_90"

train_path = os.path.join(root, 'test.lmdb')

train_data = read_lmdb(train_path)


new_data = []

for d in tqdm(train_data):
    #rint(d.keys())
    # convert atoms and coordinates into a dataframe with key elemnt for atoms and x,y,z for coordinates

    atoms = d["atoms"]
    # convert element name to element type (without number in the end)

    atoms = [remove_subscripts(a) for a in atoms]
    #print(atoms)
    #print(atoms)
    coordinates = d["coordinates"][0]

    df_mol = pd.DataFrame(atoms, columns=["element"])

    df_mol["x"] = coordinates[:,0]
    df_mol["y"] = coordinates[:,1]
    df_mol["z"] = coordinates[:,2]

    import numpy as np

    pocket_atoms = d["pocket_atoms"]
    pocket_coordinates = np.array(d["pocket_coordinates"])
    #print(pocket_coordinates.shape)
    df_pocket = pd.DataFrame(pocket_atoms, columns=["element"])

    df_pocket["x"] = pocket_coordinates[:,0]
    df_pocket["y"] = pocket_coordinates[:,1]
    df_pocket["z"] = pocket_coordinates[:,2]

    #print(df_mol)
    #print(df_pocket)

    scores = d["label"]
    data = {
        "atoms_pocket": df_pocket,
        "atoms_ligand": df_mol,
        "scores": scores,
        "source": d["source_data"],
        "id": d["source_data"]
    }

    #print(data["source"])

    new_data.append(data)



make_lmdb_dataset(new_data, "/data/protein/ssmopi_train_data/atom3d/split_90/test")

    







