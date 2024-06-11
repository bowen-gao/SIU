import os

import dotenv as de
import numpy as np
import pandas as pd

from atom3d.datasets import LMDBDataset
from atom3d.util.voxelize import dotdict, get_center, gen_rot_matrix, get_grid
from torch.utils.data import DataLoader
import pickle

de.load_dotenv(de.find_dotenv(usecwd=True))

import lmdb

# class LMDBDataset:
#     def __init__(self, db_path, transform=None):
#         self.db_path = db_path
#         assert os.path.isfile(self.db_path), "{} not found".format(
#             self.db_path
#         )
#         self._transform = transform
#         env = self.connect_db(self.db_path)
#         with env.begin() as txn:
#             self._keys = list(txn.cursor().iternext(values=False))

#     def connect_db(self, lmdb_path, save_to_self=False):
#         env = lmdb.open(
#             lmdb_path,
#             subdir=False,
#             readonly=True,
#             lock=False,
#             readahead=False,
#             meminit=False,
#             max_readers=256,
#         )
#         if not save_to_self:
#             return env
#         else:
#             self.env = env

#     def __len__(self):
#         return len(self._keys)

#     @lru_cache(maxsize=16)
#     def __getitem__(self, idx):
#         if not hasattr(self, 'env'):
#             self.connect_db(self.db_path, save_to_self=True)
#         datapoint_pickled = self.env.begin().get(self._keys[idx])
#         data = pickle.loads(datapoint_pickled)
#         if self._transform:
#             data = self._transform(data)
#         return data


class CNN3D_TransformLBA(object):
    def __init__(self, random_seed=None, **kwargs):
        self.random_seed = random_seed
        self.grid_config =  dotdict({
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
        # Update grid configs as necessary
        self.grid_config.update(kwargs)

    def _voxelize(self, atoms_pocket, atoms_ligand):
        # Use center of ligand as subgrid center
        ligand_pos = atoms_ligand[['x', 'y', 'z']].astype(np.float32)
        #print(ligand_pos.shape)
        #print(atoms_ligand)
        ligand_center = get_center(ligand_pos)
        # Generate random rotation matrix
        rot_mat = gen_rot_matrix(self.grid_config, random_seed=self.random_seed)
        # Transform protein/ligand into voxel grids and rotate
        grid = get_grid(pd.concat([atoms_pocket, atoms_ligand]),
                        ligand_center, config=self.grid_config, rot_mat=rot_mat)
        
        # Last dimension is atom channel, so we need to move it to the front
        # per pytroch style
        grid = np.moveaxis(grid, -1, 0)
        return grid

    def __call__(self, item):
        # Transform protein/ligand into voxel grids.
        # Apply random rotation matrix.
        #print(item["atoms_pocket"], item['atoms_ligand'])
        #print(item["scores"])
        scores = []
        types = ["ic50", "ec50", "kd", "ki", "potency"]
        #print(item["scores"])
        for t in types:
            if t in item["scores"]:
                scores.append(item["scores"][t])

            else:
                scores.append(-1.0)
        #print(scores)
        transformed = {
            'feature': self._voxelize(item['atoms_pocket'], item['atoms_ligand']),
            'label': scores,
            'id': item['id'],
            'source': item['source']
        }
        return transformed


if __name__=="__main__":
    dataset_path = os.path.join(os.environ['LBA_DATA'], 'val', "data.mdb")
    dataset = LMDBDataset(dataset_path, transform=CNN3D_TransformLBA(radius=10.0))
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    for item in dataloader:
        print('feature shape:', item['feature'].shape)
        print('label:', item['label'])
        print()
        break
