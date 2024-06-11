# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from functools import lru_cache

import numpy as np
from unicore.data import BaseWrapperDataset

from . import data_utils
import re

class AffinityDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms,
        coordinates,
        pocket_atoms,
        pocket_coordinates,
        affinity,
        is_train=False,
        pocket="pocket"
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.affinity = affinity
        self.is_train = is_train
        self.pocket=pocket
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def pocket_atom(self, atom):
        if atom[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return atom[1]
        else:
            return atom[0]
    
    def mol_atom(self, formula):
        # Use a regular expression to remove digits
        cleaned_formula = re.sub(r'\d+', '', formula)
        return cleaned_formula

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array([self.mol_atom(item) for item in self.dataset[index][self.atoms]])
        #print(atoms)
        #atoms = np.array(self.dataset[index][self.atoms])
        #print(atoms)
        #coordinates = self.dataset[index][self.coordinates]
        size = len(self.dataset[index][self.coordinates])
        if self.is_train:
            with data_utils.numpy_seed(self.seed, epoch, index):
                sample_idx = np.random.randint(size)
        else:
            with data_utils.numpy_seed(self.seed, 1, index):
                sample_idx = np.random.randint(size)
        #print(len(self.dataset[index][self.coordinates][sample_idx]))
        coordinates = self.dataset[index][self.coordinates][sample_idx]
        #print(coordinates.shape)
        #print(coordinates.shape)
        pocket_atoms = np.array(
            [self.pocket_atom(item) for item in self.dataset[index][self.pocket_atoms]]
        )
        pocket_coordinates = np.stack(self.dataset[index][self.pocket_coordinates])

        #smi = self.dataset[index]["smi"]
        if self.pocket in self.dataset[index]:
            pocket = self.dataset[index][self.pocket]
        else:
            pocket = "a,b"
        
        # check whether the label is a dict or float 

        if isinstance(self.dataset[index]["label"], dict): 
        
            affinity_list = {'ic50':-1.0, 'ki':-1.0, 'kd':-1.0, 'ec50':-1.0, "potency":-1.0}

            for type in affinity_list:
                if type in self.dataset[index]["label"]:
                    affinity_list[type] = float(self.dataset[index]["label"][type])
            
            affinity_list = [affinity_list["ic50"], affinity_list["ki"], affinity_list["kd"], affinity_list["ec50"], affinity_list["potency"]]

        else:
            affinity_list = float(self.dataset[index]["label"])

        
                
        
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "pocket": pocket,
            "affinity_list": affinity_list,
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class AffinityAugDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms,
        coordinates,
        pocket_atoms,
        pocket_coordinates,
        affinity,
        is_train=False,
        pocket="pocket_id"
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.affinity = affinity
        self.is_train = is_train
        self.pocket=pocket
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def pocket_atom(self, atom):
        if atom[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        #mol_atoms_list = self.dataset[index][self.atoms]
        with data_utils.numpy_seed(self.seed, epoch, index):
            mol_idx = np.random.randint(len(self.dataset[index][self.atoms]))
        atoms = np.array(self.dataset[index][self.atoms][mol_idx])
        ori_mol_length = len(atoms)
        #coordinates = self.dataset[index][self.coordinates]
        size = len(self.dataset[index][self.coordinates][mol_idx])
        if self.is_train:
            with data_utils.numpy_seed(self.seed, epoch, index):
                sample_idx = np.random.randint(size)
        else:
            with data_utils.numpy_seed(self.seed, 1, index):
                sample_idx = np.random.randint(size)
        #print(len(self.dataset[index][self.coordinates][sample_idx]))
        coordinates = self.dataset[index][self.coordinates][mol_idx][sample_idx]


        #pocket_list = self.dataset[index][self.pocket_atoms]
        with data_utils.numpy_seed(self.seed, epoch, index):
            pocket_idx = np.random.randint(len(self.dataset[index][self.pocket_atoms]))
        pocket_atoms = np.array(
            [self.pocket_atom(item) for item in self.dataset[index][self.pocket_atoms][pocket_idx]]
        )

        ori_pocket_length = len(pocket_atoms)
        pocket_coordinates = np.stack(self.dataset[index][self.pocket_coordinates][pocket_idx])

        smi = self.dataset[index]["smiles"][mol_idx]
        pocket = self.dataset[index][self.pocket][0]
        if self.affinity in self.dataset[index]:
            affinity = float(self.dataset[index][self.affinity])
        else:
            affinity = 1
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "holo_coordinates": coordinates.astype(np.float32),#placeholder
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_pocket_coordinates": pocket_coordinates.astype(np.float32),#placeholder
            "smi": smi,
            "pocket": pocket,
            "affinity": affinity,
            "ori_mol_length": ori_mol_length,
            "ori_pocket_length": ori_pocket_length
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class AffinityHNSDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms,
        coordinates,
        atoms_hns,
        coordinates_hns,
        pocket_atoms,
        pocket_coordinates,
        affinity,
        is_train=False,
        pocket="pocket"
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.atoms_hns = atoms_hns
        self.coordinates_hns = coordinates_hns
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.affinity = affinity
        self.is_train = is_train
        self.pocket=pocket
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def pocket_atom(self, atom):
        if atom[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        ori_mol_length = len(atoms)
        #coordinates = self.dataset[index][self.coordinates]
        size = len(self.dataset[index][self.coordinates])
        if self.is_train:
            with data_utils.numpy_seed(self.seed, epoch, index):
                sample_idx = np.random.randint(size)
        else:
            with data_utils.numpy_seed(self.seed, 1, index):
                sample_idx = np.random.randint(size)
        #print(len(self.dataset[index][self.coordinates][sample_idx]))
        coordinates = self.dataset[index][self.coordinates][sample_idx]
        atoms_hns = np.array(self.dataset[index][self.atoms_hns])
        coordinates_hns = self.dataset[index][self.coordinates_hns][0]



        pocket_atoms = np.array(
            [self.pocket_atom(item) for item in self.dataset[index][self.pocket_atoms]]
        )
        ori_pocket_length = len(pocket_atoms)
        pocket_coordinates = np.stack(self.dataset[index][self.pocket_coordinates])

        smi = self.dataset[index]["smi"]
        pocket = self.dataset[index][self.pocket]
        if self.affinity in self.dataset[index]:
            affinity = float(self.dataset[index][self.affinity])
        else:
            affinity = 1
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "atoms_hns": atoms_hns,
            "coordinates_hns": coordinates_hns.astype(np.float32),
            "holo_coordinates": coordinates.astype(np.float32),#placeholder
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_pocket_coordinates": pocket_coordinates.astype(np.float32),#placeholder
            "smi": smi,
            "pocket": pocket,
            "affinity": affinity,
            "ori_mol_length": ori_mol_length,
            "ori_pocket_length": ori_pocket_length
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)

class AffinityTestDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms,
        coordinates,
        pocket_atoms,
        pocket_coordinates,
        affinity=None,
        is_train=False,
        pocket="pocket"
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.affinity = affinity
        self.is_train = is_train
        self.pocket=pocket
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def pocket_atom(self, atom):
        if atom[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        ori_length = len(atoms)
        #coordinates = self.dataset[index][self.coordinates]
        size = len(self.dataset[index][self.coordinates])
        with data_utils.numpy_seed(self.seed, epoch, index):
            sample_idx = np.random.randint(size)
        coordinates = self.dataset[index][self.coordinates][sample_idx]
        pocket_atoms = np.array(
            [self.pocket_atom(item) for item in self.dataset[index][self.pocket_atoms]]
        )
        #print(len(self.dataset[index][self.pocket_coordinates]))
        pocket_coordinates = np.stack(self.dataset[index][self.pocket_coordinates])

        smi = self.dataset[index]["smi"]
        pocket = self.dataset[index][self.pocket]
        affinity = self.dataset[index][self.affinity]
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "holo_coordinates": coordinates.astype(np.float32),#placeholder
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_pocket_coordinates": pocket_coordinates.astype(np.float32),#placeholder
            "smi": smi,
            "pocket": pocket,
            "affinity": affinity.astype(np.float32),
            "ori_length": ori_length
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class AffinityMolDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms,
        coordinates,
        is_train=False,
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.is_train = is_train
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def pocket_atom(self, atom):
        if atom[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        #print(self.dataset[index])
        atoms = np.array(self.dataset[index][self.atoms])
        ori_length = len(atoms)
        #coordinates = self.dataset[index][self.coordinates]
        size = len(self.dataset[index][self.coordinates])
        #print(size)
        with data_utils.numpy_seed(self.seed, epoch, index):
            sample_idx = np.random.randint(size)
        # check coordinates is 2 dimension or not
        if len(self.dataset[index][self.coordinates][sample_idx].shape) == 2:
            coordinates = self.dataset[index][self.coordinates][sample_idx]
        else:
            coordinates = self.dataset[index][self.coordinates]
        #coordinates = self.dataset[index][self.coordinates][sample_idx]
        #coordinates = self.dataset[index][self.coordinates]
        if "smi" in self.dataset[index]:
            smi = self.dataset[index]["smi"]
        else:
            smi = ""
        coordinates = np.array(coordinates)
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "holo_coordinates": coordinates.astype(np.float32),#placeholder
            "smi": smi,
            "ori_length": ori_length
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class AffinityPocketDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        pocket_atoms,
        pocket_coordinates,
        is_train=False,
        pocket="pocket"
    ):
        self.dataset = dataset
        self.seed = seed
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.is_train = is_train
        self.pocket=pocket
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def pocket_atom(self, atom):
        if atom[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        pocket_atoms = np.array(
            [self.pocket_atom(item) for item in self.dataset[index][self.pocket_atoms]]
        )
        ori_length = len(pocket_atoms)
        pocket_coordinates = np.stack(self.dataset[index][self.pocket_coordinates])
        if self.pocket in self.dataset[index]:
            pocket = self.dataset[index][self.pocket]
        else:
            pocket = ""
        return {
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_pocket_coordinates": pocket_coordinates.astype(np.float32),#placeholder
            "pocket": pocket,
            "ori_length": ori_length
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)

class AffinityValidDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms,
        coordinates,
        pocket_atoms,
        pocket_coordinates,
        pocket="pocket"
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.pocket=pocket
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def pocket_atom(self, atom):
        if atom[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        ori_mol_length = len(atoms)
        #coordinates = self.dataset[index][self.coordinates]

        size = len(self.dataset[index][self.coordinates])
        with data_utils.numpy_seed(self.seed, epoch, index):
            sample_idx = np.random.randint(size)
        coordinates = self.dataset[index][self.coordinates][sample_idx]
        pocket_atoms = np.array(
            [self.pocket_atom(item) for item in self.dataset[index][self.pocket_atoms]]
        )
        ori_pocket_length = len(pocket_atoms)
        pocket_coordinates = np.stack(self.dataset[index][self.pocket_coordinates])

        smi = self.dataset[index]["smi"]
        pocket = self.dataset[index][self.pocket]
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "holo_coordinates": coordinates.astype(np.float32),#placeholder
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_pocket_coordinates": pocket_coordinates.astype(np.float32),#placeholder
            "smi": smi,
            "pocket": pocket,
            "ori_mol_length": ori_mol_length,
            "ori_pocket_length": ori_pocket_length
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)