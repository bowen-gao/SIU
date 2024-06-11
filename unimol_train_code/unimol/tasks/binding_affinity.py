# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from IPython import embed as debug_embedded
import logging
import os
from collections.abc import Iterable
from sklearn.metrics import roc_auc_score
from xmlrpc.client import Boolean
import numpy as np
import torch
from tqdm import tqdm
from unicore import checkpoint_utils
import unicore
from unicore.data import (AppendTokenDataset, Dictionary, EpochShuffleDataset,
                          FromNumpyDataset, NestedDictionaryDataset,
                          PrependTokenDataset, RawArrayDataset,LMDBDataset, RawLabelDataset,
                          RightPadDataset, RightPadDataset2D, TokenizeDataset,SortDataset,data_utils)
from unicore.tasks import UnicoreTask, register_task
from unimol.data import (AffinityDataset, CroppingPocketDockingPoseDataset,CroppingPocketDockingPoseTestDataset,
                         CrossDistanceDataset, DistanceDataset,
                         EdgeTypeDataset, KeyDataset, LengthDataset,
                         NormalizeDataset, NormalizeDockingPoseDataset,
                         PrependAndAppend2DDataset, RemoveHydrogenDataset,
                         RemoveHydrogenPocketDataset, RightPadDatasetCoord,
                         RightPadDatasetCross2D, TTADockingPoseDataset, AffinityTestDataset, AffinityValidDataset, AffinityMolDataset, AffinityPocketDataset, ResamplingDataset)
#from skchem.metrics import bedroc_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment
from sklearn.metrics import roc_curve
logger = logging.getLogger(__name__)


def re_new(y_true, y_score, ratio):
    fp = 0
    tp = 0
    p = sum(y_true)
    n = len(y_true) - p
    num = ratio*n
    sort_index = np.argsort(y_score)[::-1]
    for i in range(len(sort_index)):
        index = sort_index[i]
        if y_true[index] == 1:
            tp += 1
        else:
            fp += 1
            if fp>= num:
                break
    return (tp*n)/(p*fp)


def calc_re(y_true, y_score, ratio_list):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    #print(fpr, tpr)
    res = {}
    res2 = {}
    total_active_compounds = sum(y_true)
    total_compounds = len(y_true)

    # for ratio in ratio_list:
    #     for i, t in enumerate(fpr):
    #         if t > ratio:
    #             #print(fpr[i], tpr[i])
    #             if fpr[i-1]==0:
    #                 res[str(ratio)]=tpr[i]/fpr[i]
    #             else:
    #                 res[str(ratio)]=tpr[i-1]/fpr[i-1]
    #             break
    
    for ratio in ratio_list:
        res2[str(ratio)] = re_new(y_true, y_score, ratio)

    #print(res)
    #print(res2)
    return res2

def cal_metrics(y_true, y_score, alpha):
    """
    Calculate BEDROC score.

    Parameters:
    - y_true: true binary labels (0 or 1)
    - y_score: predicted scores or probabilities
    - alpha: parameter controlling the degree of early retrieval emphasis

    Returns:
    - BEDROC score
    """
    
        # concate res_single and labels
    scores = np.expand_dims(y_score, axis=1)
    y_true = np.expand_dims(y_true, axis=1)
    scores = np.concatenate((scores, y_true), axis=1)
    # inverse sort scores based on first column
    scores = scores[scores[:,0].argsort()[::-1]]
    bedroc = CalcBEDROC(scores, 1, 80.5)
    count = 0
    # sort y_score, return index
    index  = np.argsort(y_score)[::-1]
    for i in range(int(len(index)*0.005)):
        if y_true[index[i]] == 1:
            count += 1
    auc = CalcAUC(scores, 1)
    ef_list = CalcEnrichment(scores, 1, [0.005, 0.01, 0.02, 0.05])
    ef = {
        "0.005": ef_list[0],
        "0.01": ef_list[1],
        "0.02": ef_list[2],
        "0.05": ef_list[3]
    }
    re_list = calc_re(y_true, y_score, [0.005, 0.01, 0.02, 0.05])
    return auc, bedroc, ef, re_list



@register_task("binding_affinity")
class BindingAffinity(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="downstream data path",
        )
        parser.add_argument(
            "--finetune-mol-model",
            default=None,
            type=str,
            help="pretrained molecular model path",
        )
        parser.add_argument(
            "--finetune-pocket-model",
            default=None,
            type=str,
            help="pretrained pocket model path",
        )
        parser.add_argument(
            "--dist-threshold",
            type=float,
            default=6.0,
            help="threshold for the distance between the molecule and the pocket",
        )
        parser.add_argument(
            "--max-pocket-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a pocket",
        )
        parser.add_argument(
            "--test-model",
            default=False,
            type=Boolean,
            help="whether test model",
        )
        parser.add_argument("--reg", action="store_true", help="regression task")

    def __init__(self, args, dictionary, pocket_dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.pocket_dictionary = pocket_dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.pocket_mask_idx = pocket_dictionary.add_symbol("[MASK]", is_special=True)


    @classmethod
    def setup_task(cls, args, **kwargs):
        mol_dictionary = Dictionary.load(os.path.join(args.data, "dict_mol.txt"))
        pocket_dictionary = Dictionary.load(os.path.join(args.data, "dict_pkt.txt"))
        logger.info("ligand dictionary: {} types".format(len(mol_dictionary)))
        logger.info("pocket dictionary: {} types".format(len(pocket_dictionary)))
        return cls(args, mol_dictionary, pocket_dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates','holo_coordinates','holo_pocket_coordinates','scaffold'
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """
        #print(1,split)
        data_path = os.path.join(self.args.data, split + ".lmdb")
        dataset = LMDBDataset(data_path)
        if split.startswith("train"):
            
            
            
            dataset = AffinityDataset(
                dataset,
                self.args.seed,
                "atoms",
                "coordinates",
                "pocket_atoms",
                "pocket_coordinates",
                "label",
                True,
                "source_data"
            )
            
        else:
            
            dataset = AffinityDataset(
                dataset,
                self.args.seed,
                "atoms",
                "coordinates",
                "pocket_atoms",
                "pocket_coordinates",
                "label",
                False,
                "source_data"
            )
            
            
        tgt_dataset = KeyDataset(dataset, "affinity_list")
        




        poc_dataset = KeyDataset(dataset, "pocket")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDockingPoseDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            self.args.max_pocket_atoms,
        )


        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)


        apo_dataset = NormalizeDataset(dataset, "coordinates")
        apo_dataset = NormalizeDataset(apo_dataset, "pocket_coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)

        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
        src_pocket_dataset = PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dictionary)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )


        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_tokens": RightPadDataset(
                        src_pocket_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "pocket_src_distance": RightPadDataset2D(
                        distance_pocket_dataset,
                        pad_idx=0,
                    ),
                    "pocket_src_edge_type": RightPadDataset2D(
                        pocket_edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_coord": RightPadDatasetCoord(
                        coord_pocket_dataset,
                        pad_idx=0,
                    ),
                },
                "target": {
                    "finetune_target": RawLabelDataset(tgt_dataset),
                },
                "pocket_name": RawArrayDataset(poc_dataset),
            },
        )
        # if split.startswith("train"):
        #     nest_dataset = EpochShuffleDataset(
        #         nest_dataset, len(nest_dataset), self.args.seed
        #     )
        # self.datasets[split] = nest_dataset
        if split == "train":
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_dataset))

            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
            self.datasets[split] = ResamplingDataset(
                self.datasets[split]
            )
        else:
            self.datasets[split] = nest_dataset
        #print(len(src_dataset))


    def load_test_dataset(self, **kwargs):
        """Load a given dataset split.
        'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates','holo_coordinates','holo_pocket_coordinates','scaffold'
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """

        #data_path = "/home/gaobowen/DrugClip/local_data/dude_official.lmdb"
        #data_path = "/data/protein/lib-pcba/process_lmdb/Lib-PCBA/lib_pcbaALDH1.lmdb"
        #data_path = "/data/protein/lib-pcba/process_lmdb/Lib-PCBA_single/PKM2.lmdb"
 
        dataset = LMDBDataset(data_path)
 
        dataset = AffinityTestDataset(
            dataset,
            self.args.seed,
            "atoms",
            "coordinates",
            "pocket_atom_type",
            "pocket_coord",
            "activity",
            False,
            "pocket_name"
        )
        smi_dataset = KeyDataset(dataset, "smi")
        poc_dataset = KeyDataset(dataset, "pocket")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            "holo_pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDockingPoseDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            "holo_pocket_coordinates",
            self.args.max_pocket_atoms,
        )
        dataset = RemoveHydrogenPocketDataset(
            dataset, "atoms", "coordinates", "holo_coordinates", True, True
        )

        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)


        apo_dataset = NormalizeDataset(dataset, "coordinates")
        apo_dataset = NormalizeDataset(apo_dataset, "pocket_coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)

        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
        src_pocket_dataset = PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dictionary)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_tokens": RightPadDataset(
                        src_pocket_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "pocket_src_distance": RightPadDataset2D(
                        distance_pocket_dataset,
                        pad_idx=0,
                    ),
                    "pocket_src_edge_type": RightPadDataset2D(
                        pocket_edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_coord": RightPadDatasetCoord(
                        coord_pocket_dataset,
                        pad_idx=0,
                    ),
                },
                "target": {
                    "finetune_target": RawArrayDataset(KeyDataset(dataset, "affinity")),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "pocket_name": RawArrayDataset(poc_dataset),
            },
        )
        return nest_dataset

    def load_pcba_dataset(self, name, **kwargs):
        """Load a given dataset split.
        'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates','holo_coordinates','holo_pocket_coordinates','scaffold'
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """

        #data_path = "/home/gaobowen/DrugClip/local_data/dude_official.lmdb"
        #data_path = "/data/protein/lib-pcba/process_lmdb/Lib-PCBA/lib_pcbaALDH1.lmdb"
        data_path = "/data/protein/lib-pcba/process_lmdb/Lib-PCBA_single/" + name + ".lmdb"
 
        dataset = LMDBDataset(data_path)
 
        dataset = AffinityTestDataset(
            dataset,
            self.args.seed,
            "atoms",
            "coordinates",
            "pocket_atoms",
            "pocket_coordinates",
            "activity",
            False,
            "pocket"
        )
        smi_dataset = KeyDataset(dataset, "smi")
        poc_dataset = KeyDataset(dataset, "pocket")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            "holo_pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDockingPoseDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            "holo_pocket_coordinates",
            self.args.max_pocket_atoms,
        )
        dataset = RemoveHydrogenPocketDataset(
            dataset, "atoms", "coordinates", "holo_coordinates", True, True
        )

        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)


        apo_dataset = NormalizeDataset(dataset, "coordinates")
        apo_dataset = NormalizeDataset(apo_dataset, "pocket_coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)

        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
        src_pocket_dataset = PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dictionary)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_tokens": RightPadDataset(
                        src_pocket_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "pocket_src_distance": RightPadDataset2D(
                        distance_pocket_dataset,
                        pad_idx=0,
                    ),
                    "pocket_src_edge_type": RightPadDataset2D(
                        pocket_edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_coord": RightPadDatasetCoord(
                        coord_pocket_dataset,
                        pad_idx=0,
                    ),
                },
                "target": {
                    "finetune_target": RawArrayDataset(KeyDataset(dataset, "affinity")),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "pocket_name": RawArrayDataset(poc_dataset),
            },
        )
        return nest_dataset


    def load_ns_mols_dataset(self, data_path, **kwargs):
        """Load a given dataset split.
        'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates','holo_coordinates','holo_pocket_coordinates','scaffold'
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """

        #data_path = "/home/gaobowen/DrugClip/local_data/dude_official.lmdb"
        #data_path = "/data/protein/lib-pcba/process_lmdb/Lib-PCBA/lib_pcbaALDH1.lmdb"
        #data_path = "/data/protein/lib-pcba/raw/lib_pcba/" + name + "/mols.lmdb"
 
        dataset = LMDBDataset(data_path)
        dataset = AffinityMolDataset(
            dataset,
            self.args.seed,
            "atoms",
            "coordinates",
            False,
        )
        
        smi_dataset = KeyDataset(dataset, "smi")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)



        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)


        apo_dataset = NormalizeDataset(dataset, "coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)


        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "mol_len": RawArrayDataset(len_dataset),
            },
        )
        return nest_dataset

    def load_mols_dataset_new(self, data_path,atoms,coords, **kwargs):
        #atom_key = 'atoms'
        #atom_key = 'atom_types'

        """Load a given dataset split.
        'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates','holo_coordinates','holo_pocket_coordinates','scaffold'
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """

        #data_path = "/home/gaobowen/DrugClip/local_data/dude_official.lmdb"
        #data_path = "/data/protein/lib-pcba/process_lmdb/Lib-PCBA/lib_pcbaALDH1.lmdb"
        #data_path = "/data/protein/lib-pcba/raw/lib_pcba/" + name + "/mols.lmdb"
 
        dataset = LMDBDataset(data_path)
        #label_dataset = KeyDataset(dataset, "label")
        #key_dataset = KeyDataset(dataset, "key")
        subset_dataset = KeyDataset(dataset, "subset")
        id_dataset = KeyDataset(dataset, "IDs")
        smi_dataset = KeyDataset(dataset, "smi")
        dataset = AffinityMolDataset(
            dataset,
            self.args.seed,
            atoms,
            coords,
            False,
        )
        
        

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)



        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)


        apo_dataset = NormalizeDataset(dataset, "coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)


        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                #"target":  RawArrayDataset(label_dataset),
                "mol_len": RawArrayDataset(len_dataset),
                #"key": RawArrayDataset(key_dataset),
                "id": RawArrayDataset(id_dataset),
                "subset": RawArrayDataset(subset_dataset),
            },
        )
        return nest_dataset

    def load_mols_dataset(self, data_path,atoms,coords, **kwargs):
        #atom_key = 'atoms'
        #atom_key = 'atom_types'

        """Load a given dataset split.
        'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates','holo_coordinates','holo_pocket_coordinates','scaffold'
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """

        #data_path = "/home/gaobowen/DrugClip/local_data/dude_official.lmdb"
        #data_path = "/data/protein/lib-pcba/process_lmdb/Lib-PCBA/lib_pcbaALDH1.lmdb"
        #data_path = "/data/protein/lib-pcba/raw/lib_pcba/" + name + "/mols.lmdb"
 
        dataset = LMDBDataset(data_path)
        #label_dataset = KeyDataset(dataset, "label")
        #key_dataset = KeyDataset(dataset, "key")
        #subset_dataset = KeyDataset(dataset, "subset")
        #id_dataset = KeyDataset(dataset, "IDs")
        smi_dataset = KeyDataset(dataset, "smi")
        dataset = AffinityMolDataset(
            dataset,
            self.args.seed,
            atoms,
            coords,
            False,
        )
        
        

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)



        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)


        apo_dataset = NormalizeDataset(dataset, "coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)


        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                #"target":  RawArrayDataset(label_dataset),
                "mol_len": RawArrayDataset(len_dataset),
                #"key": RawArrayDataset(key_dataset),
                #"id": RawArrayDataset(id_dataset),
                #"subset": RawArrayDataset(subset_dataset),
            },
        )
        return nest_dataset
    
    def load_gpcr_mols_dataset(self, data_path, **kwargs):
        """Load a given dataset split.
        'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates','holo_coordinates','holo_pocket_coordinates','scaffold'
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """

        #data_path = "/home/gaobowen/DrugClip/local_data/dude_official.lmdb"
        #data_path = "/data/protein/lib-pcba/process_lmdb/Lib-PCBA/lib_pcbaALDH1.lmdb"
        #data_path = "/data/protein/lib-pcba/raw/lib_pcba/" + name + "/mols.lmdb"
 
        dataset = LMDBDataset(data_path)
        #label_dataset = KeyDataset(dataset, "label")
        #id_dataset = KeyDataset(dataset, "id")
        key_dataset = KeyDataset(dataset, "key")
        dataset = AffinityMolDataset(
            dataset,
            self.args.seed,
            "atoms",
            "coordinates",
            False,
        )
        smi_dataset = KeyDataset(dataset, "smi")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)



        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)


        apo_dataset = NormalizeDataset(dataset, "coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)


        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "key": RawArrayDataset(key_dataset),
            },
        )
        return nest_dataset

    def load_pockets_dataset(self, data_path, **kwargs):
        """Load a given dataset split.
        'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates','holo_coordinates','holo_pocket_coordinates','scaffold'
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """

        #data_path = "/home/gaobowen/DrugClip/local_data/dude_official.lmdb"
        #data_path = "/data/protein/lib-pcba/process_lmdb/Lib-PCBA/lib_pcbaALDH1.lmdb"
        #data_path = "/data/protein/lib-pcba/raw/lib_pcba/" + name + "/pockets.lmdb"
 
        dataset = LMDBDataset(data_path)
 
        dataset = AffinityPocketDataset(
            dataset,
            self.args.seed,
            "pocket_atoms",
            "pocket_coordinates",
            False,
            "pocket"
        )
        poc_dataset = KeyDataset(dataset, "pocket")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            "holo_pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDockingPoseTestDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            "holo_pocket_coordinates",
            self.args.max_pocket_atoms,
        )




        apo_dataset = NormalizeDataset(dataset, "pocket_coordinates")



        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        len_dataset = LengthDataset(src_pocket_dataset)
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
        src_pocket_dataset = PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dictionary)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "pocket_src_tokens": RightPadDataset(
                        src_pocket_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "pocket_src_distance": RightPadDataset2D(
                        distance_pocket_dataset,
                        pad_idx=0,
                    ),
                    "pocket_src_edge_type": RightPadDataset2D(
                        pocket_edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_coord": RightPadDatasetCoord(
                        coord_pocket_dataset,
                        pad_idx=0,
                    ),
                },
                "pocket_name": RawArrayDataset(poc_dataset),
                "pocket_len": RawArrayDataset(len_dataset),
            },
        )
        return nest_dataset

    

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        
        if args.finetune_mol_model is not None:
            print("load pretrain model weight from...", args.finetune_mol_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_mol_model,
            )
            model.mol_model.load_state_dict(state["model"], strict=False)
            if args.arch == "binding_affinity_colbert":
                print("load mol_colbert_model weight")
                model.mol_colbert_model.load_state_dict(state["model"], strict=False)
            #model.mol_colbert_model.load_state_dict(state["model"], strict=False)
        if args.finetune_pocket_model is not None:
            print("load pretrain model weight from...", args.finetune_pocket_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_pocket_model,
            )
            model.pocket_model.load_state_dict(state["model"], strict=False)
            if args.arch == "binding_affinity_colbert":
                print("load pocket_colbert_model weight")
                model.pocket_colbert_model.load_state_dict(state["model"], strict=False)
            #model.pocket_colbert_model.load_state_dict(state["model"], strict=False)
        
        if args.arch == "binding_affinity_ns":
            model.global_data = torch.utils.data.DataLoader(self.global_dataset, batch_size=args.batch_size, collate_fn=self.global_dataset.collater, shuffle=True)

        return model

    def train_step(
        self, sample, model, loss, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *loss*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~unicore.data.UnicoreDataset`.
            model (~unicore.models.BaseUnicoreModel): the model
            loss (~unicore.losses.UnicoreLoss): the loss
            optimizer (~unicore.optim.UnicoreOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        if self.args.test_model == True:
            print("???")
            model.eval()
            #self.test_target_fish(model)
            #self.test_pcba(model)
            #self.fold_encode(model)
            #self.test_dude(model)
            self.retrieval_multi_folds(model)
            #self.test_yaoming(model)
            #self.find_mols(model)
            #self.encode_mols(model)
            #self.encode_pockets(model)
            #self.test_kinase(model)
            #self.test_kinase_new(model)
            #self.huice_1031(model)
            #self.retrieve_mols(model)
            exit()
        

        #find all file names end with .lmdb
        # import glob
        # lmdb_files = glob.glob("/data/protein/gpcr/gpcr_pockets/*.lmdb")
        # total_count = 0.0
        # total_pos = 0.0
        # total_random_count = 0.0
        # auc_list = []
        # ef_list = []
        # ef_random_list = []
        # pairs = []
        # with open("/data/protein/gpcr/gt.txt", "r") as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         lis = line.strip().split(',')
        #         name = lis[0]
        #         uniprot = lis[1]
        #         key = lis[2]
        #         # check whether file start with uniprot in lmdb_files
        #         lmdb_file = [file for file in lmdb_files if file.startswith("/data/protein/gpcr/gpcr_pockets/"+uniprot)]
        #         if len(lmdb_file) == 0:
        #             continue
        #         else:
        #             pairs.append((name, uniprot, key))
        # print(len(pairs))
        # labels_all = []
        # preds_all = []
        # pockets_all = []
        # keys_all = []
        # dic = {}
        # import fnmatch
        # files = []
        # for root, dirnames, filenames in os.walk("/data/protein/gpcr/gpcr_pockets/"):
        #     for filename in fnmatch.filter(filenames, '*.lmdb'):
        #         files.append(os.path.join(root, filename))

        # #for pair in tqdm(pairs):
        # for filename in tqdm(files):
        #     print(filename)
        #     preds, pocket_names = self.test_gpcr(filename, model)


        #     for i in range(len(preds)):
        #         preds_all.extend(preds[i])
        #         pockets_all.extend([pocket_names[i]]*len(preds[i]))
        #         keys_all.extend(self.keys)
        
        # # sort preds_all and return the index of top 10% largest

 
        # # sort preds_all and return the index of top 10% largest
        # sorted_index = np.argsort(preds_all)[-int(len(preds_all)*0.01):]
        # keys_save = [keys_all[i] for i in sorted_index]
        # pockets_save = [pockets_all[i] for i in sorted_index]
        # preds_save = [preds_all[i] for i in sorted_index]

        # with open("gpcr_top1%.txt", "w") as f:
        #     for i in range(len(pockets_save)):
        #         f.write(str(pockets_save[i])+","+str(keys_save[i]) +","+str(preds_save[i])+"\n")

        # exit()


        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = loss(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output
    
    def valid_step(self, sample, model, loss, test=False):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = loss(model, sample)
        return loss, sample_size, logging_output

    
    def test_model(self, model):
        num_data = len(self.test_dataset)
        bsz=48
        print(num_data//bsz)
        test_data = torch.utils.data.DataLoader(self.test_dataset, batch_size=bsz, collate_fn=self.test_dataset.collater)
        generator = iter(test_data)
        pred_dic = {}
        target_dic = {}
        label_dic = {}

        for itr, sample in enumerate((tqdm(test_data))):
            
            sample = next(generator)
            sample = unicore.utils.move_to_cuda(sample)
            pockets = sample["pocket_name"]
            smiles = sample["smi_name"]
            targets = sample["target"]["finetune_target"]
            #print(len(pockets))
            net_output = model(
                **sample["net_input"],
                features_only=True,
                is_train=False
            )
            #logit_output = torch.diagonal(net_output).detach().cpu().numpy()
            logit_output = net_output.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            #print(logit_output.shape)
            for i in range(len(pockets)):
                pocket = pockets[i]
                
                #print(pocket)
                smi = smiles[i]
                pred = logit_output[i]
                #print(pred)
                target = targets[i]
                if pocket in pred_dic:
                    pred_dic[pocket].append((smi, pred))
                else:
                    pred_dic[pocket]=[(smi, pred)]
                if pocket in label_dic:
                    label_dic[pocket].append(target)
                else:
                    label_dic[pocket]=[target]
                if target == 1:
                    if pocket in target_dic:
                        target_dic[pocket].add(smi)
                    else:
                        target_dic[pocket]=set([smi])
            



        all = 0
        re = 0
        lis=[]
        res = []
        auc_lis = []
        for pocket in pred_dic:
            pred_list = pred_dic[pocket]
            preds = np.array([x[1] for x in pred_list])
            targets = np.array(label_dic[pocket])
            print(pocket,preds, targets)
            try:
                auc = roc_auc_score(targets, preds)
                print(pocket, auc)
                auc_lis.append(auc)
            except:
                continue
            pred_list = sorted(pred_list, key = lambda x: x[1], reverse=True)
            count=0
            for i in range(len(pred_list)//100):
                if pred_list[i][0] in target_dic[pocket]:
                    count+=1
            print(pocket, len(target_dic[pocket]), count)
            all+=len(target_dic[pocket])
            re+=count
            lis.append(float(count)/len(target_dic[pocket]))
            res.append([pocket, str(len(target_dic[pocket])), str(count)])
        
        lis = np.array(lis)
        auc_lis = np.array(auc_lis)
        print(float(re)/all)
        print(np.mean(lis), np.median(lis), np.std(lis))
        print(np.mean(auc_lis), np.median(auc_lis), np.std(auc_lis))
        # with open('result_2023_03-09_10-01-26.txt', 'w') as f:
        #     for i, item in enumerate(res):
        #         f.write(item[0]+","+item[1]+","+item[2])
        #         if i!=101:
        #             f.write('\n')


    def valid_model(self, model):
        num_data = len(self.valid_dataset)
        print(num_data)
        valid_data = torch.utils.data.DataLoader(self.valid_dataset, batch_size=32, collate_fn=self.valid_dataset.collater)
        all_samples = next(iter(valid_data))
        mols = all_samples["smi_name"]
        print(len(set(mols)))
        pred_dic = {}
        target_dic = {}
        count = 0
        for _, sample in enumerate(valid_data):
            sample = unicore.utils.move_to_cuda(sample)
            net_output = model(
                **sample["net_input"],
                features_only=True,
            )
            logit_output = torch.argmax(net_output, dim=1).detach().cpu().numpy()
            count += sum(logit_output==np.arange(len(net_output)))
        print(float(count)/num_data)
        # for _, sample in enumerate(tqdm(valid_data)):
        #     #sample = next(generator)
        #     sample = unicore.utils.move_to_cuda(sample)
        #     pockets = sample["pocket_name"]
        #     smiles = sample["smi_name"]
        #     targets = sample["target"]["finetune_target"]
        #     #print(len(pockets))
        #     net_output = model(
        #         **sample["net_input"],
        #         features_only=True,
        #     )
        #     logit_output = torch.diagonal(net_output).detach().cpu().numpy()
        #     #print(logit_output.shape)
        #     for i in range(len(pockets)):
        #         pocket = pockets[i]
        #         #print(pocket)
        #         smi = smiles[i]
        #         pred = logit_output[i]
        #         #print(pred)
        #         target = targets[i]
        #         if pocket in pred_dic:
        #             pred_dic[pocket].append((smi, pred))
        #         else:
        #             pred_dic[pocket]=[(smi, pred)]
        #         if target == 1:
        #             if pocket in target_dic:
        #                 target_dic[pocket].add(smi)
        #             else:
        #                 target_dic[pocket]=set([smi])

        # for pocket in pred_dic:
        #     pred_list = pred_dic[pocket]
        #     pred_list = sorted(pred_list, key = lambda x: x[1], reverse=True)
        #     count=0
        #     for i in range(len(pred_list)//100):
        #         if pred_list[i][0] in target_dic[pocket]:
        #             count+=1
        #     print(pocket, len(target_dic[pocket]), count)

    def test_pcba_target(self, name, pocket_select, model, **kwargs):
        """Encode a dataset with the molecule encoder."""

        #names = "PPARG"
        data_path = "/data/protein/lib-pcba/raw/lib_pcba/" + name + "/mols.lmdb"
        mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
        num_data = len(mol_dataset)
        bsz=64
        print(num_data//bsz)
        mol_reps = []
        mol_names = []
        labels = []
        
        # generate mol data
        
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)
        for _, sample in enumerate(tqdm(mol_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["mol_src_distance"]
            et = sample["net_input"]["mol_src_edge_type"]
            st = sample["net_input"]["mol_src_tokens"]
            mol_padding_mask = st.eq(model.mol_model.padding_idx)
            mol_x = model.mol_model.embed_tokens(st)
            
            n_node = dist.size(-1)
            gbf_feature = model.mol_model.gbf(dist, et)

            gbf_result = model.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            mol_outputs = model.mol_model.encoder(
                mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
            )
            mol_encoder_rep = mol_outputs[0][:,0,:]
            mol_emb = model.mol_project(mol_encoder_rep)
            mol_emb = mol_emb / mol_emb.norm(dim=1, keepdim=True)
            mol_emb = mol_emb.detach().cpu().numpy()
            mol_reps.append(mol_emb)
            mol_names.extend(sample["smi_name"])
            labels.extend(sample["target"].detach().cpu().numpy())
        mol_reps = np.concatenate(mol_reps, axis=0)
        #print(mol_reps[0][:10])
        labels = np.array(labels, dtype=np.int32)
        # generate pocket data
        data_path = "/data/protein/lib-pcba/raw/lib_pcba/" + name + "/pockets.lmdb"
        pocket_dataset = self.load_pockets_dataset(data_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz, collate_fn=pocket_dataset.collater)
        pocket_reps = []

        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
            pocket_x = model.pocket_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.pocket_model.gbf(dist, et)
            gbf_result = model.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            pocket_outputs = model.pocket_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
            )
            pocket_encoder_rep = pocket_outputs[0][:,0,:]
            pocket_emb = model.pocket_project(pocket_encoder_rep)
            pocket_emb = pocket_emb / pocket_emb.norm(dim=1, keepdim=True)
            pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_names = sample["pocket_name"]
            # for index, pocket_name in enumerate(pocket_names):
            #     if pocket_name == pocket_select:
            #         pocket_reps.append(pocket_emb[index])
            pocket_reps.append(pocket_emb)
        pocket_reps = np.concatenate(pocket_reps, axis=0)
        #pocket_reps = pocket_reps[0].reshape(1,-1)
        print(pocket_reps.shape)
        #print(pocket_reps[0][:10])
        res = pocket_reps @ mol_reps.T

        # new_res = []
        # indexes = np.random.randint(len(res), size=len(res[0]))
        # for i, index in enumerate(indexes):
        #     new_res.append(res[index, i])
        # print("111", roc_auc_score(labels, np.array(new_res)))
        
        # auc_list = []
        print(name)
        # medians = np.median(res, axis=1, keepdims=True)
        # # get mad for each row
        # mads = np.median(np.abs(res - medians), axis=1, keepdims=True)
        # # get z score
        # res = 0.6745 * (res - medians) / (mads + 1e-6)
        res_single = res.max(axis=0)
        auc, bedroc, ef_list, re_list = cal_metrics(labels, res_single, 80.5)

        return auc, bedroc, ef_list, re_list
    
    def test_pcba_target_colbert(self, name, model, **kwargs):
        """Encode a dataset with the molecule encoder."""

        #names = "PPARG"
        data_path = "/data/protein/lib-pcba/raw/lib_pcba/" + name + "/mols.lmdb"
        mol_dataset = self.load_mols_dataset(data_path)
        num_data = len(mol_dataset)
        bsz=64
        print(num_data//bsz)

        data_path = "/data/protein/lib-pcba/raw/lib_pcba/" + name + "/pockets.lmdb"
        pocket_dataset = self.load_pockets_dataset(data_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz, collate_fn=pocket_dataset.collater)
        pocket_reps = []
        pocket_lengths = []
        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            pocket_padding_mask = st.eq(model.pocket_colbert_model.padding_idx)
            pocket_x = model.pocket_colbert_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.pocket_colbert_model.gbf(dist, et)
            gbf_result = model.pocket_colbert_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            pocket_outputs = model.pocket_colbert_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
            )
            pocket_encoder_rep = pocket_outputs[0]
            pocket_emb = pocket_encoder_rep #
            pocket_emb = model.pocket_project_t(pocket_emb)
            pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
            #pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_reps.append(pocket_emb)
            pocket_len = sample["pocket_len"]
            #pocket_len = pocket_len.detach().cpu().numpy()
            pocket_lengths.append(pocket_len)
        pocket_reps =torch.cat(pocket_reps, dim=0)
        pocket_len = torch.cat(pocket_lengths, dim=0)


        mol_reps = []
        mol_names = []
        labels = []
        # generate mol data
        res_list = []
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)
        for _, sample in enumerate(tqdm(mol_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["mol_src_distance"]
            et = sample["net_input"]["mol_src_edge_type"]
            st = sample["net_input"]["mol_src_tokens"]
            mol_padding_mask = st.eq(model.mol_colbert_model.padding_idx)
            mol_x = model.mol_colbert_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.mol_colbert_model.gbf(dist, et)
            gbf_result = model.mol_colbert_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            mol_outputs = model.mol_colbert_model.encoder(
                mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
            )
            mol_encoder_rep = mol_outputs[0]
            mol_emb = mol_encoder_rep #model.mol_project_t(mol_encoder_rep)
            mol_emb = model.mol_project_t(mol_emb)
            mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
            mol_len = sample["mol_len"]
            # extend orilength to mol_lengths
            # raw_sim_matrix = torch.einsum('bik,tjk->btij', pocket_reps, mol_emb)
            # b,t,i,j = raw_sim_matrix.shape
            # row_mask=torch.arange(0, i).unsqueeze(1).repeat(1, j).repeat(b,t,1,1).cuda()
            # pocket_len_mask=pocket_len.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,t,i,j)
            # row_mask=row_mask<pocket_len_mask

            # col_mask=torch.arange(0, j).unsqueeze(0).repeat(i, 1).repeat(b,t,1,1).cuda()
            # clen_mask=mol_len.unsqueeze(1).unsqueeze(1).repeat(b,1,i,j)
            # col_mask=col_mask<clen_mask
            # mask=row_mask*col_mask
            # masked_sim_mx=raw_sim_matrix*mask
            # max_sim=masked_sim_mx.max(-1)[0]
            # len_mx=pocket_len.unsqueeze(1).repeat(1,t)
            # res=max_sim.sum(-1)/len_mx
            #print(pocket_reps.shape, mol_emb.shape)
            #res = model.late_interaction(pocket_len, mol_len, pocket_reps[:,1:,:], mol_emb[:,1:,:])
            #print(pocket_reps[:,0,:].shape, mol_emb[:,0,:].shape, res.shape)
            #res = res.T
            res = torch.matmul(pocket_reps[:,0,:], mol_emb[:,0,:].T)
            res = res.detach().cpu().numpy()
            res_list.append(res)

            labels.extend(sample["target"].detach().cpu().numpy())
        #mol_reps = np.concatenate(mol_reps, axis=0)
        labels = np.array(labels, dtype=np.int32)
        #mol_len = np.array(mol_lengths, dtype=np.int32)
        # generate pocket data
        
        #print(pocket_reps.shape)
        #res = pocket_reps @ mol_reps.T

        res = np.concatenate(res_list, axis=1)
        print(res.shape)
        # new_res = []
        # indexes = np.random.randint(len(res), size=len(res[0]))
        # for i, index in enumerate(indexes):
        #     new_res.append(res[index, i])
        # print("111", roc_auc_score(labels, np.array(new_res)))
        
        # auc_list = []

        
        print(name)
        res_single = res.max(axis=0)
        auc, bedroc, ef_list, re = cal_metrics(labels, res_single, 80.5)
        
        return auc, bedroc, ef_list, re
    

    def test_pcba(self, model, **kwargs):
        ckpt_date = self.args.finetune_from_model.split("/")[-2]
        save_name = "/home/gaobowen/DrugClip/test_results/pcba/" + ckpt_date + ".txt"
        
        targets = os.listdir("/data/protein/lib-pcba/raw/lib_pcba/")
        protein_dict = {
            'ADRB2': '4lde',
            'ALDH1': '5l2n',
            'VDR': '3a2j',
            'ESR1_ago': '2qzo',
            'ESR1_ant': '5ufx',
            'GBA': '2v3d',
            'IDH1': '4umx',
            'KAT2A': '5mlj',
            'MAPK1': '4zzn',
            'MTORC1': '4dri',
            'OPRK1': '6b73',
            'PKM2': '3gr4',
            'PPARG': '3b1m',
            'TP53': '3zme',
            'FEN1': '5fv7'
        }
        lis = ['2q5s', '3hod', '3r8a', '5two', '1zgy', '4prg', '5tto', '2i4j', '4fgy', '5z5s', '2p4y', '3b1m', '4ci5', '5y2t', '2yfe']

        print(targets)
        auc_list = []
        ef_list = []
        bedroc_list = []
        # for i in range(len(lis)):
        #     target="PPARG"
        #     auc, ef = self.test_pcba_target(target,lis[i], model)
        #     auc_list.append(auc)
        #     ef_list.append(ef)
        #     print(ef)
        re_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": []
        }
        ef_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": []
        }
        for target in targets:
            #target="PPARG"
            if self.args.arch == "binding_affinity_colbert_test":
                auc, bedroc, ef, re = self.test_pcba_target_colbert(target, model)
            else:
                auc, bedroc, ef, re = self.test_pcba_target(target, protein_dict[target], model)
            auc_list.append(auc)
            bedroc_list.append(bedroc)
            for key in ef:
                ef_list[key].append(ef[key])
            print("re", re)
            print("ef", ef)
            for key in re:
                re_list[key].append(re[key])
        print(auc_list)
        print(ef_list)
        print("auc 25%", np.percentile(auc_list, 25))
        print("auc 50%", np.percentile(auc_list, 50))
        print("auc 75%", np.percentile(auc_list, 75))
        print("auc mean", np.mean(auc_list))
        print("bedroc 25%", np.percentile(bedroc_list, 25))
        print("bedroc 50%", np.percentile(bedroc_list, 50))
        print("bedroc 75%", np.percentile(bedroc_list, 75))
        print("bedroc mean", np.mean(bedroc_list))
        #print(np.median(auc_list))
        #print(np.median(ef_list))
        for key in ef_list:
            print("ef", key, "25%", np.percentile(ef_list[key], 25))
            print("ef",key, "50%", np.percentile(ef_list[key], 50))
            print("ef",key, "75%", np.percentile(ef_list[key], 75))
            print("ef",key, "mean", np.mean(ef_list[key]))
        for key in re_list:
            print("re",key, "25%", np.percentile(re_list[key], 25))
            print("re",key, "50%", np.percentile(re_list[key], 50))
            print("re",key, "75%", np.percentile(re_list[key], 75))
            print("re",key, "mean", np.mean(re_list[key]))
        # save targets, auc_list, ef_list to txt file
        
        with open(save_name, "a") as f:
            f.write("target, auc, ef\n")
            for i in range(len(targets)):
                f.write(targets[i]+","+str(auc_list[i])+","+str(ef_list[i])+"\n")
            #f.write("median auc: "+str(np.median(auc_list))+"\n")
            #f.write("median ef: "+str(np.median(ef_list))+"\n")
        return 

    def test_dude_target_colbert(self, target, model, **kwargs):

        #names = "PPARG"
        data_path = "/data/protein/DUD-E/raw/all/" + target + "/mols.lmdb"
        mol_dataset = self.load_mols_dataset(data_path)
        num_data = len(mol_dataset)
        bsz=64
        print(num_data//bsz)
        mol_reps = []
        mol_names = []
        labels = []
        
        data_path = "/data/protein/DUD-E/raw/all/" + target + "/pocket.lmdb"        
        pocket_dataset = self.load_pockets_dataset(data_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz, collate_fn=pocket_dataset.collater)
        pocket_reps = []
        pocket_lengths = []
        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            pocket_padding_mask = st.eq(model.pocket_colbert_model.padding_idx)
            pocket_x = model.pocket_colbert_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.pocket_colbert_model.gbf(dist, et)
            gbf_result = model.pocket_colbert_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            pocket_outputs = model.pocket_colbert_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
            )
            pocket_encoder_rep = pocket_outputs[0][:,0:,:]
            pocket_emb = model.pocket_project_t(pocket_encoder_rep)
            pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
            #pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_reps.append(pocket_emb)
            pocket_len = sample["pocket_len"]
            #pocket_len = pocket_len.detach().cpu().numpy()
            pocket_lengths.append(pocket_len)
        pocket_reps =torch.cat(pocket_reps, dim=0)
        pocket_len = torch.cat(pocket_lengths, dim=0)


        mol_reps = []
        mol_names = []
        labels = []
        # generate mol data
        res_list = []
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)
        for _, sample in enumerate(tqdm(mol_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["mol_src_distance"]
            et = sample["net_input"]["mol_src_edge_type"]
            st = sample["net_input"]["mol_src_tokens"]
            mol_padding_mask = st.eq(model.mol_colbert_model.padding_idx)
            mol_x = model.mol_colbert_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.mol_colbert_model.gbf(dist, et)
            gbf_result = model.mol_colbert_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            mol_outputs = model.mol_colbert_model.encoder(
                mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
            )
            mol_encoder_rep = mol_outputs[0][:,0:,:]
            mol_emb = model.mol_project_t(mol_encoder_rep)
            mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
            mol_len = sample["mol_len"]
            # extend orilength to mol_lengths
            # raw_sim_matrix = torch.einsum('bik,tjk->btij', pocket_reps, mol_emb)
            # b,t,i,j = raw_sim_matrix.shape
            # row_mask=torch.arange(0, i).unsqueeze(1).repeat(1, j).repeat(b,t,1,1).cuda()
            # pocket_len_mask=pocket_len.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,t,i,j)
            # row_mask=row_mask<pocket_len_mask

            # col_mask=torch.arange(0, j).unsqueeze(0).repeat(i, 1).repeat(b,t,1,1).cuda()
            # clen_mask=mol_len.unsqueeze(1).unsqueeze(1).repeat(b,1,i,j)
            # col_mask=col_mask<clen_mask
            # mask=row_mask*col_mask
            # masked_sim_mx=raw_sim_matrix*mask
            # max_sim=masked_sim_mx.max(-1)[0]
            # len_mx=pocket_len.unsqueeze(1).repeat(1,t)
            # res=max_sim.sum(-1)/len_mx
            #print(pocket_reps.shape, mol_emb.shape)
            #res = model.late_interaction(pocket_len, mol_len, pocket_reps[:,1:,:], mol_emb[:,1:,:])
            #res = res.T
            res = pocket_reps[:,0,:] @ mol_emb[:,0,:].T
            res = res.detach().cpu().numpy()
            res_list.append(res)

            labels.extend(sample["target"].detach().cpu().numpy())
        #mol_reps = np.concatenate(mol_reps, axis=0)
        labels = np.array(labels, dtype=np.int32)
        #mol_len = np.array(mol_lengths, dtype=np.int32)
        # generate pocket data
        
        #print(pocket_reps.shape)
        #res = pocket_reps @ mol_reps.T

        res = np.concatenate(res_list, axis=1)
        print(res.shape)
        # new_res = []
        # indexes = np.random.randint(len(res), size=len(res[0]))
        # for i, index in enumerate(indexes):
        #     new_res.append(res[index, i])
        # print("111", roc_auc_score(labels, np.array(new_res)))
        
        # auc_list = []

        res_single = res.max(axis=0)
        


        print(np.sum(labels), len(labels)-np.sum(labels))
        auc, bedroc, ef, re = cal_metrics(labels, res_single, 80.5)
        
        
        print(target)
        

        return auc, bedroc, ef, re, res_single, labels
    
    def test_dude_target(self, target, model, **kwargs):

        data_path = "/data/protein/DUD-E/raw/all/" + target + "/mols.lmdb"
        mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
        num_data = len(mol_dataset)
        bsz=64
        print(num_data//bsz)
        mol_reps = []
        mol_names = []
        labels = []
        
        # generate mol data
        
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)
        for _, sample in enumerate(tqdm(mol_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["mol_src_distance"]
            et = sample["net_input"]["mol_src_edge_type"]
            st = sample["net_input"]["mol_src_tokens"]
            mol_padding_mask = st.eq(model.mol_model.padding_idx)
            mol_x = model.mol_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.mol_model.gbf(dist, et)
            gbf_result = model.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            mol_outputs = model.mol_model.encoder(
                mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
            )
            mol_encoder_rep = mol_outputs[0][:,0,:]
            mol_emb = mol_encoder_rep
            mol_emb = model.mol_project(mol_encoder_rep)
            mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
            #print(mol_emb.dtype)
            mol_emb = mol_emb.detach().cpu().numpy()
            #print(mol_emb.dtype)
            mol_reps.append(mol_emb)
            mol_names.extend(sample["smi_name"])
            labels.extend(sample["target"].detach().cpu().numpy())
        mol_reps = np.concatenate(mol_reps, axis=0)
        labels = np.array(labels, dtype=np.int32)
        # generate pocket data
        data_path = "/data/protein/DUD-E/raw/all/" + target + "/pocket.lmdb"
        pocket_dataset = self.load_pockets_dataset(data_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz, collate_fn=pocket_dataset.collater)
        pocket_reps = []

        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
            pocket_x = model.pocket_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.pocket_model.gbf(dist, et)
            gbf_result = model.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            pocket_outputs = model.pocket_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
            )
            pocket_encoder_rep = pocket_outputs[0][:,0,:]
            #pocket_emb = pocket_encoder_rep
            pocket_emb = model.pocket_project(pocket_encoder_rep)
            pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
            pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_reps.append(pocket_emb)
        pocket_reps = np.concatenate(pocket_reps, axis=0)
        print(pocket_reps.shape)
        res = pocket_reps @ mol_reps.T

        # new_res = []
        # indexes = np.random.randint(len(res), size=len(res[0]))
        # for i, index in enumerate(indexes):
        #     new_res.append(res[index, i])
        # print("111", roc_auc_score(labels, np.array(new_res)))
        
        # auc_list = []
        # medians = np.median(res, axis=1, keepdims=True)
        # # get mad for each row
        # mads = np.median(np.abs(res - medians), axis=1, keepdims=True)
        # # get z score
        # res = 0.6745 * (res - medians) / (mads + 1e-6)
        res_single = res.max(axis=0)
        # normalize res_single
        #res_single = (res_single - res_single.min()) / (res_single.max() - res_single.min())
        #res_single = np.random.rand(len(res_single))
        
        auc, bedroc, ef_list, re_list = cal_metrics(labels, res_single, 80.5)
        
        
        print(target)

        print(np.sum(labels), len(labels)-np.sum(labels))

        return auc, bedroc, ef_list, re_list, res_single, labels

    def test_dude(self, model, **kwargs):
        #ckpt_date = self.args.finetune_from_model.split("/")[-2]
        #save_name = "/home/gaobowen/DrugClip/test_results/dude/" + ckpt_date + ".txt"
        #print(save_name)
        targets = os.listdir("/data/protein/DUD-E/raw/all/")
        with open("/data/protein/DUD-E/raw/fold3_same/test.txt", "r") as f:
            targets = f.readlines()
        targets = [target.strip() for target in targets]
        targets = os.listdir("/data/protein/DUD-E/raw/all/")
        print(targets)
        print(len(targets))
        auc_list = []
        bedroc_list = []
        ef_list = []
        res_list= []
        labels_list = []
        re_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": [],
        }
        ef_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": [],
        }
        for i,target in enumerate(targets):
            print(i,target)
            if self.args.arch == "binding_affinity_colbert_test":
                auc, bedroc, ef, re, res_single, labels = self.test_dude_target_colbert(target, model)
            else:
                auc, bedroc, ef, re, res_single, labels = self.test_dude_target(target, model)
            auc_list.append(auc)
            bedroc_list.append(bedroc)
            for key in ef:
                ef_list[key].append(ef[key])
            for key in re_list:
                re_list[key].append(re[key])
            print(auc)
            print(bedroc)
            print(ef)
            print(re)
            res_list.append(res_single)
            labels_list.append(labels)
        res = np.concatenate(res_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        #auc_all, bedroc_all, ef_all, re_all = cal_metrics(labels, res, 80.5)

        #print("all auc", roc_auc_score(labels, res))
        #print("all re", roc_auc_score(labels, res, max_fpr=0.01))
        #print("all re", re_all)
        #print("all bedroc", bedroc_all)
        #print("all ef", ef_all)
        print(auc_list)
        print(ef_list)
        #print(np.median(auc_list))
        #print(np.median(ef_list))
        print("auc 25%", np.percentile(auc_list, 25))
        print("auc 50%", np.percentile(auc_list, 50))
        print("auc 75%", np.percentile(auc_list, 75))
        print("auc mean", np.mean(auc_list))
        print("bedroc 25%", np.percentile(bedroc_list, 25))
        print("bedroc 50%", np.percentile(bedroc_list, 50))
        print("bedroc 75%", np.percentile(bedroc_list, 75))
        print(bedroc_list)
        print("bedroc mean", np.mean(bedroc_list))

        for key in ef_list:
            print("ef", key, "25", np.percentile(ef_list[key], 25))
            print("ef", key, "50", np.percentile(ef_list[key], 50))
            print("ef", key, "75", np.percentile(ef_list[key], 75))
            print("ef", key, "mean", np.mean(ef_list[key]))

        for key in re_list:
            print("re", key, "25",  np.percentile(re_list[key], 25))
            print("re", key, "50",  np.percentile(re_list[key], 50))
            print("re", key, "75",  np.percentile(re_list[key], 75))
            print("re", key, "mean",  np.mean(re_list[key]))

        # save targets, auc_list, ef_list to txt file
        save_name = "dude_pocket.txt"
        with open(save_name, "w") as f:
            f.write("target, auc, ef\n")
            for i in range(len(targets)):
                f.write(targets[i]+","+str(auc_list[i])+","+str(ef_list["0.01"][i])+"\n")
            #f.write("median auc: "+str(np.median(auc_list))+"\n")
            #f.write("median ef: "+str(np.median(ef_list))+"\n")
        return
    
    def test_target_fish(self, model, **kwargs):

        #names = "PPARG"
        data_path = "/data/protein/local_data/drug_clip_pdb_general/target_fish_mol.lmdb"
        mol_dataset = self.load_mols_dataset(data_path)
        num_data = len(mol_dataset)
        bsz=32
        print(num_data//bsz)
        mol_reps = []
        mol_names = []
        labels = []
        
        # generate mol data
        
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)
        for _, sample in enumerate(tqdm(mol_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["mol_src_distance"]
            et = sample["net_input"]["mol_src_edge_type"]
            st = sample["net_input"]["mol_src_tokens"]
            mol_padding_mask = st.eq(model.mol_model.padding_idx)
            mol_x = model.mol_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.mol_model.gbf(dist, et)
            gbf_result = model.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            mol_outputs = model.mol_model.encoder(
                mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
            )
            mol_encoder_rep = mol_outputs[0][:,0,:]
            mol_emb = model.mol_project(mol_encoder_rep)
            mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
            mol_emb = mol_emb.detach().cpu().numpy()
            mol_reps.append(mol_emb)
            mol_names.extend(sample["smi_name"])
            labels.extend(sample["target"].detach().cpu().numpy())
        mol_reps = np.concatenate(mol_reps, axis=0)
        print(mol_reps.shape)
        labels = np.array(labels, dtype=np.int32)
        # generate pocket data
        data_path = "/data/protein/local_data/drug_clip_pdb_general/target_fish_pocket.lmdb"
        pocket_dataset = self.load_pockets_dataset(data_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz, collate_fn=pocket_dataset.collater)
        pocket_reps = []

        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
            pocket_x = model.pocket_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.pocket_model.gbf(dist, et)
            gbf_result = model.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            pocket_outputs = model.pocket_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
            )
            pocket_encoder_rep = pocket_outputs[0][:,0,:]
            pocket_emb = model.pocket_project(pocket_encoder_rep)
            pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
            pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_reps.append(pocket_emb)
        pocket_reps = np.concatenate(pocket_reps, axis=0)
        #save pocket_reps and mol_reps 
        np.save("pocket_reps_cl.npy", pocket_reps)
        np.save("mol_reps_cl.npy", mol_reps)
        print(pocket_reps.shape)
        res = pocket_reps @ mol_reps.T
        #res = mol_reps @ pocket_reps.T
        print(res)
        print(res.shape)

        dic = {}
        # for mol in range(len(res)):
        #     for pocket in range(len(res[mol])):
        #         if mol not in dic:
        #             dic[mol] = []
        #         dic[mol].append((res[mol][pocket], pocket))
        


        # for mol in dic:
        #     dic[mol].sort(key=lambda x:x[0], reverse=True)
        
        for pocket in range(len(res)):
            for mol in range(len(res[pocket])):
                if pocket not in dic:
                    dic[pocket] = []
                dic[pocket].append((res[pocket][mol], mol))
        
        for pocket in dic:
            dic[pocket].sort(key=lambda x:x[0], reverse=True)

        
        # def topk_acc(res,k):
        #     count=0
        #     for mol in dic:
        #         for i in range(k):
        #             if dic[mol][i][1]==mol:
        #                 count+=1
        #                 break
        #     return count/len(dic)

        def topk_acc(res,k):
            count=0
            for pocket in dic:
                for i in range(k):
                    if dic[pocket][i][1]==pocket:
                        count+=1
                        break
            return count/len(dic)

        for i in range(5):
            print(i+1, topk_acc(dic,i+1))
        
        # plot distribution of dic[mol]
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # sns.set()

        # for mol in tqdm(dic):
        #     score_lis = [x[0] for x in dic[mol]]
        #     #print(score_lis)
        #     # nomralize it into -1 to 1
        #     score_lis = [(x-min(score_lis))/(max(score_lis)-min(score_lis))*2-1 for x in score_lis]
        #     # save score_lis
        #     with open("tf_plot/{}_rg.txt".format(mol_names[mol]), "w") as f:
        #         for x in score_lis:
        #             f.write(str(x)+"\n")
            # plot distribution line
            # sns.distplot(score_lis, bins=100)
            

            # #plt.hist(score_lis, bins=100)
            # plt.savefig("tf_plot/{}.png".format(mol))
            # plt.clf()



        
        return 

    def retrieval_multi_folds(self, model, **kwargs):
        prot_names = "D2R"
        prot_names = ["5HT2A", "CGRPR", "CRBN", "D2R", "Fascin", "GluN2A", "GluN2B", "MAPK10", "NLRP3", "NSD3", "TRPM8", "TRPV1", "ULK1", "VHL", "XIAP"]
        prot_names = ["NET"]
        # ckpts = [
        #     "/data/protein/save_dir/affinity/2023-07-18_22-10-49/checkpoint_best.pt",
        #     "/data/protein/save_dir/affinity/2023-07-18_22-19-53/checkpoint_best.pt",
        #     "/data/protein/save_dir/affinity/2023-07-19_11-03-28/checkpoint_best.pt",
        #     "/data/protein/save_dir/affinity/2023-07-19_11-04-06/checkpoint_best.pt",
        # ]

        ckpts = [
            "/drug/save_dir/affinity/2023-07-19_22-45-24/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-07-19_22-46-21/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-07-20_12-10-53/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-07-20_12-10-34/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-07-20_16-39-11/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-07-20_19-20-03/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-07-21_12-19-42/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-07-21_12-19-59/checkpoint_best.pt",
        ]


        ckpts = [
            "/drug/save_dir/affinity/2023-07-21_20-21-48/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-07-21_20-21-26/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-07-21_23-56-17/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-07-21_23-56-38/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-07-22_09-07-34/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-07-22_09-07-55/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-07-22_23-57-07/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-07-22_23-57-30/checkpoint_best.pt",
        ]

        #ckpts = ["/drug/save_dir/affinity/2023-07-24_20-18-38/checkpoint_best.pt"]
        #ckpts = ["/drug/save_dir/affinity/2023-07-24_20-19-14/checkpoint_best.pt"]

        #ckpts=["/data/protein/save_dir/affinity/2023-04-04_11-51-27/checkpoint_best.pt"]
        ckpts = ["/data/protein/save_dir/affinity/2023-05-05_00-17-09/checkpoint_best.pt"]
        ckpts = ["/drug/save_dir/affinity/2023-10-25_11-28-51/checkpoint_best.pt"]
        ckpts=["/drug/save_dir/affinity/2023-10-25_17-00-46/checkpoint_best.pt"]
        ckpts=["/drug/save_dir/affinity/2023-10-25_18-20-26/checkpoint_best.pt"] #load
        ckpts=["/drug/save_dir/affinity/2023-10-25_18-21-22/checkpoint_best.pt"] #no load

        # 6 folds
        ckpts = [
            "/drug/save_dir/affinity/2023-12-06_20-39-17/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-12-06_23-23-23/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-12-07_10-25-59/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-12-07_14-46-18/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-12-07_22-30-21/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-12-08_11-21-09/checkpoint_best.pt",
        ]



        prot_res_dic = {}
        for prot_name in prot_names:
            prot_res_dic[prot_name] = []
        prot_res_dic_ret = {}
        for prot_name in prot_names:
            prot_res_dic_ret[prot_name] = []
        
        prot_labels = {}
        prot_all_names = {}
        prot_ret_names = {}
        prot_active_names = {}
        prot_ret_smiles = {}
        prefix = "/drug/drugclip_tmp/0724_final/"
        prefix = "/drug/drugclip_tmp/yaoming_6folds/"
        prefix = "/drug/DrugCLIP_chemdata_v2024/embs/"
        prefix = "/drug/drugclip_tmp/yaoming_6folds/"
        prefix = "/drug/drugclip_tmp/0724_final/"
        prefix = "/drug/drugclip_tmp/yaoming_6folds/"
        prefix = "/drug/DrugCLIP_chemdata_v2024/embs/"
        #prefix = "/drug/drugclip_tmp/0724_final/"
        caches = ["fold0.pkl", "fold1.pkl", "fold2.pkl", "fold3.pkl", "fold4.pkl", "fold5.pkl", "fold6.pkl", "fold7.pkl"]
        caches = ["fold0.pkl", "fold1.pkl", "fold2.pkl", "fold3.pkl", "fold4.pkl", "fold5.pkl"]
        caches = [prefix + cache for cache in caches]
        #caches = ["tmp2.pkl"]
        for fold, ckpt in enumerate(ckpts):

            state = checkpoint_utils.load_checkpoint_to_cpu(ckpt)
            model.load_state_dict(state["model"], strict=False)

            #prot_names = ["XIAP"]
            for prot_name in prot_names:
                mol_data_path = "/home/gaobowen/he_pockets/chemdiv.lmdb"
                #mol_data_path = "/drug/data/files/chem_lib.lmdb"
                mol_data_path = "/drug/encoding_mols/chemdiv_1640k.lmdb"
                mol_data_path = "/drug/encoding_mols/chemdiv_1640k_both.lmdb"
                mol_data_path = "/drug/DrugCLIP_chemdata_v2024/DrugCLIP_mols_v2024.lmdb"
                # pocket_data_path = f"/data/protein/DrugClip/Validation_data/{prot_name}/PDB/pocket.lmdb"
                # active_path = f"/data/protein/DrugClip/Validation_data/{prot_name}/ChEMBL/active.lmdb"
                # inactive_path = f"/data/protein/DrugClip/Validation_data/{prot_name}/ChEMBL/inactive.lmdb"
                # res_path = f"/data/protein/DrugClip/Validation_data/{prot_name}/result.txt"

                pocket_data_path = "/home/jiayinjun/Chem_Aware_Interaction_Understanding/data/NET/PDB/pocket.lmdb"
                pocket_data_path = "/drug/DrugCLIP_chemdata_v2024/GS/GammaSecretase_Pocket2_round2.lmdb"
                pocket_data_path = "/drug/DrugCLIP_chemdata_v2024/NAV/Nav18_p1.lmdb"
                pocket_data_path = "/drug/DrugCLIP_chemdata_v2024/IGE/IgE_p1r1.lmdb"
                pocket_data_path = "/drug/DrugCLIP_chemdata_v2024/NAV/flex_r2.lmdb"
                pocket_data_path = "/drug/DrugCLIP_chemdata_v2024/MCT/mct4.lmdb"
                pocket_data_path = "/drug/DrugCLIP_chemdata_v2024/NAV/R2O_pocket.lmdb"
                pocket_data_path = "/drug/DrugCLIP_chemdata_v2024/MCT/MCT4New.lmdb"
                pocket_data_path = "/drug/DrugCLIP_chemdata_v2024/NAV/R2O_rosIFD.lmdb"
                pocket_data_path = "/data/protein/DrugClip/Validation_data/5HT2A/PDB/pocket.lmdb"
                pocket_data_path = "/drug/DrugCLIP_chemdata_v2024/others/Lilrb3_round2.lmdb"
                pocket_data_path = "/drug/DrugCLIP_chemdata_v2024/others/RNP_round2.lmdb"
                pocket_data_path = "/home/jiayinjun/Chem_Aware_Interaction_Understanding/data/NET/PDB/pocket.lmdb"
                pocket_data_path = "/drug/DrugCLIP_chemdata_v2024/NET/NET_YCY.lmdb"
                pocket_data_path = "/home/jiayinjun/DrugCLIP_screen_pipeline/targets/Nav18_0517/Nav18_0517.lmdb"
                active_path = "/home/jiayinjun/Chem_Aware_Interaction_Understanding/data/NET/ChEMBL/active.lmdb"
                inactive_path = "/home/jiayinjun/Chem_Aware_Interaction_Understanding/data/NET/ChEMBL/inactive.lmdb"
                active_path = "/drug/DrugCLIP_chemdata_v2024/GS/active.lmdb"
                active_path = "/drug/DrugCLIP_chemdata_v2024/NAV/active.lmdb"
                active_path = "/drug/DrugCLIP_chemdata_v2024/NET/NET_active.lmdb"
                inactive_path = "/drug/DrugCLIP_chemdata_v2024/NET/NET_inactive.lmdb"
                #active_path = "/drug/DrugCLIP_chemdata_v2024/MCT/active_mct4.lmdb"
                #inactive_path = "/drug/DrugCLIP_chemdata_v2024/GS/inactive.lmdb"
                res_path = "net.txt"

                #names = "PPARG"
                import pickle
                mol_dataset = self.load_mols_dataset_new(mol_data_path, "atoms", "coordinates")
                num_data = len(mol_dataset)
                bsz=64
                print(num_data//bsz)
                mol_reps = []
                mol_names = []
                labels = []
                mol_ids_subsets = []
                
                # generate mol data
                mol_cache_path = "/data/protein/DrugClip/Validation_data/saved_embs/mol_reps_0628_1.pkl"
                #mol_cache_path = "/data/protein/DrugClip/Validation_data/saved_embs/mol_reps_0703.pkl"
                #mol_cache_path = "/home/gaobowen/he_pockets/mol_reps_defaultmodel.pkl"
                mol_cache_path = "/data/protein/DrugClip/Validation_data/saved_embs/mol_reps_0703_no_pre.pkl"
                mol_cache_path = "/data/protein/DrugClip/Validation_data/saved_embs/mol_reps_0704_32.pkl"
                mol_cache_path = "/data/protein/DrugClip/Validation_data/saved_embs/mol_reps_0705_24.pkl"
                mol_cache_path = "/data/protein/DrugClip/Validation_data/saved_embs/mol_reps_0705_32.pkl"
                mol_cache_path = "/data/protein/DrugClip/Validation_data/saved_embs/mol_reps_0706_32.pkl"
                mol_cache_path = "tmp_30.pkl"
                mol_cache_path=caches[fold]
                if os.path.exists(mol_cache_path):
                    with open(mol_cache_path, "rb") as f:
                        mol_reps, mol_names = pickle.load(f)
                else:
                    mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)
                    for _, sample in enumerate(tqdm(mol_data)):
                        sample = unicore.utils.move_to_cuda(sample)
                        dist = sample["net_input"]["mol_src_distance"]
                        et = sample["net_input"]["mol_src_edge_type"]
                        st = sample["net_input"]["mol_src_tokens"]
                        mol_padding_mask = st.eq(model.mol_model.padding_idx)
                        mol_x = model.mol_model.embed_tokens(st)
                        n_node = dist.size(-1)
                        gbf_feature = model.mol_model.gbf(dist, et)
                        gbf_result = model.mol_model.gbf_proj(gbf_feature)
                        graph_attn_bias = gbf_result
                        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                        graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                        mol_outputs = model.mol_model.encoder(
                            mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
                        )
                        mol_encoder_rep = mol_outputs[0][:,0,:]
                        mol_emb = model.mol_project(mol_encoder_rep)
                        mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
                        mol_emb = mol_emb.detach().cpu().numpy()
                        mol_reps.append(mol_emb)
                        #mol_names.extend(sample["smi_name"])
                        mol_names.extend(sample["smi_name"])
                        ids = sample["id"]
                        subsets = sample["subset"]
                        # merge ids and subsets with comma
                        ids_subsets = [ids[i] + ";" + subsets[i] for i in range(len(ids))]
                        mol_ids_subsets.extend(ids_subsets)
                        #print(mol_names[:10])
                        #labels.extend(sample["target"].detach().cpu().numpy())
                        #keys.extend(sample["key"])
                    mol_reps = np.concatenate(mol_reps, axis=0)
                    with open(mol_cache_path, "wb") as f:
                        pickle.dump([mol_reps, mol_ids_subsets], f)
                print(mol_reps.shape)
                #labels = np.array(labels, dtype=np.int32)
                for i in range(len(mol_reps)):
                    labels.append(0)

                # generate activate data
                active_mol_dataset = self.load_mols_dataset(active_path, "atoms", "coordinates")
                active_mol_reps = []
                active_mol_names = []
                active_mol_data = torch.utils.data.DataLoader(active_mol_dataset, batch_size=bsz, collate_fn=active_mol_dataset.collater)
                for _, sample in enumerate(tqdm(active_mol_data)):
                    sample = unicore.utils.move_to_cuda(sample)
                    dist = sample["net_input"]["mol_src_distance"]
                    et = sample["net_input"]["mol_src_edge_type"]
                    st = sample["net_input"]["mol_src_tokens"]
                    mol_padding_mask = st.eq(model.mol_model.padding_idx)
                    mol_x = model.mol_model.embed_tokens(st)
                    n_node = dist.size(-1)
                    gbf_feature = model.mol_model.gbf(dist, et)
                    gbf_result = model.mol_model.gbf_proj(gbf_feature)
                    graph_attn_bias = gbf_result
                    graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                    graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                    mol_outputs = model.mol_model.encoder(
                        mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
                    )
                    mol_encoder_rep = mol_outputs[0][:,0,:]
                    mol_emb = model.mol_project(mol_encoder_rep)
                    mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
                    mol_emb = mol_emb.detach().cpu().numpy()
                    active_mol_reps.append(mol_emb)
                    active_mol_names.extend(sample["smi_name"])
                    #labels.extend(sample["target"].detach().cpu().numpy())
                    #keys.extend(sample["key"])
                active_mol_reps = np.concatenate(active_mol_reps, axis=0)
                print(active_mol_reps.shape)
                for i in range(len(active_mol_reps)):
                    labels.append(1)
                
                # generate inactivate data
                inactive_mol_dataset = self.load_mols_dataset(inactive_path, "atoms", "coordinates")
                inactive_mol_reps = []
                inactive_mol_names = []
                inactive_mol_data = torch.utils.data.DataLoader(inactive_mol_dataset, batch_size=bsz, collate_fn=inactive_mol_dataset.collater)
                for _, sample in enumerate(tqdm(inactive_mol_data)):
                    sample = unicore.utils.move_to_cuda(sample)
                    dist = sample["net_input"]["mol_src_distance"]
                    et = sample["net_input"]["mol_src_edge_type"]
                    st = sample["net_input"]["mol_src_tokens"]
                    mol_padding_mask = st.eq(model.mol_model.padding_idx)
                    mol_x = model.mol_model.embed_tokens(st)
                    n_node = dist.size(-1)
                    gbf_feature = model.mol_model.gbf(dist, et)
                    gbf_result = model.mol_model.gbf_proj(gbf_feature)
                    graph_attn_bias = gbf_result
                    graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                    graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                    mol_outputs = model.mol_model.encoder(
                        mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
                    )
                    mol_encoder_rep = mol_outputs[0][:,0,:]
                    mol_emb = model.mol_project(mol_encoder_rep)
                    mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
                    mol_emb = mol_emb.detach().cpu().numpy()
                    inactive_mol_reps.append(mol_emb)
                    inactive_mol_names.extend(sample["smi_name"])
                    #labels.extend(sample["target"].detach().cpu().numpy())
                    #keys.extend(sample["key"])
                inactive_mol_reps = np.concatenate(inactive_mol_reps, axis=0)
                print(inactive_mol_reps.shape)
                for i in range(len(inactive_mol_reps)):
                    labels.append(0)
                
                if prot_name not in prot_labels:
                    prot_labels[prot_name] = labels
                #inactive_mol_reps = []
                #all_mol_reps = np.concatenate([mol_reps, active_mol_reps, inactive_mol_reps], axis=0)
                all_mol_reps = np.concatenate([mol_reps, active_mol_reps, inactive_mol_reps], axis=0)
                #all_mol_reps = mol_reps
                all_names = mol_names + active_mol_names #+ inactive_mol_names
                if prot_name not in prot_all_names:
                    prot_all_names[prot_name] = all_names
                    prot_active_names[prot_name] = active_mol_names
                    prot_ret_names[prot_name] = mol_names
                    #prot_ret_smiles[prot_name] = 
                    #print(mol_names)


                # generate pocket data
                pocket_dataset = self.load_pockets_dataset(pocket_data_path)
                pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=16, collate_fn=pocket_dataset.collater)
                pocket_reps = []

                for _, sample in enumerate(tqdm(pocket_data)):
                    sample = unicore.utils.move_to_cuda(sample)
                    dist = sample["net_input"]["pocket_src_distance"]
                    et = sample["net_input"]["pocket_src_edge_type"]
                    st = sample["net_input"]["pocket_src_tokens"]
                    pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
                    pocket_x = model.pocket_model.embed_tokens(st)
                    n_node = dist.size(-1)
                    gbf_feature = model.pocket_model.gbf(dist, et)
                    gbf_result = model.pocket_model.gbf_proj(gbf_feature)
                    graph_attn_bias = gbf_result
                    graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                    graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                    pocket_outputs = model.pocket_model.encoder(
                        pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
                    )
                    pocket_encoder_rep = pocket_outputs[0][:,0,:]
                    pocket_emb = model.pocket_project(pocket_encoder_rep)
                    pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
                    pocket_emb = pocket_emb.detach().cpu().numpy()
                    pocket_reps.append(pocket_emb)
                pocket_reps = np.concatenate(pocket_reps, axis=0)
                print(pocket_reps.shape)
                # change reps to fp32
                all_mol_reps = all_mol_reps.astype(np.float32)
                mol_reps = mol_reps.astype(np.float32)
                pocket_reps = pocket_reps.astype(np.float32)
                prot_res_dic[prot_name].append(pocket_reps @ all_mol_reps.T)                
                prot_res_dic_ret[prot_name].append(pocket_reps @ mol_reps.T)


        for prot_name in prot_res_dic:
            '''
            res = np.array(prot_res_dic[prot_name])
            
            for i in range(len(res)):
                print('FOLD ',i)
                res_cur = res[i]
                # robust z score

                res_max = np.max(res_cur, axis=0)

                medians = np.median(res_cur, axis=1, keepdims=True)
                # get mad for each row
                mads = np.median(np.abs(res_cur - medians), axis=1, keepdims=True)
                # get z score
                res_cur = 0.6745 * (res_cur - medians) / (mads + 1e-6)
                # get max for each column
                
                
                
                res_max = np.max(res_cur, axis=0)

                # calculate bedroc with res_max and labels
                all_names = prot_all_names[prot_name]
                active_mol_names = prot_active_names[prot_name]

                auc, bedroc, ef_list, re_list = cal_metrics(labels, res_max, 80.5)
                print("aucs: ", auc, bedroc)
                print("ef_list: ", ef_list)
            print("ensemble")
            res_cur = np.mean(res, axis=0) 
            medians = np.median(res_cur, axis=1, keepdims=True)
            # get mad for each row
            mads = np.median(np.abs(res_cur - medians), axis=1, keepdims=True)
            # get z score
            res_cur = 0.6745 * (res_cur - medians) / (mads + 1e-6)
            # get max for each column
            res_max = np.max(res_cur, axis=0)

            # calculate bedroc with res_max and labels
            all_names = prot_all_names[prot_name]
            active_mol_names = prot_active_names[prot_name]

            auc, bedroc, ef_list, re_list = cal_metrics(labels, res_max, 80.5)
            print("aucs: ", auc, bedroc)
            print("ef_list: ", ef_list)
            '''
            # get names of all hits

            # lis = []
            # for i, score in enumerate(res_max):
            #     lis.append((score, all_names[i]))
            # lis.sort(key=lambda x:x[0], reverse=True)
            # # get top 10% of the list
            # n_top = len(all_names) // 100
            # n_hit = 0
            # n_active = len(active_mol_names)
            # hit_names = set()
            # for i in range(n_top):
            #     score = lis[i][0]
            #     key = lis[i][1]
            #     if key in active_mol_names and key not in hit_names:
            #         n_hit += 1
            #         hit_names.add(key)
            
            # print(n_hit)
            # save hit_names
            # with open("hit_names.txt", "w") as f:
            #     for name in hit_names:
            #         f.write(name+"\n")



                #res_max = res[0]


                # lis = []
                # for i,score in enumerate(res[0]):
                #     lis.append((score, mol_names[i]))
                # lis.sort(key=lambda x:x[0], reverse=True)
                # count = 0
                # res_list = []
                # for i in range(500):
                #     score = lis[i][0]
                #     #key = lis[i][2]
                #     smi = lis[i][1]
                #     # if key not in gt:
                #     #     res_list.append((smi, score))
                #     # else:
                #     #     count+=1
                #     res_list.append((smi, score))
        
                # print(count)
                # save res_list to txt
                # with open(res_path, "w") as f:
                #     for smi, score in res_list:
                #         f.write(smi+"\t"+str(score)+"\n")

                # lis = []
                # for i, score in enumerate(res_max):
                #     lis.append((score, all_names[i]))
                # lis.sort(key=lambda x:x[0], reverse=True)
                # #debug_embedded()
                # n_top = len(all_names) // 100
                # n_hit = 0
                # n_active = len(active_mol_names)
                # hit_names = set()
                # with open(res_path, "w") as f:
                #     for i in range(n_top):
                #         score = lis[i][0]
                #         key = lis[i][1]
                #         if key in active_mol_names and key not in hit_names:
                #             n_hit += 1
                #             hit_names.add(key)
                #         f.write(f"{key}\t{score}\t{key in active_mol_names}\n")
                #     str_res = (
                #         f"[{prot_name}] n_top: {n_top}, "
                #         f"n_active: {len(active_mol_names)}, "
                #         f"n_hit: {n_hit}, "
                #         f"ef1%: {n_hit / n_active  * 100}%\n"
                #     )
                #     #f.write(str_res)
                #     print(str_res)
                
                # for k in [5, 10,50,100]:
                #     n_top = k
                #     n_hit = 0
                #     for i in range(n_top):
                #         score = lis[i][0]
                #         key = lis[i][1]
                #         if key in active_mol_names:
                #             n_hit += 1
                #     print(k, float(n_hit) / float(k))
            
            

            res_new = prot_res_dic_ret[prot_name]
            res_new = np.mean(res_new, axis=0)
            medians = np.median(res_new, axis=1, keepdims=True)
            # get mad for each row
            mads = np.median(np.abs(res_new - medians), axis=1, keepdims=True)
            # get z score
            res_new = 0.6745 * (res_new - medians) / (mads + 1e-6)
            # get max for each column
            #res_max = np.max(res_cur, axis=0)
            res_max = np.max(res_new, axis=0)
            ret_names = prot_ret_names[prot_name]
            #ret_smiles = prot_ret_smiles[prot_name]
            # save res_max and ret_names to txt file
            lis = []
            for i, score in enumerate(res_max):
                lis.append((score, ret_names[i]))
            lis.sort(key=lambda x:x[0], reverse=True)
            lis = lis[:30000]
            #print(lis[:10])
            # save to txt, with ","
            res_path = "/drug/DrugCLIP_chemdata_v2024/NAV/nav_0517.txt"
            #res_path = "/home/gaobowen/DrugCLIP_old/test_tmp_8folds_no.txt"
            with open(res_path, "w") as f:
                for score, name in lis:
                    f.write(f"{name},{score}\n")

        return
    

    def test_yaoming(self, model, **kwargs):
        

        # 6 folds
        ckpts = [
            "/drug/save_dir/affinity/2023-12-06_20-39-17/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-12-06_23-23-23/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-12-07_10-25-59/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-12-07_14-46-18/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-12-07_22-30-21/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-12-08_11-21-09/checkpoint_best.pt",
        ]

        
        prefix = "/drug/drugclip_tmp/yaoming_6folds/"
        prefix = "/drug/drugclip_tmp/yaoming_6folds_chembridge/"
        caches = ["fold0.pkl", "fold1.pkl", "fold2.pkl", "fold3.pkl", "fold4.pkl", "fold5.pkl"]
        caches = [prefix + cache for cache in caches]
        #caches = ["tmp2.pkl"]


        

        mol_reps_list = []
        pocket_reps_list = []
        for fold, ckpt in enumerate(ckpts):

            state = checkpoint_utils.load_checkpoint_to_cpu(ckpt)
            model.load_state_dict(state["model"], strict=False)

            #prot_names = ["XIAP"]

            # generate pocket data
            pocket_data_path = "/home/gaobowen/he_pockets/wuxi_pocket.lmdb"
            pocket_data_path = "/home/jiayinjun/Chem_Aware_Interaction_Understanding/data/NET/PDB/pocket.lmdb"
            pocket_dataset = self.load_pockets_dataset(pocket_data_path)
            pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=16, collate_fn=pocket_dataset.collater)
            pocket_reps = []
            pocket_names = []

            for _, sample in enumerate(tqdm(pocket_data)):
                sample = unicore.utils.move_to_cuda(sample)
                dist = sample["net_input"]["pocket_src_distance"]
                et = sample["net_input"]["pocket_src_edge_type"]
                st = sample["net_input"]["pocket_src_tokens"]
                pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
                pocket_x = model.pocket_model.embed_tokens(st)
                n_node = dist.size(-1)
                gbf_feature = model.pocket_model.gbf(dist, et)
                gbf_result = model.pocket_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                pocket_outputs = model.pocket_model.encoder(
                    pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
                )
                pocket_encoder_rep = pocket_outputs[0][:,0,:]
                pocket_emb = model.pocket_project(pocket_encoder_rep)
                pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
                pocket_emb = pocket_emb.detach().cpu().numpy()
                pocket_reps.append(pocket_emb)
                pocket_names.extend(sample["pocket_name"])
            pocket_reps = np.concatenate(pocket_reps, axis=0)
            print(pocket_reps.shape)
            print(pocket_names)
            pocket_reps_list.append(pocket_reps)

            mol_data_path = "/drug/encoding_mols/chemdiv_1640k_new.lmdb"

            mol_data_path = "/drug/ChemBridge/ChemBridge.lmdb"
            


            #names = "PPARG"
            mol_dataset = self.load_mols_dataset(mol_data_path, "atoms", "coordinates")
            num_data = len(mol_dataset)
            bsz=64
            print(num_data//bsz)
            mol_reps = []
            mol_names = []
            mol_ids = []
                
            import pickle
            mol_cache_path=caches[fold]
            if os.path.exists(mol_cache_path):
                with open(mol_cache_path, "rb") as f:
                    mol_reps, mol_names = pickle.load(f)
            else:
                mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)
                for _, sample in enumerate(tqdm(mol_data)):
                    
                    sample = unicore.utils.move_to_cuda(sample)
                    dist = sample["net_input"]["mol_src_distance"]
                    et = sample["net_input"]["mol_src_edge_type"]
                    st = sample["net_input"]["mol_src_tokens"]
                    mol_padding_mask = st.eq(model.mol_model.padding_idx)
                    mol_x = model.mol_model.embed_tokens(st)
                    n_node = dist.size(-1)
                    gbf_feature = model.mol_model.gbf(dist, et)
                    gbf_result = model.mol_model.gbf_proj(gbf_feature)
                    graph_attn_bias = gbf_result
                    graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                    graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                    mol_outputs = model.mol_model.encoder(
                        mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
                    )
                    mol_encoder_rep = mol_outputs[0][:,0,:]
                    mol_emb = model.mol_project(mol_encoder_rep)
                    mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
                    mol_emb = mol_emb.detach().cpu().numpy()
                    mol_reps.append(mol_emb)
                    mol_names.extend(sample["smi_name"])
                    mol_ids.extend(sample["id"])
                    #print(sample["id"])
                    #print(mol_names[:10])
                    #labels.extend(sample["target"].detach().cpu().numpy())
                    #keys.extend(sample["key"])
                mol_reps = np.concatenate(mol_reps, axis=0)
                import pickle
                with open(mol_cache_path, "wb") as f:
                    pickle.dump([mol_reps, mol_names, mol_ids], f)
            print(mol_reps.shape)
            mol_reps_list.append(mol_reps)

        # extend dims of mol_reps_list and concatenate

        mol_reps_list = [mol_reps.reshape(mol_reps.shape[0], 1, mol_reps.shape[1]) for mol_reps in mol_reps_list]

        mol_reps_all = np.concatenate(mol_reps_list, axis=1)
        
        print(mol_reps_all.shape)
        # mean pooling of mol_reps_all
        mol_reps_all = np.mean(mol_reps_all, axis=1)
        pocket_reps_list = [pocket_reps.reshape(pocket_reps.shape[0], 1, pocket_reps.shape[1]) for pocket_reps in pocket_reps_list]
        pocket_reps_all = np.concatenate(pocket_reps_list, axis=1)
        print(pocket_reps_all.shape)
        # mean pooling of pocket_reps_all
        pocket_reps_all = np.mean(pocket_reps_all, axis=1)

        res = pocket_reps_all @ mol_reps_all.T

        print(res.shape)

        medians = np.median(res, axis=1, keepdims=True)
        # get mad for each row
        mads = np.median(np.abs(res - medians), axis=1, keepdims=True)
        # get z score
        res = 0.6745 * (res - medians) / (mads + 1e-6)

        # save res, pocket_names, mol_names
        res = np.max(res, axis=0)
        top1_idx = np.argsort(res)[::-1][:int(len(res) * 0.01)]

        with open("res_net.txt", "w") as f:
            for idx in top1_idx:
                # score and smi and id
                f.write(f"{res[idx]}\t{mol_names[idx]}\t{mol_ids[idx]}\n")

            

        
                

            
  
            
        
        return

    
    def fold_encode(self, model, **kwargs):


        

        # 6 folds
        ckpts = [
            "/drug/save_dir/affinity/2023-12-06_20-39-17/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-12-06_23-23-23/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-12-07_10-25-59/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-12-07_14-46-18/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-12-07_22-30-21/checkpoint_best.pt",
            "/drug/save_dir/affinity/2023-12-08_11-21-09/checkpoint_best.pt",
        ]


        

        # pocket_rep_folds = []
        # pocket_names = []
        # for fold, ckpt in enumerate(ckpts):

        #     state = checkpoint_utils.load_checkpoint_to_cpu(ckpt)
        #     model.load_state_dict(state["model"], strict=False)

            
                


        #     # generate pocket data
        #     pocket_data_path = "/drug/human_pockets_6A_filtered.lmdb"
        #     pocket_dataset = self.load_pockets_dataset(pocket_data_path)
        #     pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=16, collate_fn=pocket_dataset.collater)
        #     pocket_reps = []
            
        #     for _, sample in enumerate(tqdm(pocket_data)):
        #         sample = unicore.utils.move_to_cuda(sample)
        #         dist = sample["net_input"]["pocket_src_distance"]
        #         et = sample["net_input"]["pocket_src_edge_type"]
        #         st = sample["net_input"]["pocket_src_tokens"]
        #         pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
        #         pocket_x = model.pocket_model.embed_tokens(st)
        #         n_node = dist.size(-1)
        #         gbf_feature = model.pocket_model.gbf(dist, et)
        #         gbf_result = model.pocket_model.gbf_proj(gbf_feature)
        #         graph_attn_bias = gbf_result
        #         graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
        #         graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
        #         pocket_outputs = model.pocket_model.encoder(
        #             pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
        #         )
        #         pocket_encoder_rep = pocket_outputs[0][:,0,:]
        #         pocket_emb = model.pocket_project(pocket_encoder_rep)
        #         pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
        #         pocket_emb = pocket_emb.detach().cpu().numpy()
        #         pocket_reps.append(pocket_emb)
        #         if fold == 0:
        #             pocket_names.extend(sample["pocket_name"])
        #     pocket_reps = np.concatenate(pocket_reps, axis=0)
        #     print(pocket_reps.shape)
        #     pocket_rep_folds.append(pocket_reps)
        
        # pocket_reps = np.concatenate(pocket_rep_folds, axis=1)
        # print(pocket_reps.shape)
        # print(pocket_names[:10])

        # # save pocket_reps and pocket_names
        # import pickle
        # with open("/drug/admet/pocket_reps_6folds.pkl", "wb") as f:
        #     pickle.dump([pocket_names, pocket_reps], f)

        mol_ids = []
        mols_reps_folds = []
        mol_subsets = []
        for fold, ckpt in enumerate(ckpts):

            state = checkpoint_utils.load_checkpoint_to_cpu(ckpt)
            model.load_state_dict(state["model"], strict=False)


            # generate mol data

            mol_data_path = "/drug/admet/HepToxMols_train_val.lmdb"
            mol_data_path = "/drug/DrugCLIP_chemdata_v2024/DrugCLIP_mols_v2024.lmdb"

            mol_dataset = self.load_mols_dataset(mol_data_path, "atoms", "coordinates")

            mol_reps = []

            

            mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=64, collate_fn=mol_dataset.collater)
            for _, sample in enumerate(tqdm(mol_data)):
                sample = unicore.utils.move_to_cuda(sample)
                dist = sample["net_input"]["mol_src_distance"]
                et = sample["net_input"]["mol_src_edge_type"]
                st = sample["net_input"]["mol_src_tokens"]
                mol_padding_mask = st.eq(model.mol_model.padding_idx)
                mol_x = model.mol_model.embed_tokens(st)
                n_node = dist.size(-1)
                gbf_feature = model.mol_model.gbf(dist, et)
                gbf_result = model.mol_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                mol_outputs = model.mol_model.encoder(
                    mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
                )
                mol_encoder_rep = mol_outputs[0][:,0,:]
                mol_emb = model.mol_project(mol_encoder_rep)
                mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
                mol_emb = mol_emb.detach().cpu().numpy()
                mol_reps.append(mol_emb)
                if fold == 0:
                    mol_ids.extend(sample["id"])
                    mol_subsets.extend(sample["subset"])
            mol_reps = np.concatenate(mol_reps, axis=0)

            mols_reps_folds.append(mol_reps)

        
        print(mol_reps.shape)

        mol_reps = np.concatenate(mols_reps_folds, axis=1)

        print(mol_reps.shape)

        print(len(mol_ids))
        print(mol_ids[:10])
        print(len(mol_subsets))
        print(mol_subsets[:10])

        # save mol_reps and mol_ids

        import pickle

        with open("/drug/DrugCLIP_chemdata_v2024/embs/6folds.pkl", "wb") as f:
            pickle.dump([mol_ids, mol_subsets, mol_reps], f)
        

            

            
        
        return


    
    def huice_1031(self, model, **kwargs):
        import pickle
        root_dir = "/drug/Validation_data_V2"
        #root_dir = "/drug/drugclip_tmp/yaoming1201/"
        root_dir = "/drug/drugclip_tmp/Validation_data_V2_new"

        # find all dirs in root dir

        prot_list = os.listdir(root_dir)

        # check is dir

        prot_list = [x for x in prot_list if os.path.isdir(os.path.join(root_dir, x))]
        print(len(prot_list))
        #prot_list = ["5HT2A"]

        # prot_list = [
        #     "5HT2A",
        #     "ADRB2",
        #     "ALDH1",
        #     "BTK",
        #     "CB2",
        #     "D2R",
        #     "ESR2",
        #     "FEN1",
        #     "GABAA_A1B2C2",
        #     "GBA",
        #     "GluN1_2A",
        #     "GluN1_2B",
        #     "IDH1",
        #     "KAT2A",
        #     "MAPK1",
        #     "MAPK10",
        #     "MTORC1",
        #     "OPRK1",
        #     "P2X3",
        #     "SCN9A",
        #     "TRPM8",
        #     "TRPV1",
        #     "VDR",
        #     "XIAP",
        #     "XOR"
        # ]

        # prot_list = ["3CL"]
        # encode chemdiv 
        prot_list = ["BTK","MAPK1","MAPK10"]


        chemdiv_data_path = "/drug/encoding_mols/chemdiv_1640k.lmdb"

        chemdiv_dataset = self.load_mols_dataset(chemdiv_data_path, "atoms", "coordinates")
        chemdiv_data = torch.utils.data.DataLoader(chemdiv_dataset, batch_size=128, collate_fn=chemdiv_dataset.collater)
        chemdiv_names = []
        chemdiv_reps = []
        #cache_path = "chemdiv_reps_nopre.pkl"
        #cache_path = "/drug/drugclip_tmp/chemdiv_reps_mmseq_30.pkl"
        cache_path = "/drug/drugclip_tmp/chemdiv_reps_hmmer_30.pkl"
        cache_path = "/drug/drugclip_tmp/chemdiv_reps_pre_hmmer_30.pkl"
        cache_path = "/drug/drugclip_tmp/chemdiv_reps_pre_hmmer_30_21ckpt.pkl"
        cache_path = "/drug/drugclip_tmp/chemdiv_reps_pre_hmmer_30_20ckpt.pkl"
        cache_path = "/drug/drugclip_tmp/chemdiv_reps_pre_hmmer_1e4.pkl"
        cache_path = "/drug/drugclip_tmp/chemdiv_reps_nopre_hmmer_1e4.pkl"
        cache_path = "/drug/drugclip_tmp/chemdiv_reps_pre_hmmer_1e4_24ckpt.pkl"
        cache_path = "/drug/drugclip_tmp/chemdiv_reps_pre_mmseq_90_1e3.pkl"
        cache_path = "/drug/drugclip_tmp/chemdiv_reps_pre_hmmer_1e4_3000w.pkl"
        cache_path = "/drug/drugclip_tmp/reps_lib/chemdiv_reps_mmseq_30.pkl"
        #cache_path = "/drug/drugclip_tmp/chemdiv_reps_pre_hmmer_1e4_real.pkl"
        #cache_path = "chemdiv_reps_pre.pkl"
        #cache_path = "/drug/drugclip_tmp/reps_lib/chemdiv_reps_pre_hmmer_1e4.pkl"

        #cache_path = "/drug/drugclip_tmp/reps_lib/yaoming1201_mmseq30.pkl"
        #cache_path = "/drug/drugclip_tmp/reps_lib/yaoming1201_mmseq90.pkl"
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                chemdiv_reps, chemdiv_names = pickle.load(f)
        else:
            for _, sample in enumerate(tqdm(chemdiv_data)):
                sample = unicore.utils.move_to_cuda(sample)
                dist = sample["net_input"]["mol_src_distance"]
                et = sample["net_input"]["mol_src_edge_type"]
                st = sample["net_input"]["mol_src_tokens"]
                mol_padding_mask = st.eq(model.mol_model.padding_idx)
                mol_x = model.mol_model.embed_tokens(st)
                n_node = dist.size(-1)
                gbf_feature = model.mol_model.gbf(dist, et)
                gbf_result = model.mol_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                mol_outputs = model.mol_model.encoder(
                    mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
                )
                mol_encoder_rep = mol_outputs[0][:,0,:]
                mol_emb = model.mol_project(mol_encoder_rep)
                mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
                mol_emb = mol_emb.detach().cpu().numpy()
                chemdiv_reps.append(mol_emb)
                chemdiv_names.extend(sample["smi_name"])

            chemdiv_reps = np.concatenate(chemdiv_reps, axis=0)
        # save chemdiv_reps and chemdiv_names
        
            with open(cache_path, "wb") as f:
                pickle.dump([chemdiv_reps, chemdiv_names], f)

        ef_list = []
        top50_list = []
        top100_list = []
        auc_list = []
        bedroc_list = []
        prot_finish_list = []
        for prot in prot_list:
            try:
                active_mol_path = os.path.join(root_dir, prot, "ChEMBL", "active.lmdb")
                inactive_mol_path = os.path.join(root_dir, prot, "ChEMBL", "inactive.lmdb")
                pocket_path = os.path.join(root_dir, prot, "PDB", "apo_pocket.lmdb")
                #pocket_path = "/home/gaobowen/he_pockets/hADRB3.lmdb"
                #pocket_path = "/home/gaobowen/he_pockets/hadrb3_fixed.lmdb"
                active_names = []
                active_embs = []
                # encode active mols 
                active_dataset = self.load_mols_dataset(active_mol_path, "atoms", "coordinates")
                active_data = torch.utils.data.DataLoader(active_dataset, batch_size=32, collate_fn=active_dataset.collater)
                labels = [0] * len(chemdiv_names)
                for _, sample in enumerate(tqdm(active_data)):
                    sample = unicore.utils.move_to_cuda(sample)
                    dist = sample["net_input"]["mol_src_distance"]
                    et = sample["net_input"]["mol_src_edge_type"]
                    st = sample["net_input"]["mol_src_tokens"]
                    mol_padding_mask = st.eq(model.mol_model.padding_idx)
                    mol_x = model.mol_model.embed_tokens(st)
                    n_node = dist.size(-1)
                    gbf_feature = model.mol_model.gbf(dist, et)
                    gbf_result = model.mol_model.gbf_proj(gbf_feature)
                    graph_attn_bias = gbf_result
                    graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                    graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                    mol_outputs = model.mol_model.encoder(
                        mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
                    )
                    mol_encoder_rep = mol_outputs[0][:,0,:]
                    mol_emb = model.mol_project(mol_encoder_rep)
                    mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
                    mol_emb = mol_emb.detach().cpu().numpy()
                    active_embs.append(mol_emb)
                    active_names.extend(sample["smi_name"])
                labels.extend([1]*len(active_names))
                active_reps = np.concatenate(active_embs, axis=0)
                # encode inactive mols
                inactive_names = []
                inactive_embs = []
                inactive_dataset = self.load_mols_dataset(inactive_mol_path, "atoms", "coordinates")
                inactive_data = torch.utils.data.DataLoader(inactive_dataset, batch_size=32, collate_fn=inactive_dataset.collater)
                for _, sample in enumerate(tqdm(inactive_data)):
                    sample = unicore.utils.move_to_cuda(sample)
                    dist = sample["net_input"]["mol_src_distance"]
                    et = sample["net_input"]["mol_src_edge_type"]
                    st = sample["net_input"]["mol_src_tokens"]
                    mol_padding_mask = st.eq(model.mol_model.padding_idx)
                    mol_x = model.mol_model.embed_tokens(st)
                    n_node = dist.size(-1)
                    gbf_feature = model.mol_model.gbf(dist, et)
                    gbf_result = model.mol_model.gbf_proj(gbf_feature)
                    graph_attn_bias = gbf_result
                    graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                    graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                    mol_outputs = model.mol_model.encoder(
                        mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
                    )
                    mol_encoder_rep = mol_outputs[0][:,0,:]
                    mol_emb = model.mol_project(mol_encoder_rep)
                    mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
                    mol_emb = mol_emb.detach().cpu().numpy()
                    inactive_embs.append(mol_emb)
                    inactive_names.extend(sample["smi_name"])
                inactive_reps = np.concatenate(inactive_embs, axis=0)
                all_mols_reps = np.concatenate([chemdiv_reps, active_reps, inactive_reps], axis=0)
                labels.extend([0]*len(inactive_names))
                all_mols_names = chemdiv_names + active_names + inactive_names
                # encode pocket
                pocket_dataset = self.load_pockets_dataset(pocket_path)
                pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=16, collate_fn=pocket_dataset.collater)
                pocket_reps = []
                pocket_names = []
                for _, sample in enumerate(tqdm(pocket_data)):
                    sample = unicore.utils.move_to_cuda(sample)
                    dist = sample["net_input"]["pocket_src_distance"]
                    et = sample["net_input"]["pocket_src_edge_type"]
                    st = sample["net_input"]["pocket_src_tokens"]
                    pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
                    pocket_x = model.pocket_model.embed_tokens(st)
                    n_node = dist.size(-1)
                    gbf_feature = model.pocket_model.gbf(dist, et)
                    gbf_result = model.pocket_model.gbf_proj(gbf_feature)
                    graph_attn_bias = gbf_result
                    graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                    graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                    pocket_outputs = model.pocket_model.encoder(
                        pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
                    )
                    pocket_encoder_rep = pocket_outputs[0][:,0,:]
                    pocket_emb = model.pocket_project(pocket_encoder_rep)
                    pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
                    pocket_emb = pocket_emb.detach().cpu().numpy()
                    pocket_reps.append(pocket_emb)
                    pocket_name = sample["pocket_name"]
                    pocket_names.extend(pocket_name)
                pocket_reps = np.concatenate(pocket_reps, axis=0)
                # cast to float32
                all_mols_reps = all_mols_reps.astype(np.float32)
                pocket_reps = pocket_reps.astype(np.float32)
                res = pocket_reps @ all_mols_reps.T
                # get z score for each row
                #res = (res - np.mean(res, axis=1, keepdims=True)) / (np.std(res, axis=1, keepdims=True)+1e-6)

                # robust z score method
                # get median for each row
                medians = np.median(res, axis=1, keepdims=True)
                # get mad for each row
                mads = np.median(np.abs(res - medians), axis=1, keepdims=True)
                # get z score
                res = 0.6745 * (res - medians) / (mads + 1e-6)
                # get max for each column
                res_max = np.max(res, axis=0)
                # get top 1% based on score from high to low

                top1_idx = np.argsort(res_max)[::-1][:int(len(res_max) * 0.01)]
                # get top 1% names
                top1_names = [all_mols_names[i] for i in top1_idx]
                # check how many active mols in top 1%
                n_active = len(active_names)
                n_hit = 0
                for name in top1_names:
                    if name in active_names:
                        n_hit += 1
                ef1 = n_hit / n_active
                # get top 50 acc
                ef_list.append(ef1*100)
                top50_idx = np.argsort(res_max)[::-1][:50]
                top50_names = [all_mols_names[i] for i in top50_idx]
                n_hit = 0
                for name in top50_names:
                    if name in active_names:
                        n_hit += 1
                
                top50_acc = n_hit / 50
                top50_list.append(top50_acc)

                # get top 100 acc
                top100_idx = np.argsort(res_max)[::-1][:100]
                top100_names = [all_mols_names[i] for i in top100_idx]
                n_hit = 0
                for name in top100_names:
                    if name in active_names:
                        n_hit += 1
                top100_acc = n_hit / 100
                top100_list.append(top100_acc)

                auc = roc_auc_score(labels, res_max)
                bedroc = cal_metrics(labels, res_max, 80.5)[1]
                auc_list.append(auc)
                bedroc_list.append(bedroc)
                #print(f"{prot} n_active: {n_active}, n_hit: {n_hit}, ef1%: {n_hit / n_active * 100}%")
                #pocket_name = pocket_names[i]
                #print(f"{prot} {pocket_name} ef1%: {ef1 * 100}%, top50_acc: {top50_acc * 100}%, top100_acc: {top100_acc * 100}%")
                #print(f"{prot} ef1%: {ef1 * 100}%, top50_acc: {top50_acc * 100}%, top100_acc: {top100_acc * 100}%")
                # print ef,auc,bedroc
                print(f"{prot} ef1%: {ef1 * 100}%, auc: {auc}, bedroc: {bedroc}")
                res_chemdiv = res_max[:len(chemdiv_names)]
                sort_index = np.argsort(res_chemdiv)[::-1]

                # hit_list = ["HIT106974677","HIT103076211","HIT102650505","HIT105976162","HIT211471431","HIT101492901","HIT100387222","HIT100823162"]
                # for hit in hit_list:
                #     # find index in sort index
                #     hit_index = np.where(np.array(chemdiv_names)[sort_index] == hit)[0][0]
                #     print(f"{hit} {hit_index}")
                    
                # calculate bedroc
                # labels = [1 if x in active_names else 0 for x in all_mols_names]
                # auc, bedroc, ef_list, re_list = cal_metrics(labels, res, 80.5)
                # print("aucs: ", auc, bedroc)
                # get top 1% of chemdiv
                
                res_chemdiv = res_max[:len(chemdiv_names)]
                top1_idx = np.argsort(res_chemdiv)[::-1][:int(len(res_chemdiv) * 0.01)]
                top1_names = [chemdiv_names[i] for i in top1_idx]
                top1_scores = [res_chemdiv[i] for i in top1_idx]
                # save
                # res_path = os.path.join(root_dir, prot, "res_chemdiv_5ht2a.txt")
                # with open(res_path, "w") as f:
                #     for name, score in zip(top1_names, top1_scores):
                #         f.write(f"{name}\t{score}\n")

                # res_path = "/drug/drugclip_tmp/res_chemdiv_hadrb2.txt"
                # with open(res_path, "w") as f:
                #     for name, score in zip(chemdiv_names, res_chemdiv):
                #         f.write(f"{name}\t{score}\n")
                

                #save res_chemdiv[0], res_chemdiv[1], res_chemdiv[2] and names, scores
                # res_path = "/drug/drugclip_tmp/res_chemdiv_hadrb3_1106.txt"
                # res_chemdiv = res[:, :len(chemdiv_names)]
                # with open(res_path, "w") as f:
                #     for i in range(len(res_chemdiv[0])):
                #         score1 = res_chemdiv[0][i]
                #         score2 = res_chemdiv[1][i]
                #         #score3 = res_chemdiv[2][i]
                #         name = chemdiv_names[i]
                #         #f.write(f"{name}\t{score1}\t{score2}\t{score3}\n")
                #         f.write(f"{name}\t{score1}\t{score2}\n")
                



                # print(active_names[:10])
                # print(chemdiv_names[:10])
                # print(inactive_names[:10])
                prot_finish_list.append(prot)
            except:
                continue
        print(",,,,,,,,,,,,all")
        print(np.mean(ef_list))
        print(np.percentile(ef_list, 25))
        print(np.percentile(ef_list, 50))
        print(np.percentile(ef_list, 75))        
        print(",,,,,,,,,,,,50")
        print(np.mean(top50_list))
        print(np.percentile(top50_list, 25))
        print(np.percentile(top50_list, 50))
        print(np.percentile(top50_list, 75))
        print(",,,,,,,,,,,,100")
        print(np.mean(top100_list))
        print(np.percentile(top100_list, 25))
        print(np.percentile(top100_list, 50))
        print(np.percentile(top100_list, 75))
        for prot, auc, bedroc, ef1 in zip(prot_finish_list, auc_list, bedroc_list, ef_list):
            # also print ef
            print(f"{prot} auc: {auc}, bedroc: {bedroc}, ef1%: {ef1}%")
        return
    
    
    def encode_mols_once(self, model, data_path,atoms,coords, **kwargs):

        #names = "PPARG"
        import pickle

        #data_path = "/home/gaobowen/he_pockets/chemdiv.lmdb"
        ckpt_data =  self.args.finetune_from_model.split("/")[-2]
        name = data_path.split("/")[-1].split(".")[0]
        cache_path = "/drug/clpocket/mol_libs/" + ckpt_data +"_"+ name + ".pkl"
        #cache_path =  "/drug/clpocket/mol_libs/" + name + ".pkl"
        print(cache_path)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                mol_reps, mol_names = pickle.load(f)
            return mol_reps, mol_names

        
        mol_dataset = self.load_mols_dataset(data_path,atoms,coords)
        mol_reps = []
        mol_names = []
        bsz=32
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)
        for _, sample in enumerate(tqdm(mol_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["mol_src_distance"]
            et = sample["net_input"]["mol_src_edge_type"]
            st = sample["net_input"]["mol_src_tokens"]
            mol_padding_mask = st.eq(model.mol_model.padding_idx)
            mol_x = model.mol_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.mol_model.gbf(dist, et)
            gbf_result = model.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            mol_outputs = model.mol_model.encoder(
                mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
            )
            mol_encoder_rep = mol_outputs[0][:,0,:]
            mol_emb = model.mol_project(mol_encoder_rep)
            mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
            mol_emb = mol_emb.detach().cpu().numpy()
            mol_reps.append(mol_emb)
            mol_names.extend(sample["smi_name"])
            #labels.extend(sample["target"].detach().cpu().numpy())
            #keys.extend(sample["key"])
        mol_reps = np.concatenate(mol_reps, axis=0)
        print(mol_reps.shape)
        #labels = np.array(labels, dtype=np.int32)
        # generate pocket data
        with open(cache_path, "wb") as f:
            pickle.dump([mol_reps, mol_names], f)
        return mol_reps, mol_names
    
    def find_mols(self, model, **kwargs):
 
        #names = "PPARG"
        import pickle

        chemdiv_path = "/drug/encoding_mols/chemdiv_1640k.lmdb"
        chemdiv_path = "/home/gaobowen/he_pockets/chemdiv.lmdb"
        enam_paths = ["/drug/clpocket/mol_libs/enam/enam_full_1.lmdb", "/drug/clpocket/mol_libs/enam/enam_full_2.lmdb"]
        zinc_path = "/drug/clpocket/mol_libs/zinc_readydock.lmdb"
        all_sxj_lib_path = "/home/gaobowen/he_pockets/all_sxj_lib_0914.lmdb"
        mol_reps_chemdiv, mol_names_chemdiv = self.encode_mols_once(model, chemdiv_path,"atoms", "coordinates")
        mol_reps_enam_list = []
        mol_names_enam_list = []
        for path in enam_paths:
            mol_reps_enam, mol_names_enam = self.encode_mols_once(model, path, "atom_types", "coords")
            mol_reps_enam_list.append(mol_reps_enam)
            mol_names_enam_list.append(mol_names_enam)
        mol_reps_zinc, mol_names_zinc = self.encode_mols_once(model, zinc_path, "atom_types", "coords")

        all_reps = np.concatenate([mol_reps_chemdiv, mol_reps_zinc] + mol_reps_enam_list, axis=0)
        mol_names_enam = mol_names_enam_list[0] + mol_names_enam_list[1]
        all_names = mol_names_chemdiv + mol_names_zinc + mol_names_enam
        # change to float32
        all_reps = all_reps.astype(np.float32) 

        mol_reps_enam = np.concatenate(mol_reps_enam_list, axis=0)

        mol_reps_sxj, mol_names_sxj = self.encode_mols_once(model, all_sxj_lib_path, "atoms", "coordinates")

        data_path = "/home/gaobowen/he_pockets/WZW_yeast.lmdb"
        pocket_dataset = self.load_pockets_dataset(data_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=16, collate_fn=pocket_dataset.collater)
        pocket_reps = []
        pocket_names = []
        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
            pocket_x = model.pocket_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.pocket_model.gbf(dist, et)
            gbf_result = model.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            pocket_outputs = model.pocket_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
            )
            pocket_encoder_rep = pocket_outputs[0][:,0,:]
            pocket_emb = model.pocket_project(pocket_encoder_rep)
            #pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
            pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_reps.append(pocket_emb)
            pocket_names.extend(sample["pocket_name"])
        pocket_reps = np.concatenate(pocket_reps, axis=0)
        print(pocket_reps.shape)
        

        data_path = "/home/gaobowen/he_pockets/WZW_human.lmdb"
        pocket_dataset = self.load_pockets_dataset(data_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=16, collate_fn=pocket_dataset.collater)
        pocket_reps_2 = []
        pocket_names = []
        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
            pocket_x = model.pocket_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.pocket_model.gbf(dist, et)
            gbf_result = model.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            pocket_outputs = model.pocket_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
            )
            pocket_encoder_rep = pocket_outputs[0][:,0,:]
            pocket_emb = model.pocket_project(pocket_encoder_rep)
            #pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
            pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_reps_2.append(pocket_emb)
            pocket_names.extend(sample["pocket_name"])
        pocket_reps_2 = np.concatenate(pocket_reps_2, axis=0)


        res_sxj = pocket_reps @ mol_reps_sxj.T
        res_sxj = np.max(res_sxj, axis=0)


        res_zinc =  pocket_reps @ mol_reps_zinc.T
        res_zinc = np.max(res_zinc, axis=0)

        res_zinc_human =  pocket_reps_2 @ mol_reps_zinc.T
        res_zinc_human = np.max(res_zinc_human, axis=0)

        res_enam = pocket_reps @ mol_reps_enam.T
        res_enam = np.max(res_enam, axis=0)

        res_enam_human = pocket_reps_2 @ mol_reps_enam.T
        res_enam_human = np.max(res_enam_human, axis=0)

        res_chemdiv = pocket_reps @ mol_reps_chemdiv.T
        res_chemdiv = np.max(res_chemdiv, axis=0)

        res_chemdiv_human = pocket_reps_2 @ mol_reps_chemdiv.T
        res_chemdiv_human = np.max(res_chemdiv_human, axis=0)

        lis = []
        lis_human = []

        # get top 10000 value of lis
        indices = np.argsort(res_enam)[-10000:]
        #lis.sort(key=lambda x:x[0], reverse=True)
        # 
        count = 0
        res_list = []
        for i in range(10000):
            index = indices[i]
            score = res_enam[index]
            score_human = res_enam_human[index]
            #key = lis[i][2]
            smi = mol_names_enam[index]
            # if key not in gt:
            #     res_list.append((smi, score))
            # else:
            #     count+=1
            delta_score = score - score_human
            res_list.append((smi, delta_score))
        
        res_list.sort(key=lambda x:x[1], reverse=True)
        res_list = res_list[:1000]
 
        print(count)
        # save res_list to txt
        with open("WzW_enam.txt", "w") as f:
            for smi, score in res_list:
                f.write(smi+"\t"+str(score)+"\n")

        
        return 


    def retrieve_mols(self, model, **kwargs):
 
        #names = "PPARG"
        import pickle

        chemdiv_path = "/drug/encoding_mols/chemdiv_1640k.lmdb"
        chemdiv_path = "/drug/encoding_mols/chemdiv_1634476_filter.lmdb"
        crossdocked_path = "/drug/retrieval_gen/train_processed.lmdb"
        mol_reps_chemdiv, mol_names_chemdiv = self.encode_mols_once(model, chemdiv_path,"atoms", "coordinates")
        mol_reps_crossdocked, mol_names_crossdocked = self.encode_mols_once(model, crossdocked_path,"ligand_atoms", "ligand_coordinates")
        
        mol_reps_chemdiv = mol_reps_chemdiv.astype(np.float32)
        mol_reps_crossdocked = mol_reps_crossdocked.astype(np.float32)
        

        

        pocket_path = "/drug/retrieval_gen/train_processed.lmdb"
        pocket_path = "/drug/retrieval_gen/crossdock_train/valid.lmdb"
        pocket_path = "/drug/retrieval_gen/all_processed.lmdb"
        pocket_dataset = self.load_pockets_dataset(pocket_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=8, collate_fn=pocket_dataset.collater)
        pocket_reps = []
        pocket_names = []
        res_all = []
        top1000_index = []
        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
            pocket_x = model.pocket_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.pocket_model.gbf(dist, et)
            gbf_result = model.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            pocket_outputs = model.pocket_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
            )
            pocket_encoder_rep = pocket_outputs[0][:,0,:]
            pocket_emb = model.pocket_project(pocket_encoder_rep)
            pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
            pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_emb = pocket_emb.astype(np.float32)
            res_tmp = pocket_emb @ mol_reps_crossdocked.T
            pocket_reps.append(pocket_emb)
            pocket_names.extend(sample["pocket_name"])
            #print(sample["pocket_name"])
            #res_all.append(res_tmp)
            #top1000_index.append(np.argsort(-res_tmp, axis=1)[:,:1000])
            res_tmp = torch.from_numpy(res_tmp)#.cuda()
            values, idx = torch.topk(res_tmp, k=10000, axis=-1)
            top1000_index.append(idx.detach().cpu().numpy())

        pocket_reps = np.concatenate(pocket_reps, axis=0)
        print(pocket_reps.shape)
        #res = np.concatenate(res_all, axis=0)
        top1000_index = np.concatenate(top1000_index, axis=0)
        print(top1000_index.shape)

        #top1000_index = np.argsort(res, axis=1)[:,::-1][:,:1000]



        # save to one txt file

        with open("/drug/retrieval_gen/all_retrieval.txt", "w") as f:
            for i in range(len(top1000_index)):
                f.write(pocket_names[i]+"\t")
                for j in range(len(top1000_index[i])):
                    index = top1000_index[i][j]
                    f.write(mol_names_crossdocked[index]+"\t")
                f.write("\n")


        
        



        

        
        return 

    def test_kinase(self, model, **kwargs):

        #names = "PPARG"
        import pickle
        data_path = "kinase.lmdb"
        mol_dataset = self.load_mols_dataset(data_path)
        num_data = len(mol_dataset)
        bsz=32
        print(num_data//bsz)
        
        labels = []
        
        # generate mol data

        mol_reps = []
        mol_names = []
        mol_keys = []
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)
        for _, sample in enumerate(tqdm(mol_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["mol_src_distance"]
            et = sample["net_input"]["mol_src_edge_type"]
            st = sample["net_input"]["mol_src_tokens"]
            mol_padding_mask = st.eq(model.mol_model.padding_idx)
            mol_x = model.mol_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.mol_model.gbf(dist, et)
            gbf_result = model.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            mol_outputs = model.mol_model.encoder(
                mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
            )
            mol_encoder_rep = mol_outputs[0][:,0,:]
            mol_emb = model.mol_project(mol_encoder_rep)
            mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
            mol_emb = mol_emb.detach().cpu().numpy()
            mol_reps.append(mol_emb)
            mol_names.extend(sample["smi_name"])
            mol_keys.extend(sample["key"])
            #labels.extend(sample["target"].detach().cpu().numpy())
            #keys.extend(sample["key"])
        mol_reps = np.concatenate(mol_reps, axis=0)
        print(mol_reps.shape)

        data_path = "KLIFS_human.lmdb"
        pocket_dataset = self.load_pockets_dataset(data_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=16, collate_fn=pocket_dataset.collater)
        pocket_reps = []
        pocket_names = []
        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
            pocket_x = model.pocket_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.pocket_model.gbf(dist, et)
            gbf_result = model.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            pocket_outputs = model.pocket_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
            )
            pocket_encoder_rep = pocket_outputs[0][:,0,:]
            pocket_emb = model.pocket_project(pocket_encoder_rep)
            pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
            pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_reps.append(pocket_emb)
            pocket_names.extend(sample["pocket_name"])
        pocket_reps = np.concatenate(pocket_reps, axis=0)
        print(pocket_reps.shape)
        # convert to float32
        mol_reps = mol_reps.astype(np.float32)
        pocket_reps = pocket_reps.astype(np.float32)
        res = mol_reps @ pocket_reps.T
        
        index_to_name = {}
        name_to_index = {}
        count = 0
        for i in range(len(pocket_names)):
            if pocket_names[i] not in name_to_index:
                name_to_index[pocket_names[i]] = count
                index_to_name[count] = pocket_names[i]
                count += 1


        #for i in tqdm(range(res.shape[0])):

        num_data = len(name_to_index)

        res_dic = {}
        print("here")
        for i in tqdm(range(res.shape[0])):
            smi = mol_names[i]
            if smi not in res_dic:
                res_dic[smi] = [-10] * num_data
            for j in range(res.shape[1]):
                name = pocket_names[j]
                index = name_to_index[name]
                res_dic[smi][index] = max(res_dic[smi][index], res[i][j])

        def topk(res_dic, k=10):
            count = 0
            for i in tqdm(range(res.shape[0])):
                smi = mol_names[i]
                true_name = mol_keys[i]
                if true_name not in name_to_index:
                    continue
                true_index = name_to_index[true_name]
                #res_dic[smi].sort(key=lambda x:x[1], reverse=True)

                # get index of topk 
                topk_index = np.argsort(res_dic[smi])[::-1][:k]
                for index in topk_index:
                    if index == true_index:
                        count += 1
                        break
            return count / res.shape[0]
                
        for k in [10, 20, 30, 40, 50]:
            print("topk", k, topk(res_dic, k=k))

        
        
        return 

    def test_kinase_new(self, model, **kwargs):

        #names = "PPARG"
        import pickle
        data_path = "/drug/kinase_drugdata/mols.lmdb"
        mol_dataset = self.load_mols_dataset(data_path)
        num_data = len(mol_dataset)
        bsz=32
        print(num_data//bsz)
        
        labels = []
        
        # generate mol data

        mol_reps = []
        mol_names = []
        mol_keys = []
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)
        for _, sample in enumerate(tqdm(mol_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["mol_src_distance"]
            et = sample["net_input"]["mol_src_edge_type"]
            st = sample["net_input"]["mol_src_tokens"]
            mol_padding_mask = st.eq(model.mol_model.padding_idx)
            mol_x = model.mol_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.mol_model.gbf(dist, et)
            gbf_result = model.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            mol_outputs = model.mol_model.encoder(
                mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
            )
            mol_encoder_rep = mol_outputs[0][:,0,:]
            mol_emb = model.mol_project(mol_encoder_rep)
            #mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
            mol_emb = mol_emb.detach().cpu().numpy()
            mol_reps.append(mol_emb)
            mol_names.extend(sample["smi_name"])
            #mol_keys.extend(sample["key"])
            #labels.extend(sample["target"].detach().cpu().numpy())
            #keys.extend(sample["key"])
        mol_reps = np.concatenate(mol_reps, axis=0)
        print(mol_reps.shape)

        data_path = "/drug/kinase_drugdata/kinase_pocket.lmdb"
        pocket_dataset = self.load_pockets_dataset(data_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=16, collate_fn=pocket_dataset.collater)
        pocket_reps = []
        pocket_names = []
        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
            pocket_x = model.pocket_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.pocket_model.gbf(dist, et)
            gbf_result = model.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            pocket_outputs = model.pocket_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
            )
            pocket_encoder_rep = pocket_outputs[0][:,0,:]
            pocket_emb = model.pocket_project(pocket_encoder_rep)
            #pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
            pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_reps.append(pocket_emb)
            pocket_names.extend(sample["pocket_name"])
        pocket_reps = np.concatenate(pocket_reps, axis=0)
        print(pocket_reps.shape)
        # convert to float32
        mol_reps = mol_reps.astype(np.float32)
        pocket_reps = pocket_reps.astype(np.float32)
        res = mol_reps @ pocket_reps.T

        
        with open("/drug/kinase_drugdata/smiles_to_kinase_final.pkl", "rb") as f:
            label_dic = pickle.load(f)
        print(pocket_names)


        new_dic = {}
        for key in label_dic:
            pockets = label_dic[key]
            for pocket in pockets:
                if pocket not in new_dic:
                    new_dic[pocket] = []
                new_dic[pocket].append(key)
        new_res = res.T

        
        #k=2
        def topk_mol(new_res, new_dic, k):
            total = 0
            count = 0
            for i in range(len(new_res)):
                pocket = pocket_names[i]
                #print(smi in label_dic)
                if pocket not in new_dic:
                    continue
                label = new_dic[pocket]
                #print(len(label))
                total+=len(label)
                #print(label)
                
                #print(label)
                # sort
                index = np.argsort(new_res[i])[::-1][:k]
                #index = np.random.choice(len(mol_names), k, replace=False)
                for j in index:
                    smi = mol_names[j]
                    #print(pocket)
                    if smi in label:
                        count += 1
                        break
            print(total)
            return count/len(new_res)
        #for i in tqdm(range(res.shape[0])):
        #res = np.random.rand(res.shape[0], res.shape[1])
       
        # def topk(res, label_dic, k):
        #     count = 0
        #     for i in range(len(res)):
        #         smi = mol_names[i]
        #         #print(smi in label_dic)
        #         if smi not in label_dic:
        #             continue
        #         label = label_dic[smi]
        #         label = np.random.choice(label, 1, replace=False)
        #         #print(label)
        #         #print(label)
                
        #         #print(label)
        #         # sort
        #         index = np.argsort(res[i])[::-1][:k]
        #         #index = np.argsort(res[i])[:k]
        #         #print(np.sort(res[i])[::-1][:k])
        #         index = np.random.choice(len(pocket_names), k, replace=False)
        #         #print(index)
        #         for j in index:
        #             pocket = pocket_names[j]
        #             #print(pocket)
        #             if pocket in label:
        #                 count += 1
        #                 break
        #     return count / len(res)


        
        # for k in [5,10,15]:
        #     print("topk", k, topk(res, label_dic, k))
        
        for k in [5,10,15,20]:
            print("topk", k, topk_mol(new_res, new_dic, k))

        
        
        return 

    def encode_pockets(self, model, **kwargs):
        """Encode a pocket."""

        #data_path = self.args.pocket_path
        data_path = "/drug/animo/human_pockets.lmdb"
        pocket_dataset = self.load_pockets_dataset(data_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=16, collate_fn=pocket_dataset.collater)
        pocket_reps = []
        pocket_names = []
        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
            pocket_x = model.pocket_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.pocket_model.gbf(dist, et)
            gbf_result = model.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            pocket_outputs = model.pocket_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
            )
            pocket_encoder_rep = pocket_outputs[0][:,0,:]
            pocket_emb = model.pocket_project(pocket_encoder_rep)
            #pocket_emb = pocket_emb / pocket_emb.norm(dim=1, keepdim=True)
            pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_names.extend(sample["pocket_name"])
            # for index, pocket_name in enumerate(pocket_names):
            #     if pocket_name == pocket_select:
            #         pocket_reps.append(pocket_emb[index])
            pocket_reps.append(pocket_emb)
        pocket_reps = np.concatenate(pocket_reps, axis=0)
        
        # save to pickle
        import pickle
        #pickle.dump(pocket_reps, open("/drug/demo/pocket_reps/" + pocket_name + ".pkl", "wb"))
        pickle.dump([pocket_reps,pocket_names], open("/drug/animo/human_pockets.pkl", "wb"))


        return 

    def encode_mols(self, model, **kwargs):

        #names = "PPARG"
        import pickle
        data_path = "/data/protein/zinc_all/zinc_all.lmdb"
        data_path = "/data/protein/zinc_11M/zinc_11M_100k.lmdb"
        data_path = "/data/protein/local_data/drug_clip_pdb_general/train_no_test/train.lmdb"
        data_path = "/data/protein/pdbbind_2020/pdb_2020_ligand.lmdb"
        data_path = "/drug/drugclip_tmp/docking_benchmark/mols.lmdb"
        #data_path = "/drug/prot_frag/train_ligand_pocket/100k.lmdb"
        #data_path = "/drug/chemdiv_1640k.lmdb"
        #data_path = "/drug/encoding_mols/ADME.lmdb"
        #data_path = "/drug/encoding_mols/Drug_smiles.lmdb"
        data_path = "/drug/retrieval_gen/chemdiv_top100_valid.lmdb"
        data_path = "/drug/retrieval_gen/ret_crossdocked_top100.lmdb"
        data_path = "/drug/prot_frag/train_ligand_pocket/train_1m_frad_ptvdn.lmdb"
        mol_dataset = self.load_mols_dataset(data_path, "lig_atoms_real", "lig_coord_real")
        num_data = len(mol_dataset)
        bsz=64
        print(num_data//bsz)
        
        
        # generate mol data

        
        mol_reps = []
        mol_names = []
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)
        for _, sample in enumerate(tqdm(mol_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["mol_src_distance"]
            et = sample["net_input"]["mol_src_edge_type"]
            st = sample["net_input"]["mol_src_tokens"]
            mol_padding_mask = st.eq(model.mol_model.padding_idx)
            mol_x = model.mol_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.mol_model.gbf(dist, et)
            gbf_result = model.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            mol_outputs = model.mol_model.encoder(
                mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
            )
            mol_encoder_rep = mol_outputs[0][:,0,:]
            #mol_emb = model.mol_project(mol_encoder_rep)
            mol_emb = mol_encoder_rep
            #mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
            mol_emb = mol_emb.detach().cpu().numpy()
            mol_reps.append(mol_emb)
            mol_names.extend(sample["smi_name"])
            #print(sample["smi_name"])
            #labels.extend(sample["target"].detach().cpu().numpy())
            #keys.extend(sample["key"])
        mol_reps = np.concatenate(mol_reps, axis=0)
        mol_cache_path = "/drug/data/zinc_embs/mol_reps.pkl"
        mol_cache_path = "/drug/demo_reps.pkl"
        mol_cache_path = "/drug/encoding_mols/ADME.pkl"
        mol_cache_path = "/drug/encoding_mols/Drug_smiles.pkl"
        mol_cache_path = "unimol_pdb_2020_real_encode.pkl"
        #mol_cache_path = "unimol_pep_real_encode.pkl"
        mol_cache_path = "/drug/drugclip_tmp/docking_benchmark/mols_unimol_random.pkl"
        mol_cache_path = "/drug/retrieval_gen/encode_feates/chemdiv_top100_unimol_valid.pkl"
        mol_cache_path = "/drug/retrieval_gen/encode_feates/crossdock_top100_unimol.pkl"
        mol_cache_path = "/drug/prot_frag/train_ligand_pocket/train_1m_unimol.pkl"
        with open(mol_cache_path, "wb") as f:
            pickle.dump({"index":mol_names, "data":mol_reps}, f)

        
        return 

    def test_gpcr(self, filename, model, **kwargs):
        """Encode a dataset with the molecule encoder."""

        #names = "PPARG"
        pockets_path = filename
        mol_path = "/home/gaobowen/gpcr/ligands_merge.lmdb"
        mol_path = "/data/protein/gpcr/all_mols.lmdb"
        # read from txt
        
        keys = []

        labels = []
        bsz=64
        if self.mol_reps is None:
            mol_dataset = self.load_gpcr_mols_dataset(mol_path)
            num_data = len(mol_dataset)
            
            print(num_data//bsz)
            mol_reps = []
            mol_names = []
            
            
            # generate mol data
            
            mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)
            for num, sample in enumerate(tqdm(mol_data)):
                sample = unicore.utils.move_to_cuda(sample)
                dist = sample["net_input"]["mol_src_distance"]
                et = sample["net_input"]["mol_src_edge_type"]
                st = sample["net_input"]["mol_src_tokens"]
                mol_padding_mask = st.eq(model.mol_model.padding_idx)
                mol_x = model.mol_model.embed_tokens(st)
                n_node = dist.size(-1)
                gbf_feature = model.mol_model.gbf(dist, et)
                gbf_result = model.mol_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                mol_outputs = model.mol_model.encoder(
                    mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
                )
                mol_encoder_rep = mol_outputs[0][:,0,:]
                mol_emb = model.mol_project(mol_encoder_rep)
                mol_emb = mol_emb.detach().cpu().numpy()
                mol_reps.append(mol_emb)
                mol_names.extend(sample["smi_name"])
                #labels.extend(sample["target"].detach().cpu().numpy())
                #ids = sample["id"].detach().cpu().numpy()
                cur_keys = sample["key"]
                #print(cur_keys)
                # each elemet of ids in true_ligands
                for cur_key in cur_keys:
                    keys.append(cur_key)
                
            mol_reps = np.concatenate(mol_reps, axis=0)
            self.mol_reps = mol_reps
            self.keys = keys
            #
            #self.labels = labels

        mol_reps = self.mol_reps

        # generate pocket data

        pocket_dataset = self.load_pockets_dataset(pockets_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz, collate_fn=pocket_dataset.collater)
        pocket_reps = []
        pocket_names = []
        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            print(sample["pocket_name"])
            pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
            pocket_x = model.pocket_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.pocket_model.gbf(dist, et)
            gbf_result = model.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            pocket_outputs = model.pocket_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
            )
            pocket_encoder_rep = pocket_outputs[0][:,0,:]
            pocket_emb = model.pocket_project_s(pocket_encoder_rep)
            pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_reps.append(pocket_emb)
            pocket_names.extend(sample["pocket_name"])
        pocket_reps = np.concatenate(pocket_reps, axis=0)
        print(pocket_reps.shape)
        res = pocket_reps @ mol_reps.T
        

        


        return res, pocket_names
        

        
            
         

        
    
    
