# SIU


This GitHub repository contains the dataset and relevant code for the paper: 

**SIU: A Million-Scale Structural Small Molecule-Protein Interaction Dataset for Unbiased Bioactivity Prediction**


## download the dataset

dataset is available at [huggingface dataset repo](https://huggingface.co/datasets/bgao95/SIU)

Please download and unzip all files

it should contains:

### gnn_cnn_data.zip 

The dataset used to train gnn and cnn-3d model. 

### split_60.zip

The dataset used to train Uni-Mol or ProFSA model for the SIU 0.6 version. It contains train.lmdb, valid.lmdb and test.lmdb.

### split_90.zip

The dataset used to train Uni-Mol or ProFSA model for the SIU 0.9 version. It contains train.lmdb, valid.lmdb and test.lmdb.

### pretrain_weights.zip

Pretrained weights for Uni-Mol and ProFSA.

profsa.pt: weights for pretrained ProFSA model

mol_pre_no_h_220816.pt: weights for pretrained Uni-Mol molecular Encoder

pocket_pre_220816.pt: weights for pretrained Uni-Mol pocketr Encoder

### final_dic.pkl

Complete dataset file in pickle format

### result_structures.zip

contains all structure files for protein and docked small molecules.



## dataset format

### final_dic.pkl

each key is an UniProt ID

corresponding value is a list of dictionaries. Each dictionary is a data point and has following keys:

| Key      | Description     |
|----------------|----------------|
| atoms  | atom types in ligand  |
| coordinates |  list of different conformations of the ligand |
|  pocket_atoms| atom types in pocket |
|  pocket_coordinates | atom positions of the pocket |
|  source_data |  UniProt ID and PDB ID information |
|  label |  dictionary for assay types and assay values |
|  ik | InChI key of the ligand |
|  smi |  SMILES notation of the ligand |



### Other lmdb files

All training and testing data are in lmdb format, and have the same keys as shown above.

Note that for single task learning, the label is a float value instead of a dictionary.


## Read Dataset

In read_data.py, we provide a script to read from lmdb files and pickle files.



## Environment

Follow the environment setting in [Uni-Mol](https://github.com/dptech-corp/Uni-Mol) and [Atom3D](https://github.com/drorlab/atom3d)

Or use **siu.yaml**

## Train the model with GNN/CNN-3D


```bash
cd ./atom3d/examples/lba
```

for cnn3d

```bash
cd cnn3d 
```

for gnn

```bash
cd gnn 
```

start training
```bash
python train.py

```

Note that the data path in train.py should be changed to atom3d_data/split_60 or atom3d_data/split_90


All the parameters are in train.py. We train the model with one NVIDIA A100 GPU.  


## Train the model with ProFSA

```bash
cd ./unimol_train_code

bash binding_affinity_unimol.sh or binding_affinity_profsa.sh

```

use the pretrained weights in pretrain_weights.zip


Note that the data_path should in the bash file should be pointed to dir split_60 or split_90 (0.6 version and 0.9 version respectively)

If you want to to Multi Task Learning, please let --num-heads equals to 5, else set it to 1 and point to the correct directory in split_60 or split_90 (ic50, ec50, ki, kd)


All the parameters are in bash scripts. We train the model with 4 NVIDIA A100 GPU.  
