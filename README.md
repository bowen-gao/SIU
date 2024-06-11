# SIU


## download the dataset

dataset is available at [huggingface dataset repo](https://huggingface.co/datasets/bgao95/SIU)

Please download and unzip all files

it should contains:

atom3d_data.zip : the dataset used to train gnn and cnn-3d model

split_60: the dataset used to train Uni-Mol or ProFSA model

pretrain_weights.zip:pretrained weights for Uni-Mol and ProFSA




## Environment




## Train the model with GNN/CNN-3D


```bash
cd ./atom3d/examples/lba

cd cnn3d or gnn

python train.py

```

note that the data path in train.py should be changed to atom3d_data/split_60 or atom3d_data/split_90


## Train the model with ProFSA

```bash
cd ./unimol_train_code

bash binding_affinity_unimol.sh or binding_affinity_profsa.sh

```

use the pretrained weights in pretrain_weights.zip


note that the data_path should in the bash file should be pointed to dir split_60 or split_90

If you want to to Multi Task Learning, please let --num-heads equals to 5, else set it to 1 and point to the correct directory in split_60 or split_90 (ic50, ec50, ki, kd)


