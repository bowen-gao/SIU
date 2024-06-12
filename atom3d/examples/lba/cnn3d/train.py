import argparse
import datetime
import json
import os
import time
import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from atom3d.datasets import LMDBDataset
from scipy.stats import spearmanr, pearsonr 

from model import CNN3D_LBA
from data import CNN3D_TransformLBA


# Construct model
def conv_model(in_channels, spatial_size, args):
    num_conv = args.num_conv
    conv_filters = [32 * (2**n) for n in range(num_conv)]
    conv_kernel_size = 3
    max_pool_positions = [0, 1]*int((num_conv+1)/2)
    max_pool_sizes = [2]*num_conv
    max_pool_strides = [2]*num_conv
    fc_units = [512]

    model = CNN3D_LBA(
        in_channels, spatial_size,
        args.conv_drop_rate,
        args.fc_drop_rate,
        conv_filters, conv_kernel_size,
        max_pool_positions,
        max_pool_sizes, max_pool_strides,
        fc_units,
        batch_norm=args.batch_norm,
        dropout=not args.no_dropout)
    return model


def train_loop(model, loader, optimizer, device):
    model.train()

    loss_all = 0
    total = 0
    epoch_loss = 0
    progress_format = 'train loss: {:6.6f}'
    with tqdm.tqdm(total=len(loader), desc=progress_format.format(0)) as t:
        for i, data in enumerate(loader):
            feature = data['feature'].to(device).to(torch.float32)
            optimizer.zero_grad()
            output = model(feature)

            label = torch.cat([d.unsqueeze(1) for d in data["label"]], dim=1).to(device).to(torch.float32)


            mask = label >= 0

            squared_diff = (output - label) ** 2
            
            # Apply the mask
            masked_squared_diff = squared_diff * mask
            
            # Compute the mean of the masked squared differences

            #print(predicts[:3], targets[:3])
            loss = masked_squared_diff.sum() / mask.sum()
            loss.backward()
            loss_all += loss.item() * label.shape[0]
            total += label.shape[0]
            optimizer.step()
            #('loss:', loss.item())
            wandb.log({'train_loss': loss.item()})
            epoch_loss += (loss.item() - epoch_loss) / float(i + 1)
            t.set_description(progress_format.format(np.sqrt(epoch_loss)))
            t.update(1)
        return np.sqrt(loss_all / total)


@torch.no_grad()
def test(model, loader, device, split):
    model.eval()

    total = 0

    y_true = []
    y_pred = []
    uniprots = []
    pdbs = []
    mse_all = 0
    mae_all = 0

    for data in loader:
        feature = data['feature'].to(device).to(torch.float32)
        #label = data['label'].to(device).to(torch.float32)
        label = torch.cat(data["label"], dim=0).view(feature.shape[0], -1).to(device).to(torch.float32)
        output = model(feature)
        #print(label.shape)
        #loss = F.mse_loss(output, data.y)
        mask = label >= 0

        squared_diff = (output - label) ** 2
        
        # Apply the mask
        masked_squared_diff = squared_diff * mask
        
        
        # compute mae

        mae = (output - label).abs().sum() / mask.sum()

        mse = masked_squared_diff.sum() / mask.sum()
        mse_all += mse.item() * label.shape[0]
        
        mae_all += mae.item() * label.shape[0]
        
        
        total += label.shape[0]


        y_true.extend(label.tolist())
        y_pred.extend(output.tolist())
        uniprots.extend([d.split(",")[0] for d in data["source"]])
        pdbs.extend([d.split(",")[1] for d in data["source"]])
    
    predicts = np.array(y_pred)
    targets = np.array(y_true)

    label_types = ["ic50", "ec50", "kd", "ki", "potency"]
            
            
    res_dic_pdb = {}
    res_dic_uniprot = {}



    for type in label_types:
        res_dic_pdb[type] = {}
        res_dic_uniprot[type] = {}


    for i in range(len(predicts)):
        for j in range(5):
            if targets[i][j]>0:
                type = label_types[j]
                if pdbs[i] not in res_dic_pdb[type]:
                    res_dic_pdb[type][pdbs[i]] = []
                res_dic_pdb[type][pdbs[i]].append((predicts[i][j], targets[i][j]))

                if uniprots[i] not in res_dic_uniprot[type]:
                    res_dic_uniprot[type][uniprots[i]] = []
                res_dic_uniprot[type][uniprots[i]].append((predicts[i][j], targets[i][j]))
                

    # get correlation for each type

    for type in label_types:
        pr_list = []
        spr_list = []
        for pdb in res_dic_pdb[type]:
            pred = [item[0] for item in res_dic_pdb[type][pdb]]
            target = [item[1] for item in res_dic_pdb[type][pdb]]
            if len(pred) <2:
                continue
            # get pearson correlation
            

            pr_list.append(pearsonr(pred, target)[0])
            spr_list.append(spearmanr(pred, target)[0])
            

        if len(pr_list) == 0:
            pr_list = [np.nan]
        if len(spr_list) == 0:
            spr_list = [np.nan]

        rp = np.nanmean(pr_list)
        rs = np.nanmean(spr_list)
        
        # wandb log valid

        wandb.log({f'{split}_{type}_pdb_rp': rp})

        wandb.log({f'{split}_{type}_pdb_rs': rs})

    
    # get type level

    for type in label_types:

        predicts = []
        targets = []
        for pdb in res_dic_pdb[type]:
            for item in res_dic_pdb[type][pdb]:
                predicts.append(item[0])
                targets.append(item[1])
        predicts = np.array(predicts)
        targets = np.array(targets)
        if len(predicts) == 0:
            continue

        pearson = pearsonr(predicts, targets)[0]
        spearman = spearmanr(predicts, targets)[0]
        wandb.log({f'{split}_{type}_all_rp': pearson})
        wandb.log({f'{split}_{type}_all_rs': spearman})

        mse = np.mean((predicts - targets) ** 2)
        mae = np.mean(np.abs(predicts - targets))
        rmse = np.sqrt(mse)
        wandb.log({f'{split}_{type}_all_rmse': rmse})
        wandb.log({f'{split}_{type}_all_mae': mae})
    
    



    rmse = np.sqrt(mse_all / total)

    mae = mae_all / total

    wandb.log({f'{split}_rmse': rmse})
    wandb.log({f'{split}_mae': mae})



    return mse_all / total, mae_all / total, y_true, y_pred


def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)


def train(args, device, test_mode=False):
    print("Training model with config:")
    print(str(json.dumps(args.__dict__, indent=4)) + "\n")

    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    train_dataset = LMDBDataset(os.path.join(args.data_dir, "train"),
                                transform=CNN3D_TransformLBA(random_seed=args.random_seed))
    val_dataset = LMDBDataset(os.path.join(args.data_dir,  "val"),
                              transform=CNN3D_TransformLBA(random_seed=args.random_seed))
    test_dataset = LMDBDataset(os.path.join(args.data_dir,  "test"),
                               transform=CNN3D_TransformLBA(random_seed=args.random_seed))

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    for data in train_loader:
        in_channels, spatial_size = data['feature'].size()[1:3]
        print('num channels: {:}, spatial size: {:}'.format(in_channels, spatial_size))
        break

    model = conv_model(in_channels, spatial_size, args)
    #print(model)
    model.to(device)

    best_val_mse = np.Inf
    best_rp = 0
    best_rs = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)





    for epoch in range(1, args.num_epochs+1):
        start = time.time()
        train_loss = train_loop(model, train_loader, optimizer, device)
        print('train_loss:', train_loss)
        val_mse, val_mae, y_true, y_pred = test(model, val_loader, device, "val")
        
        if val_mse < best_val_mse:
            print(f"\nSave model at epoch {epoch:03d}, val_mse: {val_mse:.4f}")
            save_weights(model, os.path.join(args.output_dir, f'best_weights.pt'))
            best_val_mse = val_mse
            test_mse, test_mae, y_true, y_pred = test(model, test_loader, device, "test")

    elapsed = (time.time() - start)
    print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
    print('\tTrain RMSE: {:.7f}, Val RMSE: {:.7f}'.format(train_loss, np.sqrt(val_mse)))
    # logger.info('{:03d}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\n'.format(epoch, train_loss, val_loss, r_p, r_s))


    if test_mode:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, f'best_weights.pt')))
        rmse, pearson, spearman, test_df = test(model, test_loader, device)
        test_df.to_pickle(os.path.join(args.output_dir, 'test_results.pkl'))
        print('Test RMSE: {:.7f}, Pearson R: {:.7f}, Spearman R: {:.7f}'.format(
            rmse, pearson, spearman))
        test_file = os.path.join(args.output_dir, f'test_results.txt')
        with open(test_file, 'a+') as out:
            out.write('{}\t{:.7f}\t{:.7f}\t{:.7f}\n'.format(
                args.random_seed, rmse, pearson, spearman))

    return 1


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./gnn_cnn_data/split_60")
    parser.add_argument('--mode', type=str, default='test',
                        choices=['train', 'test'])
    parser.add_argument('--output_dir', type=str, default="/data/protein/cnn_logs")
    parser.add_argument('--unobserved', action='store_true', default=False)

    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--conv_drop_rate', type=float, default=0.1)
    parser.add_argument('--fc_drop_rate', type=float, default=0.25)
    parser.add_argument('--num_epochs', type=int, default=20)

    parser.add_argument('--num_conv', type=int, default=4)
    parser.add_argument('--batch_norm', action='store_true', default=False)
    parser.add_argument('--no_dropout', action='store_true', default=False)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--random_seed', type=int, default=int(np.random.randint(1, 10e6)))

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # wandb

    import wandb
    wandb.init(project="lba_cnn", reinit=True)

    # Set up output dir
    args.output_dir = os.path.join(args.output_dir, 'lba')
    assert args.output_dir != None
    if args.unobserved:
        args.output_dir = os.path.join(args.output_dir, 'None')
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        num = 0
        while True:
            dirpath = os.path.join(args.output_dir, str(num))
            if os.path.exists(dirpath):
                num += 1
            else:
                args.output_dir = dirpath
                print('Creating output directory {:}'.format(args.output_dir))
                os.makedirs(args.output_dir)
                break

    print(f"Running mode {args.mode:} with seed {args.random_seed:} "
          f"and output dir {args.output_dir}")
    train(args, device, args.mode=='test')
