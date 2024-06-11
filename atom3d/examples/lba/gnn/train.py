import argparse
import logging
import os
import time
import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from model import GNN_LBA
from data import GNNTransformLBA
from atom3d.datasets import LMDBDataset, PTGDataset
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import wandb

def train_loop(model, loader, optimizer, device):
    model.train()

    loss_all = 0
    total = 0
    for data in tqdm(loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr.view(-1), data.batch)
        label = torch.tensor(data.y, dtype=torch.float32).to(device)

        #print(label.shape)
        #loss = F.mse_loss(output, data.y)
        mask = label >= 0

        squared_diff = (output - label) ** 2
        
        # Apply the mask
        masked_squared_diff = squared_diff * mask
        
        # Compute the mean of the masked squared differences

        #print(predicts[:3], targets[:3])
        loss = masked_squared_diff.sum() / mask.sum()
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        total += data.num_graphs
        optimizer.step()
        #('loss:', loss.item())
        wandb.log({'train_loss': loss.item()})
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
        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr.view(-1), data.batch)
        #loss = F.mse_loss(output, data.y)
        label = torch.tensor(data.y, dtype=torch.float32).to(device)
        #print(label.shape)
        #loss = F.mse_loss(output, data.y)
        mask = label >= 0

        squared_diff = (output - label) ** 2
        
        # Apply the mask
        masked_squared_diff = squared_diff * mask
        
        
        # compute mae

        mae = (output - label).abs().sum() / mask.sum()

        mse = masked_squared_diff.sum() / mask.sum()
        mse_all += mse.item() * data.num_graphs
        
        mae_all += mae.item() * data.num_graphs
        
        
        total += data.num_graphs


        y_true.extend(label.tolist())
        y_pred.extend(output.tolist())
        uniprots.extend([d.split(",")[0] for d in data.source])
        pdbs.extend([d.split(",")[1] for d in data.source])
    
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

# def plot_corr(y_true, y_pred, plot_dir):
#     plt.clf()
#     sns.scatterplot(y_true, y_pred)
#     plt.xlabel('Actual -log(K)')
#     plt.ylabel('Predicted -log(K)')
#     plt.savefig(plot_dir)

def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)

def train(args, device, log_dir, rep=None, test_mode=False):
    # logger = logging.getLogger('lba')
    # logger.basicConfig(filename=os.path.join(log_dir, f'train_{split}_cv{fold}.log'),level=logging.INFO)

    if args.precomputed:
        train_dataset = PTGDataset(os.path.join(args.data_dir, 'train'))
        val_dataset = PTGDataset(os.path.join(args.data_dir, 'val'))
        test_dataset = PTGDataset(os.path.join(args.data_dir, 'test'))
    else:
        transform=GNNTransformLBA()
        train_dataset = LMDBDataset(os.path.join(args.data_dir, 'train'), transform=transform)
        val_dataset = LMDBDataset(os.path.join(args.data_dir, 'val'), transform=transform)
        test_dataset = LMDBDataset(os.path.join(args.data_dir, 'test'), transform=transform)
    
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4)

    for data in train_loader:
        num_features = data.num_features
        break

    model = GNN_LBA(num_features, hidden_dim=args.hidden_dim).to(device)
    model.to(device)

    best_val_mse = 999

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.num_epochs+1):
        start = time.time()
        train_loss = train_loop(model, train_loader, optimizer, device)
        print('train_loss:', train_loss)
        val_mse, val_mae, y_true, y_pred = test(model, val_loader, device, "val")
        
        if val_mse < best_val_mse:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                }, os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
            # plot_corr(y_true, y_pred, os.path.join(log_dir, f'corr_{split}.png'))
            best_val_mse = val_mse
            test_mse, test_mae, y_true, y_pred = test(model, test_loader, device, "test")

        elapsed = (time.time() - start)
        print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
        print('\tTrain RMSE: {:.7f}, Val RMSE: {:.7f}'.format(train_loss, np.sqrt(val_mse)))
        # logger.info('{:03d}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\n'.format(epoch, train_loss, val_loss, r_p, r_s))

        



    if test_mode:
        train_file = os.path.join(log_dir, f'lba-rep{rep}.best.train.pt')
        val_file = os.path.join(log_dir, f'lba-rep{rep}.best.val.pt')
        test_file = os.path.join(log_dir, f'lba-rep{rep}.best.test.pt')
        cpt = torch.load(os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
        model.load_state_dict(cpt['model_state_dict'])
        _, _, _, y_true_train, y_pred_train = test(model, train_loader, device)
        torch.save({'targets':y_true_train, 'predictions':y_pred_train}, train_file)
        _, _, _, y_true_val, y_pred_val = test(model, val_loader, device)
        torch.save({'targets':y_true_val, 'predictions':y_pred_val}, val_file)
        rmse, pearson, spearman, y_true_test, y_pred_test = test(model, test_loader, device)
        print(f'\tTest RMSE {rmse}, Test Pearson {pearson}, Test Spearman {spearman}')
        torch.save({'targets':y_true_test, 'predictions':y_pred_test}, test_file)

    return 1


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/data/protein/ssmopi_train_data/atom3d/split_90")
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--seqid', type=int, default=30)
    parser.add_argument('--precomputed', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = args.log_dir

    # init wandb


    wandb = wandb.init(project='lba', reinit=True)






    if args.mode == 'train':
        if log_dir is None:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dir = os.path.join('logs', now)
        else:
            log_dir = os.path.join('logs', log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        train(args, device, log_dir)
        
    elif args.mode == 'test':
        for rep, seed in enumerate(np.random.randint(0, 1000, size=3)):
            print('seed:', seed)
            log_dir = os.path.join('logs', f'lba_test_withH_{args.seqid}')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            np.random.seed(seed)
            torch.manual_seed(seed)
            train(args, device, log_dir, rep, test_mode=True)
