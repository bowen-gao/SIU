# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from scipy.stats import pearsonr, spearmanr


@register_loss("finetune_mse")
class FinetuneMSELoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            features_only=True,
        )
        #print(net_output.shape)
        reg_output = net_output
        loss = self.compute_loss(model, reg_output, sample, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        source = sample["pocket_name"]
        uniprots = [source.split(",")[0] for source in source]
        pdbs = [source.split(",")[1] for source in source]
        
        if not self.training:
            #if self.task.mean and self.task.std:
            #    targets_mean = torch.tensor(self.task.mean, device=reg_output.device)
            #    targets_std = torch.tensor(self.task.std, device=reg_output.device)
            #    reg_output = reg_output * targets_std + targets_mean
            logging_output = {
                "loss": loss.data,
                "predict": reg_output.data,
                "target": sample["target"]["finetune_target"].data,
                "sample_size": sample_size,
                "uniprots": uniprots,
                "pdbs": pdbs,
                "num_task": 5,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        predicts = net_output.float()
        targets = (
            sample["target"]["finetune_target"].float()
        )
        #print(predicts.shape, targets.shape)
        assert(targets.shape == predicts.shape)
        #if self.task.mean and self.task.std:
        #    targets_mean = torch.tensor(self.task.mean, device=targets.device)
        #    targets_std = torch.tensor(self.task.std, device=targets.device)
        #    targets = (targets - targets_mean) / targets_std

        # # mask targets that have value < 0 
        # mask = targets >= 0

        # mse = torch.sqrt()

        # loss = F.mse_loss(
        #     predicts,
        #     targets,
        #     reduction="sum" if reduce else "none",
        # )

        # calculate mse loss with mask

        # Calculate the squared difference

        mask = targets >= 0

        squared_diff = (predicts - targets) ** 2
        
        # Apply the mask
        masked_squared_diff = squared_diff * mask
        
        # Compute the mean of the masked squared differences

        #print(predicts[:3], targets[:3])
        loss = masked_squared_diff.sum() / mask.sum()

        

        #print(loss)

        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        #sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        sample_size = len(logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        #print(sample_size)
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            f'{split}_loss', loss_sum / sample_size, sample_size, round=3
        )
        if "valid" in split or "test" in split:
            predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
            predicts = predicts.detach().cpu().numpy()
            targets = torch.cat([log.get("target") for log in logging_outputs], dim=0)
            targets = targets.detach().cpu().numpy()

            
            label_types = ['ic50', 'ki', 'kd', 'ec50', "potency"]
            
            
            res_dic_pdb = {}
            res_dic_uniprot = {}



            for type in label_types:
                res_dic_pdb[type] = {}
                res_dic_uniprot[type] = {}

            pdbs = []
            for log in logging_outputs:
                pdbs.extend(log.get("pdbs"))

            uniprots = []
            for log in logging_outputs:
                uniprots.extend(log.get("uniprots"))

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
                
                metrics.log_scalar(f"{split}_{type}_pdb_rp", rp, sample_size, round=4)
                metrics.log_scalar(f"{split}_{type}_pdb_rs", rs, sample_size, round=4)

            
            
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
                metrics.log_scalar(f"{split}_{type}_all_rp", pearson, sample_size, round=4)
                metrics.log_scalar(f"{split}_{type}_all_rs", spearman, sample_size, round=4)

                mse = ((predicts - targets) ** 2).mean()
                #metrics.log_scalar(f"{split}_{type}_all_mse", mse, sample_size, round=4)
                rmse = np.sqrt(mse)
                metrics.log_scalar(f"{split}_{type}_rmse", rmse, sample_size, round=4)
                mae = np.abs(predicts - targets).mean()
                metrics.log_scalar(f"{split}_{type}_mae", mae, sample_size, round=4)



            
            




            

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train


@register_loss("finetune_mse_single")
class FinetuneMSESingleLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            features_only=True,
        )
        #print(net_output.shape)
        reg_output = net_output.squeeze(1)
        loss = self.compute_loss(model, reg_output, sample, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        source = sample["pocket_name"]
        uniprots = [source.split(",")[0] for source in source]
        pdbs = [source.split(",")[1] for source in source]
        
        if not self.training:
            #if self.task.mean and self.task.std:
            #    targets_mean = torch.tensor(self.task.mean, device=reg_output.device)
            #    targets_std = torch.tensor(self.task.std, device=reg_output.device)
            #    reg_output = reg_output * targets_std + targets_mean
            logging_output = {
                "loss": loss.data,
                "predict": reg_output.data,
                "target": sample["target"]["finetune_target"].data,
                "sample_size": sample_size,
                "uniprots": uniprots,
                "pdbs": pdbs,
                "num_task": 1,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        predicts = net_output.float()
        targets = (
            sample["target"]["finetune_target"].float()
        )
        

        loss = F.mse_loss(
            predicts,
            targets,
            reduction="mean" if reduce else "none",
        )

        # get pairwise difference loss for all

        targets = targets.view(-1, 1)

        targets = targets.repeat(1, targets.size(0))

        pair_diff_targets = targets - targets.t()

        pair_diff_predicts = predicts.view(-1, 1).repeat(1, predicts.size(0)) - predicts.view(-1, 1).repeat(1, predicts.size(0)).t()

        pair_diff_loss = F.mse_loss(pair_diff_predicts, pair_diff_targets, reduction="mean" if reduce else "none")

        
        #loss = loss + pair_diff_loss
        

        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        #sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        sample_size = len(logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        #print(sample_size)
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            f'{split}_loss', loss_sum / sample_size, sample_size, round=3
        )
        if "test" in split or "test" in split:
            predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
            predicts = predicts.detach().cpu().numpy()
            predicts = predicts.reshape((-1, 1))
            targets = torch.cat([log.get("target") for log in logging_outputs], dim=0)
            targets = targets.detach().cpu().numpy()
            targets = targets.reshape((-1, 1))

            
            label_types = ['ic50', 'ki', 'kd', 'ec50', "potency"]

            # ori_types = ["ic50", "ec50", "ki", "kd"]

            label_types = ["ki"]
            
            
            res_dic_pdb = {}
            res_dic_uniprot = {}



            for type in label_types:
                res_dic_pdb[type] = {}
                res_dic_uniprot[type] = {}

            pdbs = []
            for log in logging_outputs:
                pdbs.extend(log.get("pdbs"))

            uniprots = []
            for log in logging_outputs:
                uniprots.extend(log.get("uniprots"))

            for i in range(len(predicts)):
                for j in range(1):
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
                
                metrics.log_scalar(f"{split}_{type}_pdb_rp", rp, sample_size, round=4)
                metrics.log_scalar(f"{split}_{type}_pdb_rs", rs, sample_size, round=4)

            
            
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
                metrics.log_scalar(f"{split}_{type}_all_rp", pearson, sample_size, round=4)
                metrics.log_scalar(f"{split}_{type}_all_rs", spearman, sample_size, round=4)

                mse = ((predicts - targets) ** 2).mean()
                #metrics.log_scalar(f"{split}_{type}_all_mse", mse, sample_size, round=4)
                rmse = np.sqrt(mse)
                metrics.log_scalar(f"{split}_{type}_rmse", rmse, sample_size, round=4)
                mae = np.abs(predicts - targets).mean()
                metrics.log_scalar(f"{split}_{type}_mae", mae, sample_size, round=4)



            
            




            

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train

@register_loss("finetune_mae")
class FinetuneMAELoss(FinetuneMSELoss):
    def __init__(self, task):
        super().__init__(task)

    def compute_loss(self, model, net_output, sample, reduce=True):
        predicts = net_output.view(-1, 1).float()
        targets = (
            sample["target"]["finetune_target"].view(-1, 1).float()
        )
        #if self.task.mean and self.task.std:
        #    targets_mean = torch.tensor(self.task.mean, device=targets.device)
        #    targets_std = torch.tensor(self.task.std, device=targets.device)
        #    targets = (targets - targets_mean) / targets_std
        loss = F.l1_loss(
            predicts,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss


@register_loss("finetune_smooth_mae")
class FinetuneSmoothMAELoss(FinetuneMSELoss):
    def __init__(self, task):
        super().__init__(task)

    def compute_loss(self, model, net_output, sample, reduce=True):
        predicts = net_output.view(-1, 1).float()
        targets = (
            sample["target"]["finetune_target"].view(-1, 1).float()
        )
        #if self.task.mean and self.task.std:
        #    targets_mean = torch.tensor(self.task.mean, device=targets.device)
        #    targets_std = torch.tensor(self.task.std, device=targets.device)
        #    targets = (targets - targets_mean) / targets_std
        loss = F.smooth_l1_loss(
            predicts,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            #num_task = logging_outputs[0].get("num_task", 0)
            #conf_size = logging_outputs[0].get("conf_size", 0)
            y_true = (
                torch.cat([log.get("target", 0) for log in logging_outputs], dim=0)
                .view(-1, 1)
                .cpu()
                .numpy()
                .mean(axis=1)
            )
            y_pred = (
                torch.cat([log.get("predict") for log in logging_outputs], dim=0)
                .view(-1, 1)
                .cpu()
                .numpy()
                .mean(axis=1)
            )
            agg_mae = np.abs(y_pred - y_true).mean()
            metrics.log_scalar(f"{split}_agg_mae", agg_mae, sample_size, round=4)


@register_loss("finetune_mse_pocket")
class FinetuneMSEPocketLoss(FinetuneMSELoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            features_only=True
        )
        reg_output = net_output[0]
        loss = self.compute_loss(model, reg_output, sample, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            #if self.task.mean and self.task.std:
            #    targets_mean = torch.tensor(self.task.mean, device=reg_output.device)
            #    targets_std = torch.tensor(self.task.std, device=reg_output.device)
            #    reg_output = reg_output * targets_std + targets_mean
            logging_output = {
                "loss": loss.data,
                "predict": reg_output.view(-1, 1).data,
                "target": sample["target"]["finetune_target"]
                .view(-1, 1)
                .data,
                "sample_size": sample_size,
                "num_task": 1,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
            if predicts.size(-1) == 1:
                # single label regression task
                targets = torch.cat(
                    [log.get("target", 0) for log in logging_outputs], dim=0
                )
                df = pd.DataFrame(
                    {
                        "predict": predicts.view(-1).cpu(),
                        "target": targets.view(-1).cpu(),
                    }
                )
                mse = ((df["predict"] - df["target"]) ** 2).mean()
                metrics.log_scalar(f"{split}_mse", mse, sample_size, round=3)
                metrics.log_scalar(f"{split}_rmse", np.sqrt(mse), sample_size, round=4)
