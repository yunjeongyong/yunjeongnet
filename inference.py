import os
from config import Config

import torch
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr, pearsonr
from utils.process import RandCrop, ToTensor, RandHorizontalFlip, Normalize, five_point_crop
from einops import rearrange, repeat


config = Config({
        # dataset path
        "db_name": "KADID-10k",
        "train_csv_path": "C:\\Users\\yunjeongyong\\Desktop\\intern\\IQA\\iqa-db\\all_data_csv\\KADID-10k.csv",
        "train_data_path": "C:\\Users\\yunjeongyong\\Desktop\\intern\\IQA\\iqa-db\\",
        "train_txt_file_name": "./data/pipal21_train.txt",
        "val_txt_file_name": "./data/pipal21_val.txt",

        # optimization
        "batch_size": 1,
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "n_epoch": 10,
        "val_freq": 1,
        "T_max": 50,
        "eta_min": 0,
        "num_avg_val": 5,
        "crop_size": 224,
        "num_workers": 0,

        # model
        "patch_size": 8,
        "img_size": 224,
        "embed_dim": 768,
        "dim_mlp": 768,
        "num_heads": [4, 4],
        "window_size": 4,
        "depths": [2, 2],
        "num_outputs": 1,
        "num_tab": 2,
        "scale": 0.13,

        # load & save checkpoint
        "model_name": "model_maniqa",
        "output_path": "./output",
        "snap_path": "./output/models/",  # directory for saving checkpoint
        "log_path": "./output/log/maniqa/",
        "log_file": ".txt",
        "tensorboard_path": "./output/tensorboard/"
    })
def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return torch.stack(list(tuple_of_tensors), dim=0)

def train_epoch(config, epoch, model_transformer, model_backbone, criterion, optimizer, scheduler, train_loader):
    losses = []
    model_transformer.train()
    model_backbone.train()

    pred_epoch = []
    labels_epoch = []
    for img, dmos, name in tqdm(train_loader):
        x_d = img.cuda()
        labels = dmos.float().cuda()

        x_d = rearrange(x_d, 'b h w c -> b c h w')
        feat_d = model_backbone(x_d)

        optimizer.zero_grad()


        pred = model_transformer(feat_d)
        loss = criterion(pred, labels)
        loss_val = loss.item()
        losses.append(loss_val)

        loss.backward()
        optimizer.step()
        scheduler.step()


        # save results in one epoch
        pred_batch_numpy = pred.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)

        # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    # compute correlation coefficient
    print('[train] epoch:%d / loss:%f / SROCC:%4f / PLCC:%4f' % (epoch + 1, loss.item(), rho_s, rho_p))

    # save weights
    if (epoch + 1) % config.save_freq == 0:
        weights_file_name = "epoch%d.pth" % (epoch + 1)
        weights_file = os.path.join(config.snap_path, weights_file_name)
        torch.save({
            'epoch': epoch,
            'model_backbone_state_dict': model_backbone.state_dict(),
            'model_transformer_state_dict': model_transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss
        }, weights_file)
        print('save weights of epoch %d' % (epoch + 1))

    return np.mean(losses), rho_s, rho_p

def eval_epoch(config, epoch, model_transformer, model_backbone, criterion, test_loader):
    with torch.no_grad():
        losses = []
        model_transformer.eval()
        model_backbone.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for img, dmos, name in tqdm(test_loader):
            predd = 0
            for i in range(config.num_avg_val):
                x_d = img.cuda()
                labels = dmos.float().cuda()

                x_d = five_point_crop(i, d_img=x_d, config=config)
                feat_d = model_backbone(x_d)
                pred_d = model_transformer(feat_d)
                predd += model_backbone(pred_d)

            predd /= config.num_avg_val
            # compute loss
            loss = criterion(predd, labels)
            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = predd.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)

            # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        # compute correlation coefficient
        print('[train] epoch:%d / loss:%f / SROCC:%4f / PLCC:%4f' % (epoch + 1, loss.item(), rho_s, rho_p))

        return np.mean(losses), rho_s, rho_p

