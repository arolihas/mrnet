import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import torch

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from torch.autograd import Variable

from loader import external_load_data, mr_load_data
from model import MRNet

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--diagnosis', type=int, required=True)
    parser.add_argument('--dataset', type=int, required=True)
    parser.add_argument('--gpu', action='store_true')
    return parser

def run_model(model, loader, train=False, optimizer=None):
    preds = []
    labels = []

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.
    num_batches = 0

    for batch in loader:
        if train:
            optimizer.zero_grad()

        vol, label = batch
        vol = Variable(vol)
        label = Variable(label)
        if loader.dataset.use_gpu:
            vol = vol.cuda()
            label = label.cuda()
            model = model.cuda()

        logit = model.forward(vol)

        loss = loader.dataset.weighted_loss(logit, label)
        total_loss += loss.item()

        pred = torch.sigmoid(logit)
        pred_npy = pred.data.cpu().numpy()[0][0]
        label_npy = label.data.cpu().numpy()[0][0]

        preds.append(pred_npy)
        labels.append(label_npy)

        if train:
            loss.backward()
            optimizer.step()
        num_batches += 1

    avg_loss = total_loss / num_batches

    fpr, tpr, threshold = metrics.roc_curve(labels, preds)
    auc = metrics.auc(fpr, tpr)

    return avg_loss, auc, preds, labels

def evaluate(split, model_path, diagnosis, dataset, use_gpu):
    preds = None
    labels = None
    if dataset == 0:
        train_loader, valid_loader, test_loader = external_load_data(diagnosis, use_gpu)
        model = MRNet()
        state_dict = torch.load(model_path, map_location=(None if use_gpu else 'cpu'))
        model.load_state_dict(state_dict)
        if use_gpu:
            model = model.cuda()

        if split == 'train':
            loader = train_loader
        elif split == 'valid':
            loader = valid_loader
        elif split == 'test':
            loader = test_loader
        else:
            raise ValueError("split must be 'train', 'valid', or 'test'")

        loss, auc, preds, labels = run_model(model, loader)

        print(f'{split} loss: {loss:0.4f}')
        print(f'{split} AUC: {auc:0.4f}')
                
    if dataset == 1:
        train_loaders, valid_loaders = mr_load_data(diagnosis, use_gpu)
        
        model_sag = MRNet(max_layers=51)
        model_ax = MRNet(max_layers=61)
        model_cor = MRNet(max_layers=58)
        
        path_s = os.listdir(model_path + '/sagittal')
        path_a = os.listdir(model_path + '/axial')
        path_c = os.listdir(model_path + '/coronal')

        ps = [0 if 'h' in x[-2:] else int(x[-2:]) for x in path_s]
        pa = [0 if 'h' in x[-2:] else int(x[-2:]) for x in path_a]
        pc = [0 if 'h' in x[-2:] else int(x[-2:]) for x in path_c]

        epoch_sag = max(ps)
        epoch_ax= max(pa)
        epoch_cor = max(pc)

        model_path_sag = path_s[ps.index(epoch_sag)]
        model_path_ax = path_a[pa.index(epoch_ax)]
        model_path_cor = path_c[pc.index(epoch_cor)]

        state_dict_sag = torch.load(model_path + '/sagittal/' + model_path_sag, map_location=(None if use_gpu else 'cpu'))
        state_dict_ax = torch.load(model_path + '/axial/' + model_path_ax, map_location=(None if use_gpu else 'cpu'))
        state_dict_cor = torch.load(model_path + '/coronal/' + model_path_cor, map_location=(None if use_gpu else 'cpu'))
        
        model_sag.load_state_dict(state_dict_sag)
        model_ax.load_state_dict(state_dict_ax)
        model_cor.load_state_dict(state_dict_cor)

        if use_gpu:
            model_sag = model_sag.cuda()
            model_ax = model_ax.cuda()
            model_cor = model_cor.cuda()

        loss_sag, auc_sag, t_preds_sag, labels_sag = run_model(model_sag, train_loaders[0])
        _, _, preds_sag, _ = run_model(model_sag, valid_loaders[0])
        print(f'sagittal {split} loss: {loss_sag:0.4f}')
        print(f'sagittal {split} AUC: {auc_sag:0.4f}')
        loss_ax, auc_ax, t_preds_ax, labels_ax = run_model(model_ax, train_loaders[1])
        _, _, preds_ax, _ = run_model(model_ax, valid_loaders[1])
        print(f'axial {split} loss: {loss_ax:0.4f}')
        print(f'axial {split} AUC: {auc_ax:0.4f}')
        loss_cor, auc_cor, t_preds_cor, labels_cor = run_model(model_cor, train_loaders[2])
        _, _, preds_cor, valid_labels = run_model(model_cor, valid_loaders[2])
        print(f'coronal {split} loss: {loss_cor:0.4f}')
        print(f'coronal {split} AUC: {auc_cor:0.4f}')
        
        X = np.zeros((len(t_preds_cor), 3))
        X[:, 0] = t_preds_sag
        X[:, 1] = t_preds_ax
        X[:, 2] = t_preds_cor

        y = np.array(labels_cor)
        lgr = LogisticRegression(solver='lbfgs')
        lgr.fit(X,y)

        X_valid = np.zeros((len(preds_cor), 3))
        X_valid[:, 0] = preds_sag
        X_valid[:, 1] = preds_ax
        X_valid[:, 2] = preds_cor

        y_preds = lgr.predict(X)
        y_true = np.array(valid_labels)
        print(metrics.roc_auc_score(y_true, y_preds))
        print(metrics.classification_report(y_true, y_preds, target_names=['class 0', 'class 1']))

    return preds, labels

if __name__ == '__main__':
    args = get_parser().parse_args()
    evaluate(args.split, args.model_path, args.diagnosis, args.dataset, args.gpu)
