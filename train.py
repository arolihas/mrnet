import argparse
import json
import numpy as np
import os
import torch
import glob

from datetime import datetime
from pathlib import Path
from sklearn import metrics

from evaluate import run_model
from loader import external_load_data, mr_load_data
from model import MRNet

def train(rundir, diagnosis, dataset, epochs, learning_rate, use_gpu, attention):
    models = []
    if (dataset == 0):
        train_loader, valid_loader, test_loader = external_load_data(diagnosis, use_gpu)
        models.append((MRNet(useMultiHead = attention), train_loader, valid_loader, 'external_validation'))
    elif (dataset == 1):
        train_loaders, valid_loaders = mr_load_data(diagnosis, use_gpu)
        train_loader_sag, train_loader_ax, train_loader_cor = train_loaders
        valid_loader_sag, valid_loader_ax, valid_loader_cor = valid_loaders
        models = [(MRNet(max_layers=51), train_loader_sag, valid_loader_sag, 'sagittal'),
                  (MRNet(max_layers=61),train_loader_ax, valid_loader_ax, 'axial'),
                  (MRNet(max_layers=58), train_loader_cor, valid_loader_cor, 'coronal')]

    for model, train_loader, valid_loader, fname in models:
        if use_gpu:
            model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.3, threshold=1e-4)

        best_val_loss = float('inf')

        start_time = datetime.now()

        for epoch in range(epochs):
            change = datetime.now() - start_time
            print('starting epoch {}. time passed: {}'.format(epoch+1, str(change)))
            
            train_loss, train_auc, _, _ = run_model(model, train_loader, train=True, optimizer=optimizer)
            print(f'train loss: {train_loss:0.4f}')
            print(f'train AUC: {train_auc:0.4f}')

            val_loss, val_auc, _, _ = run_model(model, valid_loader)
            print(f'valid loss: {val_loss:0.4f}')
            print(f'valid AUC: {val_auc:0.4f}')

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                file_name = f'val{val_loss:0.4f}_train{train_loss:0.4f}_epoch{epoch+1}'
                save_path = Path(rundir) / fname / file_name
                folder = rundir + '/' + fname
                try:
                    for f in os.listdir(folder):
                        os.remove(folder + '/' + f)
                    torch.save(model, save_path)
                except:
                    os.makedirs(rundir+'/'+fname)
                    torch.save(model, save_path)
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rundir', type=str, required=True)
    parser.add_argument('--diagnosis', type=int, required=True)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--learning_rate', default=1e-05, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--max_patience', default=5, type=int)
    parser.add_argument('--factor', default=0.3, type=float)
    parser.add_argument('--dataset', default=0, type=int)
    parser.add_argument('--attention', action='store_true')
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.rundir, exist_ok=True)

    with open(Path(args.rundir) / 'args.json', 'w') as out:
        json.dump(vars(args), out, indent=4)

    train(args.rundir, args.diagnosis, args.dataset, args.epochs, args.learning_rate, args.gpu, args.attention)
