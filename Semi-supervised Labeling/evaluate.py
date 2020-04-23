import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import torch

from torch.autograd import Variable

from loader import mr_load_data
from model import LabellingMRNet

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
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

        #vol, label = batch
        #print(str(vol.shape) + "   ---  "  + str(label[0][0]))
        num_batches += 1
        print(num_batches)
        #continue


        if train:
            optimizer.zero_grad()

        vol, label = batch
        vol = Variable(vol)
        label = Variable(label)

        if loader.dataset.use_gpu:
            vol = vol.cuda()
            label = label.cuda()
            model = model.cuda()


        pred = model.forward(vol)

        loss = loader.dataset.weighted_loss(pred, label)
        total_loss += loss.item()

        if train:
            loss.backward()
            optimizer.step()

        pred_npy = pred.data.cpu().numpy()
        label_npy = label.data.cpu().numpy()

        preds.append(pred_npy)
        labels.append(label_npy)

    avg_loss = total_loss / num_batches

    return avg_loss, preds, labels

def evaluate(model_path, use_gpu):
    loader = mr_load_data()

    model = LabellingMRNet()
    state_dict = torch.load(model_path, map_location=(None if use_gpu else 'cpu'))
    model.load_state_dict(state_dict)

    if use_gpu:
        model = model.cuda()

    loss, preds, labels = run_model(model, loader)

    print(f'{split} loss: {loss:0.4f}')
    print(f'{split} AUC: {auc:0.4f}')

    return preds, labels

if __name__ == '__main__':
    args = get_parser().parse_args()
    evaluate(args.model_path, args.gpu)
