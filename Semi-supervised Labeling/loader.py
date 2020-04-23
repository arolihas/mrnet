import numpy as np
import pandas as pd
import os
import pickle
import torch
import torch.nn as nn
import torch.utils.data as data

from torch.autograd import Variable

INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73

MRPATH = '../mrnet_data/'

class MRDataset(data.Dataset):
    def __init__(self, train=True, transform=None, weights=None):
        super().__init__()

        self.root_dir = MRPATH
        self.train = train

        if self.train:
            self.folder_path = self.root_dir + 'train/'
        else:
            self.folder_path = self.root_dir + 'valid/'

        self.paths = []
        self.labels = []

        folder = self.folder_path + "axial/"
        self.paths_A = [folder + filename for filename in os.listdir(folder) if ".npy" in filename]
        folder = self.folder_path + "coronal/"
        self.paths_S = [folder + filename for filename in os.listdir(folder) if ".npy" in filename]
        folder = self.folder_path + "sagittal/"
        self.paths_C = [folder + filename for filename in os.listdir(folder) if ".npy" in filename]

        self.paths += self.paths_A
        self.labels += [0 for i in range(len(self.paths_A))]
        self.paths += self.paths_C
        self.labels += [1 for i in range(len(self.paths_A))]
        self.paths += self.paths_S
        self.labels += [2 for i in range(len(self.paths_A))]


        self.transform = transform
        if weights == "scale":
            self.weights = torch.Tensor([len(self.paths_A)/len(self.paths), len(self.paths_C)/len(self.paths), len(self.paths_S)/len(self.paths)])
        else:
            self.weights = None

        self.entropy = nn.CrossEntropyLoss(weight = self.weights)
        #add weights maybe

    def weighted_loss(self, prediction, target):
        # prediction, target  N x 3 tensors
        loss = self.entropy(prediction, target.view(-1))
        return loss

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        array = np.load(self.paths[index])
        label = torch.LongTensor([self.labels[index]])

        if self.transform:
            array = self.transform(array)
        else:
            array = np.stack((array,)*3, axis=1)
            array = torch.FloatTensor(array)

        return array, label

def mr_load_data():

    train_loaders = []
    valid_loaders = []

    train_dataset = MRDataset()
    valid_dataset = MRDataset(train=False)

    train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=8, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=1, num_workers=8, shuffle=False)

    return train_loader, valid_loader






"""

dat = MRDataset(train=True)
print(len(dat.labels))
print(len(dat))
print(dat[1002][1])
#print(dat.labels)
"""
