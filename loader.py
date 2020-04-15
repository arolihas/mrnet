import numpy as np
import pandas as pd
import os
import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as data

from torch.autograd import Variable

INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73
PATH = 'external_validation/'
MRPATH = 'mrnet_data/'

class MRDataset(data.Dataset):
    def __init__(self, task, plane, train=True, transform=None, weights=None):
        super().__init__()
        if task == 0:
            task = 'abnormal'
        elif task == 1:
            task = 'acl'
        else:
            task = 'meniscus'
        self.task = task
        self.plane = plane
        self.root_dir = MRPATH
        self.train = train
        if self.train:
            self.folder_path = self.root_dir + 'train/{0}/'.format(plane)
            self.records = pd.read_csv(
                self.root_dir + 'train-{0}.csv'.format(task), header=None, names=['id', 'label'])
        else:
            transform = None
            self.folder_path = self.root_dir + 'valid/{0}/'.format(plane)
            self.records = pd.read_csv(
                self.root_dir + 'valid-{0}.csv'.format(task), header=None, names=['id', 'label'])

        self.records['id'] = self.records['id'].map(
            lambda i: '0' * (4 - len(str(i))) + str(i))
        self.paths = [self.folder_path + filename +
                      '.npy' for filename in self.records['id'].tolist()]
        self.labels = self.records['label'].tolist()

        self.transform = transform
        # if weights is None:
        #     pos = np.sum(self.labels)
        #     neg = len(self.labels) - pos
        #     self.weights = torch.FloatTensor([1, neg / pos])
        # else:
        #     self.weights = torch.FloatTensor(weights)
        neg_weight = np.mean(self.labels)
        self.weights = [neg_weight, 1 - neg_weight]

    def weighted_loss(self, prediction, target):
        weights_npy = np.array([self.weights[int(t[0])] for t in target.data])
        weights_tensor = torch.FloatTensor(weights_npy)
        loss = F.binary_cross_entropy_with_logits(prediction, target, weight=Variable(weights_tensor))
        return loss

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        array = np.load(self.paths[index])
        label = self.labels[index]
        label = torch.FloatTensor([label])

        if self.transform:
            array = self.transform(array)
        else:
            array = np.stack((array,)*3, axis=1)
            array = torch.FloatTensor(array)

        # if label.item() == 1:
        #     weight = np.array([self.weights[1]])
        #     weight = torch.FloatTensor(weight)
        # else:
        #     weight = np.array([self.weights[0]])
        #     weight = torch.FloatTensor(weight)

        return array, label

class Dataset(data.Dataset):
    def __init__(self, datadirs, diagnosis, use_gpu):
        super().__init__()
        self.use_gpu = use_gpu

        label_dict = {}
        self.paths = []

        for i, line in enumerate(open(PATH+'metadata.csv').readlines()):
            if i == 0:
                continue
            line = line.strip().split(',')
            path = line[10]
            label = line[2]
            label_dict[path] = int(int(label) > diagnosis)

        for dire in datadirs:
            for file in os.listdir(dire):
                self.paths.append(dire+'/'+file)

        self.labels = [label_dict[path[len(PATH)+6:]] for path in self.paths]

        neg_weight = np.mean(self.labels)
        self.weights = [neg_weight, 1 - neg_weight]

    def weighted_loss(self, prediction, target):
        weights_npy = np.array([self.weights[int(t[0])] for t in target.data])
        weights_tensor = torch.FloatTensor(weights_npy)
        if self.use_gpu:
            weights_tensor = weights_tensor.cuda()
        loss = F.binary_cross_entropy_with_logits(prediction, target, weight=Variable(weights_tensor))
        return loss

    def __getitem__(self, index):
        path = self.paths[index]
        with open(path, 'rb') as file_handler: # Must use 'rb' as the data is binary
            vol = pickle.load(file_handler).astype(np.int32)

        # crop middle
        pad = int((vol.shape[2] - INPUT_DIM)/2)
        vol = vol[:, pad:-pad, pad:-pad]
        
        # standardize
        vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * MAX_PIXEL_VAL

        # normalize
        vol = (vol - MEAN) / STDDEV
        
        # convert to RGB
        vol = np.stack((vol,)*3, axis=1)

        vol_tensor = torch.FloatTensor(vol)
        label_tensor = torch.FloatTensor([self.labels[index]])

        return vol_tensor, label_tensor

    def __len__(self):
        return len(self.paths)


#task/diagnosis: 0 = abnormality, 1 = acl/full tear (external), 2 = mcl  
def external_load_data(diagnosis, use_gpu=False):

    train_dirs = ['vol08', 'vol04', 'vol03', 'vol09', 'vol06', 'vol07']
    valid_dirs = ['vol10', 'vol05']
    test_dirs = ['vol01', 'vol02']
    
    train_dirs = [PATH + path for path in train_dirs]
    valid_dirs = [PATH + path for path in valid_dirs]
    test_dirs = [PATH + path for path in test_dirs]
    
    train_dataset = Dataset(train_dirs, diagnosis, use_gpu)
    valid_dataset = Dataset(valid_dirs, diagnosis, use_gpu)
    test_dataset = Dataset(test_dirs, diagnosis, use_gpu)

    train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=8, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=1, num_workers=8, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=False)

    return train_loader, valid_loader, test_loader

def mr_load_data(task):

    train_loaders = []
    valid_loaders = []

    for plane in ['sagittal', 'axial', 'coronal']:
        train_dataset = MRDataset(task, plane)
        valid_dataset = MRDataset(task, plane)

        train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=8, shuffle=True)
        valid_loader = data.DataLoader(valid_dataset, batch_size=1, num_workers=8, shuffle=False)

        train_loaders.append(train_loader)
        valid_loaders.append(valid_loader)
    
    return tuple(train_loaders), tuple(valid_loaders)