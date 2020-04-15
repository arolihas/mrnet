import numpy as np
from tqdm import tqdm_notebook
from PIL import Image
from loader import Dataset

import os
import shutil
import sys
sys.path.append('../../mrnet')

import torch
import torch.nn.functional as F
import cv2
from torchvision import models, transforms
import model

task = 'acl'
plane = 'sagittal'
prefix = 'v3'

model_name = [name  for name in os.listdir('../models/') 
              if (task in name) and 
                 (plane in name) and 
                 (prefix in name)][0]

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

mrnet = torch.load(f'../models/{model_name}')
mrnet = mrnet.to(device)

_ = mrnet.eval()

dataset = Dataset('../data/', 
                    task, 
                    plane, 
                    transform=None, 
                    train=False)
loader = torch.utils.data.DataLoader(dataset, 
                                     batch_size=1, 
                                     shuffle=False, 
                                     num_workers=0, 
                                     drop_last=False)

def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    slice_cams = []
    for s in range(bz):
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature_conv[s].reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            slice_cams.append(cv2.resize(cam_img, size_upsample))
    return slice_cams

patients = []

for i, (image, label, _) in tqdm_notebook(enumerate(loader), total=len(loader)):
    patient_data = {}
    patient_data['mri'] = image
    patient_data['label'] = label[0][0][1].item()
    patient_data['id'] = '0' * (4 - len(str(i))) + str(i)
    patients.append(patient_data)

acl = list(filter(lambda d: d['label'] == 1, patients))

def create_patiens_cam(case, plane):
    patient_id = case['id']
    mri = case['mri']

    folder_path = f'./CAMS/{plane}/{patient_id}/'
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)
    os.makedirs(folder_path + 'slices/')
    os.makedirs(folder_path + 'cams/')
    
    params = list(mrnet.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
    
    num_slices = mri.shape[1]
    global feature_blobs
    feature_blobs = []
    mri = mri.to(device)
    logit = mrnet(mri)
    size_upsample = (256, 256)
    feature_conv = feature_blobs[0]
    
    h_x = F.softmax(logit, dim=1).data.squeeze(0)
    probs, idx = h_x.sort(0, True)
    probs = probs.cpu().numpy()
    idx = idx.cpu().numpy()
    slice_cams = returnCAM(feature_blobs[-1], weight_softmax, idx[:1])
    
    for s in tqdm_notebook(range(num_slices), leave=False):
        slice_pil = (transforms
                     .ToPILImage()(mri.cpu()[0][s] / 255))
        slice_pil.save(folder_path + f'slices/{s}.png', 
                       dpi=(300, 300))
         
        img = mri[0][s].cpu().numpy()
        img = img.transpose(1, 2, 0)
        heatmap = (cv2
                    .cvtColor(cv2.applyColorMap(
                        cv2.resize(slice_cams[s], (256, 256)),
                        cv2.COLORMAP_JET), 
                               cv2.COLOR_BGR2RGB)
                  )
        result = heatmap * 0.3 + img * 0.5  
        
        pil_img_cam = Image.fromarray(np.uint8(result))
        pil_img_cam.save(folder_path + f'cams/{s}.png', dpi=(300, 300))