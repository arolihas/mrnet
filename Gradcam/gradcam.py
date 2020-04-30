# imports and setup
import torch
from torch.autograd import Variable
from torch.autograd import Function as TorchFunc
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import random
import os

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib
from cs7643.image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD
from PIL import Image
import captum
from model import MRNet


# Preprocess


def preprocess(img, size=224):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img, should_rescale=True):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
        T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled

def blur_image(X, sigma=1):
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X

def visualize_attr_maps(attributions, titles, attr_preprocess=lambda attr: attr.permute(1, 2, 0).detach().numpy(),cmap='viridis', alpha=0.7):
    '''
    A helper function to visualize captum attributions for a list of captum attribution algorithms.

    attributions(A list of torch tensors): Each element in the attributions list corresponds to an
                      attribution algorithm, such an Saliency, Integrated Gradient, Perturbation, etc.
                      Each row in the attribution tensor contains
    titles(A list of strings): A list of strings, names of the attribution algorithms corresponding to each element in
                      the `attributions` list. len(attributions) == len(titles)
    '''
    N = attributions[0].shape[0]
    plt.figure()
    for i in range(N):
        axs = plt.subplot(len(attributions) + 1, N + 1, i+1)
        plt.imshow(X[i])
        plt.axis('off')
        plt.title(class_names[y[i]])

    plt.subplot(len(attributions) + 1, N + 1 , N + 1)
    plt.text(0.0, 0.5, 'Original Image', fontsize=14)
    plt.axis('off')
    for j in range(len(attributions)):
        for i in range(N):
            plt.subplot(len(attributions) + 1, N + 1 , (N + 1) * (j + 1) + i + 1)
            attr = np.array(attr_preprocess(attributions[j][i]))
            attr = (attr - np.mean(attr)) / np.std(attr).clip(1e-20)
            attr = attr * 0.2 + 0.5
            attr = attr.clip(0.0, 1.0)
            plt.imshow(attr, cmap=cmap, alpha=alpha)
            plt.axis('off')
        plt.subplot(len(attributions) + 1, N + 1 , (N + 1) * (j + 1) + N + 1)
        plt.text(0.0, 0.5, titles[j], fontsize=14)
        plt.axis('off')

    plt.gcf().set_size_inches(20, 13)
    plt.show()

def compute_attributions(algo, inputs, **kwargs):
    '''
    A common function for computing captum attributions
    '''
    return algo.attribute(inputs, **kwargs)



# load model
"""
gc_model = torchvision.models.squeezenet1_1(pretrained=True)
for param in gc_model.parameters():
    param.requires_grad = True

"""
gc_model = MRNet(useMultiHead=True)
weights = torch.load("weights", map_location=torch.device('cpu'))
gc_model.load_state_dict(weights)

#print(gc_model)

# Grad cam code

conv_module = gc_model.model.features[10]
#conv_module = gc_model.features[12]

gradient_value = None # Stores gradient of the module you chose above during a backwards pass.
activation_value = None # Stores the activation of the module you chose above during a forwards pass.

def gradient_hook(a,b,gradient):
    global gradient_value
    gradient_value = gradient[0]

def activation_hook(a,b,activation):
    global activation_value
    activation_value = activation

conv_module.register_forward_hook(activation_hook)
conv_module.register_backward_hook(gradient_hook)


def grad_cam(X_tensor, y_tensor):
    ##############################################################################
    # TODO: Implement GradCam as described in paper.                             #
    ##############################################################################
    res = gc_model(X_tensor)

    print(X_tensor.shape)
    print(res.shape)

    loss = F.binary_cross_entropy_with_logits(res, y_tensor)
    #loss = res.gather(1, y_tensor.view(-1, 1)).squeeze()
    #loss = torch.sum(loss)
    loss.backward()

    N, C, H, W = gradient_value.shape
    neuron_importance = torch.sum(gradient_value, (2,3)) / (H*W)
    neuron_importance = neuron_importance.view(N, C, 1, 1)

    cam = F.relu(torch.sum(neuron_importance * activation_value, 1)).detach().numpy()


    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################

    # Rescale GradCam output to fit image.
    cam_scaled = []
    for i in range(cam.shape[0]):
        cam_scaled.append(np.array(Image.fromarray(cam[i]).resize(X_tensor[i,0,:,:].shape)))
    cam = np.array(cam_scaled)
    cam -= np.min(cam)
    cam /= np.max(cam)
    return cam

# Add Image and label

inputs = ["0139", "0140", "0141", "0142", "0267"]
Y = [1, 1, 1, 1, 0]
#orients = ['_A', '_C', '_S']
orient = ""

for _i in range(len(inputs)):

    num = inputs[_i]
    y = Y[_i]
    _X = np.load(num + orient + '.npy')





    X = np.zeros((_X.shape[0], 3,256, 256))
    X[:,0, :, :] = _X
    X[:,1,:, :] = _X
    X[:, 2,:, :] = _X

    XX = np.zeros((_X.shape[0], 256, 256, 3))
    XX[:,:, :, 0] = _X
    XX[:,:, :, 1] = _X
    XX[:,:, :, 2] = _X

    X = X.astype(np.uint8)
    class_names = {0: "No Abnormalityr", 1 : "Abnormality"}

    # Run gradcam

    X_tensor = torch.Tensor(X) #torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0).requires_grad_(True)
    y_tensor = torch.Tensor([y]).view(1,1)
    gradcam_result = grad_cam(X_tensor, y_tensor)



    plt.figure(figsize=(24, 24))

    for i in range(X.shape[0]):
        gradcam_val = gradcam_result[i]
        img = XX[i] + (matplotlib.cm.jet(gradcam_val)[:,:,:3]*255)
        img = img / np.max(img)

        folder_path = num + orient + "_Folder"
        filename = str(i) + "_file.jpg"

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file = os.path.join(folder_path, filename)
        plt.imsave(file, img)

"""
for i in range(5):
    gradcam_val = gradcam_result[i + s]
    img = XX[i + s] + (matplotlib.cm.jet(gradcam_val)[:,:,:3]*255)
    img = img / np.max(img)
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.title(class_names[y[i]])
    plt.axis('off')
plt.gcf().tight_layout()

plt.show()
"""
