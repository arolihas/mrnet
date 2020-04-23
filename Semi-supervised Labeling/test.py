from loader import mr_load_data
from model import LabellingMRNet
import torch
import torch.nn as nn

loader, _ = mr_load_data()
mod = LabellingMRNet()

#pred = mod.forward(loader.dataset[1][0]).view(1, 3)
#label = loader.dataset[1][1]



label = torch.LongTensor([0, 0])
pred = torch.Tensor([[1., 0.0, 0.0], [1., 0.0, 0.0]]).view(2, 3)

cel = nn.CrossEntropyLoss()

print("loss:")
print(cel(pred, label))

"""
pred = torch.Tensor([1., .0, .0]).view(1, 3)
print(pred)
loss = loader.dataset.weighted_loss(pred, label)
print(loss)
"""
