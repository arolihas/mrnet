import torch
import torch.nn as nn
import math

from torchvision import models

class MRNet(nn.Module):
    def __init__(self, useMultiHead = True, num_sublayers = 2, num_heads = 8,
                 hidden_dim = 256, dim_feedforward = 512, dim_kq = None, dim_v = None, max_layers=110):
        super().__init__()
        self.useMultiHead = useMultiHead
        self.model = models.alexnet(pretrained=True)
        self.classifier = nn.Linear(256, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        if useMultiHead:
            print("Using Multiheaded Attention")
            self.posEmbedLayer = nn.Embedding(max_layers, hidden_dim)
            self.attention = Attention(num_sublayers, num_heads, hidden_dim, dim_feedforward, dim_kq, dim_v).cuda()
        else:
            print("Not using Multiheaded Attention")

    def forward(self, x):
        x = torch.squeeze(x, dim=0) # only batch size 1 supported
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), -1)

        if self.useMultiHead:
            # made this cuda-only I guess?
            positions = torch.tensor(range(x.size(0))).cuda()
            posEmbeddings = self.posEmbedLayer(positions).view(x.size(0), 1, -1)

            x = x.view(x.size(0), 1, -1)
            x = x + posEmbeddings

            x = self.attention(x)
            x = torch.sum(x, 0, keepdim=True)[0]/x.size(0)

        else:
            x = torch.max(x, 0, keepdim=True)[0]

        x = self.classifier(x)
        return x

class Attention(nn.Module):
    def __init__(self, num_sublayers, num_heads, hidden_dim, dim_feedforward, dim_kq=None, dim_v=None):
        super().__init__()
        self.num_sublayers = num_sublayers
        self.layers = [MultiheadedAttentionSubLayer
                       (num_heads, hidden_dim, dim_feedforward, dim_kq, dim_v).cuda() for i in range(num_sublayers)]

    def forward(self, x):
        for i in range(self.num_sublayers):
            x = self.layers[i](x)

        return x


class MultiheadedAttentionSubLayer(nn.Module):
    def __init__(self, num_heads, hidden_dim, dim_feedforward, dim_kq = None, dim_v=None):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, kdim=dim_kq, vdim=dim_v) # Input : L, N, E

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward

        self.FFN1 =  nn.Linear(self.hidden_dim, self.dim_feedforward)
        self.FFN2 =  nn.Linear(self.dim_feedforward, self.hidden_dim)

        self.ReLU = nn.functional.relu
        self.norm_mh = nn.LayerNorm(self.hidden_dim)
        self.norm_ff = nn.LayerNorm(self.hidden_dim)


    def multi_head_attention(self, x):
        A, attn_output_weights = self.multihead_attn(x, x, x)
        return self.norm_mh(A + x)

    def feedforward_layer(self, x):
        A = self.FFN1(x)
        A = self.ReLU(A)
        A = self.FFN2(A)

        return self.norm_ff(A + x)

    def forward(self, x):
        x = self.multi_head_attention(x)
        x = self.feedforward_layer(x)

        return x
