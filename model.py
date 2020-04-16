import torch
import torch.nn as nn
import math

from torchvision import models

class MRNet(nn.Module):
    def __init__(self, useMultiHead = True, num_sublayers = 2, num_heads = 8,
                 hidden_dim = 256, dim_feedforward = 512, dim_kq = None, dim_v = None):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.classifier = nn.Linear(256, 1)
        self.useMultiHead = useMultiHead
        self.gap = nn.AdaptiveAvgPool2d(1)
        if useMultiHead:
            self.attention = Attention(num_sublayers, num_heads, hidden_dim, dim_feedforward, dim_kq, dim_v)
            #self.classifier = nn.Linear(256 * 30, 1)

    def forward(self, x):
        print(x.shape)
        x = torch.squeeze(x, dim=0) # only batch size 1 supported
        x = self.model.features(x)
        print(x.shape)
        print("Next")
        x = self.gap(x).view(x.size(0), -1)
        if self.useMultiHead:
            x = x.view(x.size(0), 1, -1)
            x = self.attention(x).view(x.size(0), -1)

        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)

        return x

class Attention(nn.Module):
    def __init__(self, num_sublayers, num_heads, hidden_dim, dim_feedforward, dim_kq=None, dim_v=None):
        super().__init__()
        self.num_sublayers = num_sublayers
        self.layers = [MultiheadedAttentionSubLayer
                       (num_heads, hidden_dim, dim_feedforward, dim_kq, dim_v) for i in range(num_sublayers)]

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

    def feedforward_layer(self, inputs):
        r1 = self.FFN1(inputs)
        r2 = self.ReLU(r1)
        r3 = self.FFN2(r2)

        return self.norm_ff(r3 + inputs)

    def forward(self, x):
        x = self.multi_head_attention(x)
        x = self.feedforward_layer(x)

        return x


"""
class MultiheadedAttentionSubLayer(nn.Module):
    def __init__(self, num_heads, hidden_dim, dim_feedforward, dim_kq, dim_v):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward

        self.dim_k = dim_kq
        self.dim_v = dim_v
        self.dim_q = dim_kq

        self.k = [nn.Linear(self.hidden_dim, self.dim_k) for i in range(num_heads)]
        self.v = [nn.Linear(self.hidden_dim, self.dim_v) for i in range(num_heads)]
        self.q = [nn.Linear(self.hidden_dim, self.dim_q) for i in range(num_heads)]

        self.FFN1 =  nn.Linear(self.hidden_dim, self.dim_feedforward)
        self.FFN1 =  nn.Linear(self.dim_feedforward, self.hidden_dim)

        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)

        self.ReLU = nn.functional.relu
        self.norm_mh = nn.LayerNorm(self.hidden_dim)
        self.norm_ff = nn.LayerNorm(self.hidden_dim)
        self.softmax = nn.Softmax(dim=2)


    def single_head_attention(self, inputs, headNum):
        sqr_dk = math.sqrt(self.dim_k)

        K = self.k[headNum](inputs)      # N x T x dk
        Q = self.q[headNum](inputs)      # N x T x dk
        V = self.v[headNum](inputs)      # N x T x dv

        K_t = torch.transpose(K, 1, 2)

        scores = torch.matmul(Q, K_t)/sqr_dk  # N x T x T
        smvalues = self.softmax(scores)       # N x T x T
        return torch.matmul(smvalues, V)      # N x T x dv

    def multi_head_attention(self, inputs):
        sing_atten = []
        for i in range(self.num_heads):
            sing_atten.append(self.single_head_attention(inputs, i))

        A = self.attention_head_projection(torch.cat(sing_atten, dim=2)) # N x T x H

        return self.norm_mh(A + inputs)

    def feedforward_layer(self, inputs):
        r1 = self.FFN1(inputs)
        r2 = self.ReLU(r1)
        r3 = self.FFN2(r2)

        return self.norm_ff(r3 + inputs)


    def forward(self, x):

        mha = self.multi_head_attention(x)
        ff = self.feedforward_layer(mha)

        return ff
"""
