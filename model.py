#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GatedAttentionLayerV(nn.Module):
    '''
    $\text{tanh}\left(\boldsymbol{W}_{v}^\top \boldsymbol{h}_{i,j} + \boldsymbol{b}_v \right)$ in Equation (2)
    '''
    def __init__(self, dim=512):
        super(GatedAttentionLayerV, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1)


    def forward(self, features, W_V, b_V):
        out = F.linear(features, W_V, b_V)
        out_tanh = torch.tanh(out)

        return out_tanh


class GatedAttentionLayerU(nn.Module):
    '''
    $\text{sigm}\left(\boldsymbol{W}_{u}^\top \boldsymbol{h}_{i,j} + \boldsymbol{b}_u \right)$ in Equation (2)
    '''
    def __init__(self, dim=512):
        super(GatedAttentionLayerU, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1)


    def forward(self, features, W_U, b_U):
        out = F.linear(features, W_U, b_U)
        out_sigmoid = torch.sigmoid(out)

        return out_sigmoid


class GatedAttention(nn.Module):
    def __init__(self, args):
        super(GatedAttention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1
        self.args = args
        self.nr_fea = self.args.nr_fea

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.Dropout(),
            nn.ReLU(),
        )
        self.att_layer_V = GatedAttentionLayerV(self.L)
        self.att_layer_U = GatedAttentionLayerU(self.L)
        self.linear_V = nn.Linear(self.L * self.K, self.args.nr_class)
        self.linear_U = nn.Linear(self.L * self.K, self.args.nr_class)
        self.attention_weights = nn.Sequential(
            nn.Linear(self.args.nr_class, self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.K),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, self.args.nr_class),
            nn.Sigmoid()
        )


    def forward(self, X, args):
        X = X.squeeze(0)
        H = self.feature_extractor_part1(X)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)
        A_V = self.att_layer_V(H, self.linear_V.weight, self.linear_V.bias)
        A_U = self.att_layer_U(H, self.linear_V.weight, self.linear_V.bias)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        A = torch.sigmoid(A)
        M = torch.mm(A, H) / torch.sum(A)   # Equation (3)
        Y_prob = self.classifier(M)

        return Y_prob, A


    def full_loss(self, A, prediction, target, args):
        '''
        Equation (8)
        '''
        Y_candiate = torch.zeros(target.shape).to(device)
        Y_candiate[target > 0] = 1
        prediction_mask = prediction * Y_candiate
        new_prediction = prediction_mask / prediction_mask.sum(dim=1).repeat(prediction_mask.size(1), 1).transpose(0, 1)
        entropy_Y = - target * torch.log(prediction)
        entropy_A = - A * torch.log(A)
        loss = (torch.sum(entropy_Y)) / entropy_Y.size(0) + args.w_entropy_A * (torch.sum(entropy_A)) / entropy_A.size(0)

        return new_prediction, loss


    def calculate_objective(self, X, Y, args):
        '''
        calculate the full loss, weighted partial labels, and attention scores
        '''
        Y = Y.reshape(-1)
        Y_prob, A = self.forward(X, args)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        Y_prob = F.softmax(Y_prob, dim=1)
        new_prob, loss = self.full_loss(A, Y_prob, Y, args)

        return loss, new_prob, A


    def evaluate_objective(self, X, args):
        '''
        model testing
        '''
        Y_prob, _  = self.forward(X, args)
        Y_prob = F.softmax(Y_prob, dim=1)

        return Y_prob