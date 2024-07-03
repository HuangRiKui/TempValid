import pdb
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class LCWT(nn.Module):
    def __init__(self, rule_dim, args):
        super().__init__()
        self.rule_num = rule_dim-1
        self.W = nn.Parameter(torch.randn(self.rule_num,1))
        self.W0 = nn.Parameter(torch.randn(1))

        self.W.data.clamp_(0, 1)

    def forward(self, rel_data):
        pos_data = rel_data[:, 0, :]
        neg_data = rel_data[:, 1:, :]

        pos_score = self.score_function(pos_data).squeeze(-1)
        neg_score = self.score_function(neg_data).squeeze(-1)

        return pos_score, neg_score

    def score_function(self, data):
        # score = torch.matmul(data, self.W) + self.W0
        score = torch.ones_like(data)
        score[data == 0] = 0
        score = torch.matmul(score, self.W)

        return score

class LCFT(nn.Module):
    def __init__(self, rule_dim, args):
        super().__init__()
        self.rule_num = rule_dim-1
        self.W = nn.Parameter(torch.randn(self.rule_num,1))
        self.beta = args.beta

        self.W.data.clamp_(0,1)

    def forward(self, rel_data):
        # rel_data[rel_data==0] = 1e9

        pos_data = rel_data[:,0,:]
        neg_data = rel_data[:,1:,:]

        pos_score = self.score_function(pos_data).squeeze(-1)
        neg_score = self.score_function(neg_data).squeeze(-1)

        return pos_score, neg_score

    def score_function(self, data):
        # score = torch.matmul(data, self.W) + self.W0
        # score = F.relu(data - self.T0)
        score = torch.exp(-self.beta*data)
        mask = torch.ones_like(score)
        mask[data==0] = 0
        score = score * mask
        score = torch.matmul(score, self.W)+1e-9

        return score

class LTV(nn.Module):
    def __init__(self, rule_dim, conf_tensor, args):
        super().__init__()
        self.rule_num = rule_dim-1
        self.W = conf_tensor.reshape(-1,1)
        if args.cuda:
            self.W = self.W.cuda()
        self.beta = nn.Parameter(torch.randn(1, self.rule_num))
        # self.W0 = nn.Parameter(torch.randn(1, 1))

        self.W.data.clamp_(0,1)
        # self.W0.data.clamp_(0, 1)
        self.beta.data.clamp_(min=0)

    def forward(self, rel_data):
        pos_data = rel_data[:,0,:]
        neg_data = rel_data[:,1:,:]

        pos_score = self.score_function(pos_data).squeeze(-1)
        neg_score = self.score_function(neg_data).squeeze(-1)

        return pos_score, neg_score

    def score_function(self, data):
        score = torch.exp(-self.beta*data)
        # score = torch.sigmoid(self.beta1 * data)
        mask = torch.ones_like(score)
        mask[data==0] = 0
        score = score * mask
        score = torch.matmul(score, self.W) + 1e-9

        return score

class TempValid(nn.Module):
    def __init__(self, rule_dim, args):
        super().__init__()
        self.rule_num = rule_dim-1
        self.W = nn.Parameter(torch.randn(self.rule_num,1))
        self.beta = nn.Parameter(torch.randn(1, self.rule_num))
        # self.W0 = nn.Parameter(torch.randn(1, 1))

        self.W.data.clamp_(0,1)
        # self.W0.data.clamp_(0, 1)
        self.beta.data.clamp_(min=0)

    def forward(self, rel_data):
        pos_data = rel_data[:,0,:]
        neg_data = rel_data[:,1:,:]

        pos_score = self.score_function(pos_data).squeeze(-1)
        neg_score = self.score_function(neg_data).squeeze(-1)

        return pos_score, neg_score

    def score_function(self, data):
        score = torch.exp(-self.beta*data)
        # score = torch.sigmoid(self.beta1 * data)
        mask = torch.ones_like(score)
        mask[data==0] = 0
        score = score * mask
        score = torch.matmul(score, self.W) + 1e-9

        return score

class Noisy_OR(nn.Module):
    def __init__(self, rule_dim, args):
        super().__init__()
        self.rule_num = rule_dim-1
        self.W = nn.Parameter(torch.randn(1, self.rule_num))
        self.beta = nn.Parameter(torch.randn(1, self.rule_num))

        self.W.data.clamp_(0,1)
        self.beta.data.clamp_(min=0)

    def forward(self, rel_data):
        pos_data = rel_data[:,0,:]
        neg_data = rel_data[:,1:,:]

        pos_score = self.score_function(pos_data).squeeze(-1)
        neg_score = self.score_function(neg_data).squeeze(-1)

        return pos_score, neg_score

    def score_function(self, data):
        score = torch.exp(-self.beta*data)
        mask = torch.ones_like(score)
        mask[data==0] = 0
        score = score * mask
        score = 1 - torch.prod((1-score * self.W), dim=-1) + 1e-9

        return score