# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import time

class GAT_gate(torch.nn.Module):
    def __init__(self, n_in_feature, n_out_feature):
        super(GAT_gate, self).__init__()

        self.W = nn.Linear(n_in_feature, n_out_feature)
        #self.A = nn.Parameter(torch.Tensor(n_out_feature, n_out_feature))
        self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
        self.gate = nn.Linear(n_out_feature*2, 1)
        self.leakyrelu = nn.LeakyReLU(0.2) # not used...

        self.print_forward_info = False
        pass

    def forward(self, x, adj):
        fnm = __class__.__name__ + '.' + whoami()

        h = self.W(x)
        if not self.print_forward_info:
            print('{}: x:{}, W:{} --> h:size:{}\n{}'.format(fnm, x, self.W, h.size(), h))
            pass

        batch_size = h.size()[0]
        N = h.size()[1]

        if not self.print_forward_info:
            print('{}: batch_size:{}, N:{}'.format(fnm, batch_size, N))
            pass

        #e = torch.einsum('ijl,ikl->ijk', (torch.matmul(h,self.A), h))
        einsum_str = 'ijl,ikl->ijk'
        mm = torch.matmul(h, self.A)

        #e = torch.einsum(einsum_str, (torch.matmul(h,self.A), h))
        e = torch.einsum(einsum_str, (mm, h))

        if not self.print_forward_info:
            print('{}:#1: einsum({}, (toch.matmul({}, {}):{}, h:{}) --> {}'.format(fnm, einsum_str, h, self.A, mm, h, e))
            pass

        e = e + e.permute((0,2,1))

        if not self.print_forward_info:
            print('{}:#2: e:{}'.format(fnm, e))
            pass

        zero_vec = -9e15*torch.ones_like(e)

        if not self.print_forward_info:
            print('{}:zero_vec:{}'.format(fnm, zero_vec))
            pass

        attention = torch.where(adj > 0, e, zero_vec)

        if not self.print_forward_info:
            print('{}:#1: attention:{}'.format(fnm, attention))
            pass

        attention = F.softmax(attention, dim=1)

        if not self.print_forward_info:
            print('{}:#2: attention:{}'.format(fnm, attention))
            pass
        #attention = F.dropout(attention, self.dropout, training=self.training)
        #h_prime = torch.matmul(attention, h)
        attention = attention*adj

        if not self.print_forward_info:
            print('{}:#3: adj:{} --> attention:{}'.format(fnm, adj, attention))
            pass

        #h_prime = F.relu(torch.einsum('aij,ajk->aik',(attention, h)))
        einsum_str = 'aij,ajk->aik'
        einsum = torch.einsum(einsum_str, (attention, h))
        h_prime = F.relu(einsum)

        if not self.print_forward_info:
            print('{}:einsum({},  (attention:{}, h:{})) --> {} --> h_prime:{}'.format(fnm, einsum_str, attention, h, einsum, h_prime))
            pass

        coeff = torch.sigmoid(self.gate(torch.cat([x,h_prime], -1))).repeat(1,1,x.size(-1))

        if not self.print_forward_info:
            print('{}: coeff:{}'.format(fnm, coeff))
            pass

        retval = coeff * x + (1 - coeff) * h_prime

        if not self.print_forward_info:
            print('{}: retval:{}'.format(fnm, retval))
            pass

        self.print_forward_info = True

        return retval

    pass
