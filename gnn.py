import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import time
from multiprocessing import Pool
from layers import GAT_gate
from dataset import N_ATOM_FEATURES, N_FEATURES

#N_atom_features = 28

class gnn(torch.nn.Module):
    def __init__(self, args):
        super(gnn, self).__init__()

        fnm = __class__.__name__ + '.' + whoami()

        n_graph_layer = args.n_graph_layer
        d_graph_layer = args.d_graph_layer

        n_FC_layer = args.n_FC_layer
        d_FC_layer = args.d_FC_layer

        self.dropout_rate = args.dropout_rate 

        # n_graph_layer:4, d_graph_layer:140, self.dropout_rate:0.3
        print('{}: n_graph_layer:{}, d_graph_layer:{}, self.dropout_rate:{}'.format(fnm, n_graph_layer, d_graph_layer, self.dropout_rate))

        self.layers1 = [d_graph_layer for i in range(n_graph_layer+1)]
        print('{}: self.layers1: {}'.format(fnm, self.layers1))

        #self.layers2 = [64 * 25 * 35]
        #self.layers2 = [32 * 25 * 35]
        n_fc_input_features = 32 * 25 * 35 # 28000
        self.layers2 = [n_fc_input_features, 875, 35, 2]

        print('{}: self.layers2: {}'.format(fnm, self.layers2))

        self.gconv1 = nn.ModuleList([GAT_gate(self.layers1[i], self.layers1[i+1]) for i in range(len(self.layers1)-1)]) 

        self.build_cnn()

        #self.FC = nn.ModuleList([nn.Linear(self.layers1[-1], d_FC_layer) if i==0 else
        #                         nn.Linear(d_FC_layer, 1) if i==n_FC_layer-1 else
        #                         nn.Linear(d_FC_layer, d_FC_layer) for i in range(n_FC_layer)])
        #self.FC = nn.ModuleList([nn.Linear(self.layers2[-1], d_FC_layer) if i == 0 else
        #                         nn.Linear(d_FC_layer, 1) if i == n_FC_layer - 1 else
        #                         nn.Linear(d_FC_layer, d_FC_layer) for i in range(n_FC_layer)])

        #self.FC = nn.ModuleList([nn.Linear(n_fc_input_features          , int(n_fc_input_features  /   4))
        #                        ,nn.Linear(int(n_fc_input_features /   4), int(n_fc_input_features /  64))
        #                        ,nn.Linear(int(n_fc_input_features /  64), int(n_fc_input_features / 320))
        #                        ,nn.Linear(int(n_fc_input_features / 320), int(n_fc_input_features / 320))
        #                        ,nn.Linear(int(n_fc_input_features / 320), int(n_fc_input_features /28000))
        #                        ])
        #self.FC = nn.ModuleList([nn.Linear(n_fc_input_features            , int(n_fc_input_features /   16))
        #                        ,nn.Linear(int(n_fc_input_features /   16), int(n_fc_input_features /  400))
        #                        ,nn.Linear(int(n_fc_input_features /  400), int(n_fc_input_features / 1400))
        #                        ,nn.Linear(int(n_fc_input_features / 1400), int(n_fc_input_features /14000))
        #                        ])
        self.FC = nn.ModuleList([nn.Linear(self.layers2[i], self.layers2[i + 1]) for i in range(len(self.layers2) - 1)])

        self.mu = nn.Parameter(torch.Tensor([args.initial_mu]).float())
        print('{}: args.initial_mu:{} --> self.mu:{}'.format(fnm, args.initial_mu, self.mu))

        self.dev = nn.Parameter(torch.Tensor([args.initial_dev]).float())
        print('{}: args.initial_dev:{} --> self.dev:{}'.format(fnm, args.initial_dev, self.dev))

        #self.embede = nn.Linear(2*N_atom_features, d_graph_layer, bias = False)
        self.embede = nn.Linear(N_FEATURES, d_graph_layer, bias = False)
        print('{}: self.embede: (2*N_ATOM_FEATURES({}), d_graph_layer({}): --> {}'.format(fnm, N_ATOM_FEATURES, d_graph_layer, self.embede))

        self.emb_info_printed = False
        self.cnn_info_printed = False
        self.gnn_info_printed = False
        self.fnn_info_printed = False
        pass
        

    def embede_graph(self, data):
        fnm = __class__.__name__ + '.' + whoami()

        c_hs, c_adjs1, c_adjs2, c_valid = data

        if not self.emb_info_printed:
            print('{}:#1: c_hs:shape:{}, c_adjs1.shape:{}, c_adjs2.shape:{}, c_valid.shape:{}'.format(
                   fnm, c_hs.shape, c_adjs1.shape, c_adjs2.shape, c_valid.shape))
            pass

        c_hs = self.embede(c_hs)

        if not self.emb_info_printed:
            print('{}:#2: embedding --> c_hs:shape:{}'.format(fnm, c_hs.shape))
            pass

        hs_size = c_hs.size()

        c_adjs2 = torch.exp(-torch.pow(c_adjs2-self.mu.expand_as(c_adjs2), 2)/self.dev) + c_adjs1

        regularization = torch.empty(len(self.gconv1), device=c_hs.device)

        if not self.emb_info_printed:
            print('{}:#3: c_adjs2.shape:{}, regularization(not used):shape:{}, len(self.gconv1):{}'.format(
                  fnm, c_adjs2.shape, regularization.shape, len(self.gconv1)))
            pass

        for k in range(len(self.gconv1)):
            c_hs1 = self.gconv1[k](c_hs, c_adjs1)
            c_hs2 = self.gconv1[k](c_hs, c_adjs2)
            c_hs = c_hs2-c_hs1
            c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)
            pass

        if not self.emb_info_printed:
            print('{}:#4: c_hs.shape:{}, c_hs1.shape:{}, c_hs2.shape:{}'.format(
                  fnm, c_hs.shape, c_hs1.shape, c_hs2.shape))
            pass

        #c_hs = c_hs*c_valid.unsqueeze(-1).repeat(1, 1, c_hs.size(-1))

        #if not self.emb_info_printed:
        #    print('{}:#5 --> unsqueezed, c_hs.shape:{}'.format(fnm, c_hs.shape))
        #    pass

        #c_hs = c_hs.sum(1)

        #if not self.emb_info_printed:
        #    print('{}:#6 --> summed, c_hs.shape:{}\n{}'.format(fnm, c_hs.shape, c_hs))
        #    print('+' * 10)
        #    pass

        self.emb_info_printed = True

        return c_hs

    def build_cnn(self):
        # in_channels   =  1
        # out_channels  = 16
        # kernel_size   =  5
        # stride        =  1
        # padding       =  2
        #self.cnn1_conv = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.cnn1_conv = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.cnn1_acti = nn.ReLU()
        self.cnn1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cnn1_batch_norm = nn.BatchNorm2d(16)
        self.cnn1_dropout = nn.Dropout()

        # in_channels   = 16
        # out_channels  = 64
        # kernel_size   =  5
        # stride        =  1
        # padding       =  2(default)
        #self.cnn2_conv = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.cnn2_conv = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.cnn2_acti = nn.ReLU()
        self.cnn2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cnn2_batch_norm = nn.BatchNorm2d(32)
        self.cnn2_dropout = nn.Dropout()

        #self.cnn_dropout = nn.Dropout()
        self.cnn_flatten = nn.Flatten()
        pass

    def cnn(self, c_hs):
        fnm = __class__.__name__ + '.' + whoami()

        if not self.cnn_info_printed:
            print('{}:#0:c_hs.shape:{}'.format(fnm, c_hs.shape))
            pass

        c_hs = c_hs.unsqueeze(1)

        if not self.cnn_info_printed:
            print('{}:#1:c_hs.shape:{} after unsqueeze(0)'.format(fnm, c_hs.shape))
            pass

        c_hs = self.cnn1_conv(c_hs)

        if not self.cnn_info_printed:
            print('{}:#2:c_hs.shape:{} after 1st conv.'.format(fnm, c_hs.shape))
            pass

        c_hs = self.cnn1_acti(c_hs)
        c_hs = self.cnn1_pool(c_hs)
        c_hs = self.cnn1_batch_norm(c_hs)
        c_hs = self.cnn1_dropout(c_hs)

        if not self.cnn_info_printed:
            print('{}:#3:c_hs.shape:{} after 1st pool.'.format(fnm, c_hs.shape))
            pass

        c_hs = self.cnn2_conv(c_hs)

        if not self.cnn_info_printed:
            print('{}:#4:c_hs.shape:{} after 2nd conv.'.format(fnm, c_hs.shape))
            pass

        c_hs = self.cnn2_acti(c_hs)
        c_hs = self.cnn2_pool(c_hs)
        c_hs = self.cnn2_batch_norm(c_hs)
        c_hs = self.cnn2_dropout(c_hs)

        if not self.cnn_info_printed:
            print('{}:#5:c_hs.shape:{} after 2nd pool.'.format(fnm, c_hs.shape))
            pass

        #c_hs = self.cnn_dropout(c_hs)

        #c_hs = c_hs.reshape(c_hs.size(0), -1)
        #c_hs = c_hs.reshape(c_hs.size(0), -1, self.layers1[-1])
        c_hs = self.cnn_flatten(c_hs)

        if not self.cnn_info_printed:
            print('{}:#6:c_hs.shape:{} after flattened'.format(fnm, c_hs.shape))
            pass

        self.cnn_info_printed = True

        return c_hs


    def fully_connected(self, c_hs):
        fnm = __class__.__name__ + '.' + whoami()

        regularization = torch.empty(len(self.FC)*1-1, device=c_hs.device)

        for k in range(len(self.FC)):
            #c_hs = self.FC[k](c_hs)
            if k < len(self.FC) - 1:
                if not self.fnn_info_printed:
                    print('{}:#{}:[BF] len(self.FC):{}, c_hs.shape:{}'.format(fnm, k, len(self.FC), c_hs.shape))
                    pass
                c_hs = self.FC[k](c_hs)
                if not self.fnn_info_printed:
                    print('{}:#{}:[AF] len(self.FC):{}, c_hs.shape:{}'.format(fnm, k, len(self.FC), c_hs.shape))
                    pass
                c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)
                c_hs = F.relu(c_hs)
                pass
            else:
                if not self.fnn_info_printed:
                    print('{}:#{}:[BF] len(self.FC):{}, c_hs.shape:{}'.format(fnm, k, len(self.FC), c_hs.shape))
                    pass
                c_hs = self.FC[k](c_hs)
                #c_hs = c_hs.reshape(c_hs.size(0), 1, -1)
                if not self.fnn_info_printed:
                    print('{}:#{}:[AF] len(self.FC):{}, c_hs.shape:{}'.format(fnm, k, len(self.FC), c_hs.shape))
                    pass
                pass

        #
        # FC의 마지막 activation function을 softmax로 변경
        #  --> softmax는 뒤에 호출할 torch.nn.CrossEntropyLoss() 에 포함되어 있다고 가정함

        #
        #if not self.fnn_info_printed:
        #    print('{}:[BF] sigmoid, c_hs.shape:{}'.format(fnm, c_hs.shape))
        #    pass

        #c_hs = torch.sigmoid(c_hs)

        #if not self.fnn_info_printed:
        #    print('{}:[AF] sigmoid, c_hs.shape:{}'.format(fnm, c_hs.shape))
        #    pass

        if not self.fnn_info_printed:
            print('{}:FINAL: c_hs.shape:{}'.format(fnm, c_hs.shape))
            pass

        self.fnn_info_printed = True

        return c_hs

    def train_model(self, data):
        fnm = __class__.__name__ + '.' + whoami()

        #embede a graph to a vector
        c_hs = self.embede_graph(data)

        if not self.gnn_info_printed:
            print('{}:#1 self.embede_graph() --> c_hs.shape:{}'.format(fnm, c_hs.shape))
            pass

        c_hs = self.cnn(c_hs)

        #fully connected NN
        c_hs = self.fully_connected(c_hs)

        if not self.gnn_info_printed:
            print('{}:#2 self.fully_connected() --> c_hs.shape:{}'.format(fnm, c_hs.shape))
            pass

        #c_hs = c_hs.view(-1) 

        #if not self.gnn_info_printed:
        #    print('{}:#3 c_hs.view(-1):flatten --> c_hs.shape:{}'.format(fnm, c_hs.shape))
        #    print('#' * 80)
        #    pass

        self.gnn_info_printed = True

        #note that if you don't use concrete dropout, regularization 1-2 is zero

        return c_hs
    

    def test_model(self,data1 ):
        c_hs = self.embede_graph(data1)
        c_hs = self.fully_connected(c_hs)
        c_hs = c_hs.view(-1)
        return c_hs
