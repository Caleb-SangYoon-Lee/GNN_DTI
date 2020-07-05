# -*- coding: utf-8 -*-

import os, sys
import datetime

import pickle
import time
import numpy as np
import time
import argparse
import time
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader                                     
#import torch.nn.functional as F
from scipy.special import softmax

from gnn import gnn
import utils
from dataset import MolDataset, collate_fn, DTISampler


def main():
    now = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    print (s)

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--lr", help="learning rate", type=float, default = 0.0001)
    parser.add_argument("--epoch", help="epoch", type=int, default = 10000)
    parser.add_argument("--ngpu", help="number of gpu", type=int, default = 1)
    parser.add_argument("--batch_size", help="batch_size", type=int, default = 32)
    parser.add_argument("--num_workers", help="number of workers", type=int, default = 7)
    parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default = 4)
    parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default = 140)
    parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default = 4)
    parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default = 128)
    parser.add_argument("--dude_data_fpath", help="file path of dude data", type=str, default='data/')
    parser.add_argument("--save_dir", help="save directory of model parameter", type=str, default = './save/')
    parser.add_argument("--initial_mu", help="initial value of mu", type=float, default = 4.0)
    parser.add_argument("--initial_dev", help="initial value of dev", type=float, default = 1.0)
    parser.add_argument("--dropout_rate", help="dropout_rate", type=float, default = 0.0)
    parser.add_argument("--train_keys", help="train keys", type=str, default='keys/train_keys.pkl')
    parser.add_argument("--test_keys", help="test keys", type=str, default='keys/test_keys.pkl')
    args = parser.parse_args()
    print (args)


    #hyper parameters
    num_epochs = args.epoch
    lr = args.lr
    ngpu = args.ngpu
    batch_size = args.batch_size
    dude_data_fpath = args.dude_data_fpath
    save_dir = args.save_dir

    #make save dir if it doesn't exist
    if not os.path.isdir(save_dir):
        os.system('mkdir ' + save_dir)
        print('save_dir({}) created'.format(save_dir))
        pass

    print('save_dir:{}'.format(save_dir))
    print('+' * 10)

    #read data. data is stored in format of dictionary. Each key has information about protein-ligand complex.
    with open (args.train_keys, 'rb') as fp:
        train_keys = pickle.load(fp)
        #
        # train_keys: type=list, len=730, ['andr_C36276925', 'dhi1_C08592133', 'hivpr_C59233791', 'hivrt_C66397637', 'cah2_C62892628', ... ]
        #
        print('train_keys({}) loaded from pickle --> type:{}, len:{}, ex:\n{}'.format(args.train_keys, type(train_keys), len(train_keys), train_keys[:5]))
        pass

    print('+' * 3)

    with open (args.test_keys, 'rb') as fp:
        test_keys = pickle.load(fp)
        #
        # test_keys: type=list, len=255, ['fnta_C59365794', 'ace_C22923016', 'aces_C21842010', 'kith_C11223989', 'kpcb_C37928874', ... ]
        #
        print('test_keys({}) loaded from pickle --> type:{}, len:{}, ex:\n{}'.format(args.test_keys, type(test_keys), len(test_keys), test_keys[:5]))
        pass

    print('+' * 10)

    #print simple statistics about dude data and pdbbind data
    print (f'Number of train data: {len(train_keys)}')
    print (f'Number of test data: {len(test_keys)}')

    if 0 < args.ngpu:
        cmd = utils.set_cuda_visible_device(args.ngpu)
        print('utils.set_cuda_visible_device({}) --> cmd:{}'.format(args.ngpu, cmd))
        os.environ['CUDA_VISIBLE_DEVICES'] = cmd[:-1]
        pass

    model = gnn(args)

    print('+' * 10)

    print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() and 0 < args.ngpu else "cpu")

    print ('device: {}'.format(device))

    # initialize model
    model = utils.initialize_model(model, device)

    print('#' * 80)

    print('dude_data_fpath:{}'.format(args.dude_data_fpath))

    #train and test dataset
    train_dataset = MolDataset(train_keys, args.dude_data_fpath)
    test_dataset  = MolDataset(test_keys , args.dude_data_fpath)

    print('#' * 80)

    num_train_chembl = len([0 for k in train_keys if 'CHEMBL'     in k])
    num_train_decoy  = len([0 for k in train_keys if 'CHEMBL' not in k])

    print('#1:num_train_chembl:{}, num_train_decoy:{}'.format(num_train_chembl, num_train_decoy))

    num_train_chembl = len([0 for k in train_keys if 'CHEMBL'     in k])
    num_train_decoy  = len(train_keys) - num_train_chembl

    print('#2:num_train_chembl:{}, num_train_decoy:{}'.format(num_train_chembl, num_train_decoy))


    #train_weights = [1/num_train_chembl if 'CHEMBL' in k else 1/num_train_decoy for k in train_keys]
    train_weight_chembl = 1.0 / num_train_chembl
    train_weight_decoy  = 1.0 / num_train_decoy
    train_weights = [train_weight_chembl if 'CHEMBL' in k else train_weight_decoy for k in train_keys]

    print('main: sum(train_weights):{}'.format(sum(train_weights)))
    print('train_weight_chembl:{} / train_weight_decoy:{}, len(train_weights):{}'.format(
           train_weight_chembl, train_weight_decoy, len(train_weights)))

    train_sampler = DTISampler(train_weights, len(train_weights), replacement=True)                     

    print('main: args.batch_size:{}, args.num_workers:{}'.format(args.batch_size, args.num_workers))

    #
    # train_dataset: object of MolDataset(torch.utils.data.Dataset)
    #
    train_dataloader = DataLoader(train_dataset, args.batch_size, \
         shuffle=False, num_workers = args.num_workers, collate_fn=collate_fn,\
         sampler = train_sampler)

    #
    # test_dataset: object of MolDataset(torch.utils.data.Dataset)
    #
    test_dataloader = DataLoader(test_dataset, args.batch_size, \
         shuffle=False, num_workers = args.num_workers, collate_fn=collate_fn, \
         )
    
    #optimizer
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    #loss function --> BCELoss (Binary Classification Error)
    #loss_fn = nn.BCELoss()
    loss_fn = nn.CrossEntropyLoss()
    

    
    test_roc_list = list()
    best_test_roc = 0.0

    for epoch in range(num_epochs):
        st = time.time()
        #collect losses of each iteration
        train_losses = [] 
        test_losses  = [] 
    
        #collect true label of each iteration
        train_true = []
        test_true  = []
        
        #collect predicted label of each iteration
        train_pred = []
        test_pred  = []
        
        model.train() # sets the model in training mode.
        #print('model.training:{}'.format(model.training))

        for i_batch, sample in enumerate(train_dataloader):
            model.zero_grad()
            H, A1, A2, Y, V, keys = sample 

            n_queried, n_max_n1, n_max_n2, n_max_adj, n_file_opened = train_dataset.get_n_queried()

            if epoch == 0 and i_batch == 0:
                print('#1:{}/{} H:type:{}, shape:{}\n{}'.format(i_batch, epoch, type(H), H.shape, H))
                print('    A1:type:{}, shape:{}\n{}'.format(type(A1), A1.shape, A1))
                print('    A2:type:{}, shape:{}\n{}'.format(type(A2), A2.shape, A2))
                print('    Y:type:{}, shape:{}\n{}'.format(type(Y), Y.shape, Y))
                print('    V:type:{}, shape:{}\n{}'.format(type(V), V.shape, V))
                print('    keys:type:{}\n{}'.format(type(keys), keys))
                print('    train_dataset: n_queried:{}, n_max_n1:{}, n_max_n2:{}, n_max_adj:{}, n_file_opened:{}'.format(
                       n_queried, n_max_n1, n_max_n2, n_max_adj, n_file_opened))
                print('+' * 10)
                pass

            H, A1, A2, Y, V = H.to(device), A1.to(device), A2.to(device),\
                                Y.to(device), V.to(device)

            if epoch == 0 and i_batch == 0:
                print('#2:{}/{} H:type:{}, shape:{}\n{}'.format(i_batch, epoch, type(H), H.shape, H))
                print('    A1:type:{}, shape:{}\n{}'.format(type(A1), A1.shape, A1))
                print('    A2:type:{}, shape:{}\n{}'.format(type(A2), A2.shape, A2))
                print('    Y:type:{}, shape:{}\n{}'.format(type(Y), Y.shape, Y))
                print('    V:type:{}, shape:{}\n{}'.format(type(V), V.shape, V))
                print('    keys:type:{}\n{}'.format(type(keys), keys))
                print('    train_dataset: n_queried:{}, n_max_n1:{}, n_max_n2:{}, n_max_adj:{}, n_file_opened:{}'.format(
                       n_queried, n_max_n1, n_max_n2, n_max_adj, n_file_opened))
                print('+' * 10)
                pass

            #train neural network
            pred = model.train_model((H, A1, A2, V))
            #pred = model.module.train_model((H, A1, A2, V))
            pred = pred.cpu()
            pred_softmax = pred.detach().numpy()
            pred_softmax = softmax(pred_softmax, axis=1)[:,1]

            if epoch == 0 and i_batch == 0:
                print('{}/{} pred:shape:{}\n{}\nY.shape:{}'.format(i_batch, epoch, pred.shape, pred, Y.shape))
                print('+' * 10)
                print('{}/{} pred_softmax:shape:{}\n{}'.format(i_batch, epoch, pred_softmax.shape, pred_softmax))
                print('+' * 10)
                pass

            loss = loss_fn(pred, Y) 

            if epoch == 0 and i_batch == 0:
                print('{}/{} loss:shape:{}\n{}'.format(i_batch, epoch, loss.shape, loss))
                print('+' * 10)
                pass

            loss.backward()
            optimizer.step()
            
            #collect loss, true label and predicted label
            train_losses.append(loss.data.cpu().numpy())
            train_true.append(Y.data.cpu().numpy())
            #train_pred.append(pred.data.cpu().numpy())
            train_pred.append(pred_softmax)
            #if i_batch>10 : break

            pass # end of for i_batch,sample


        model.eval() # equivalent with model.train(mode=False)
        for i_batch, sample in enumerate(test_dataloader):
            model.zero_grad()

            H, A1, A2, Y, V, keys = sample 
            H, A1, A2, Y, V = H.to(device), A1.to(device), A2.to(device),\
                              Y.to(device), V.to(device)
            
            #train neural network
            pred = model.train_model((H, A1, A2, V))
            pred_softmax = pred.detach().numpy()
            pred_softmax = softmax(pred_softmax, axis=1)[:,1]
    
            loss = loss_fn(pred, Y) 
            
            #collect loss, true label and predicted label
            test_losses.append(loss.data.cpu().numpy())
            test_true.append(Y.data.cpu().numpy())
            #test_pred.append(pred.data.cpu().numpy())
            test_pred.append(pred_softmax)
            #if i_batch>10 : break

            if epoch == 0  and i_batch == 0:
                print('eval: Y.shape:{}, pred.shape:{}, pred_softmax.shape:{}'.format(
                        Y.shape, pred.shape, pred_softmax.shape))
                pass
            pass
            
        train_losses = np.mean(np.array(train_losses))
        test_losses  = np.mean(np.array(test_losses ))
        
        train_pred = np.concatenate(np.array(train_pred), 0)
        test_pred  = np.concatenate(np.array(test_pred ), 0)
        
        train_true = np.concatenate(np.array(train_true), 0)
        test_true  = np.concatenate(np.array(test_true ), 0)

        #print('#' * 80)
        #print('train_pred:\n{}'.format(train_pred))
        #print('+' * 7)
        ##print(softmax(train_pred, axis=1))

        #print('+' * 10)
        #print('+' * 10)

        #print('train_true:\n{}'.format(train_true))
        #print('#' * 80, flush=True)

        train_roc = roc_auc_score(train_true, train_pred)
        test_roc  = roc_auc_score(test_true , test_pred )

        end = time.time()
        if epoch == 0:
            print ('epoch\ttrain_losses\ttest_losses\ttrain_roc\ttest_roc\telapsed_time')
            pass
        #print('#' * 80)
        #print ('epoch\ttrain_losses\ttest_losses\ttrain_roc\ttest_roc\telapsed_time')
        #print ("%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" \
        print ('%s\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%s' \
               % (epoch, train_losses, test_losses, train_roc, test_roc, end-st, datetime.datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S.%f')),
               end='')

        #name = save_dir + '/save_'+str(epoch)+'.pt'
        #torch.save(model.state_dict(), name)
        if best_test_roc < test_roc:
            name = save_dir + '/save_'+str(epoch)+'.pt'
            torch.save(model.state_dict(), name)
            print(' updated')

            best_test_roc = test_roc
            pass
        else:
            print('')
            pass

        test_roc_list.append(test_roc)
        pass
    pass


if __name__ == '__main__':
    main()
    pass
