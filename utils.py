# -*- coding: utf-8 -*-

import numpy as np
import torch
from scipy import sparse
import os.path
import time
import torch.nn as nn

# ase: Atomic Simulation Environment, https://wiki.fysik.dtu.dk/ase/
from ase import Atoms, Atom

#from rdkit.Contrib.SA_Score.sascorer import calculateScore
#from rdkit.Contrib.SA_Score.sascorer
#import deepchem as dc

from dataset import ATOM_TYPES
from misc import whoami


def set_cuda_visible_device(ngpus):
    fnm = whoami()

    import subprocess
    import os

    empty = list()
    for i in range(8):
        command = 'nvidia-smi -i '+str(i)+' | grep -i "No running" | wc -l'
        #print('{}: #{}: command:{}'.format(fnm, i, command))

        output = subprocess.check_output(command, shell=True).decode("utf-8")
        #print('nvidia-smi -i '+str(i)+' | grep "No running" | wc -l > empty_gpu_check')
        if int(output) == 1:
            empty.append(i)
            pass
        pass

    if len(empty) < ngpus:
        print ('avaliable gpus are less than required')
        exit(-1)

    cmd = ''
    for i in range(ngpus):        
        cmd+=str(empty[i])+','
        pass

    return cmd


def initialize_model(model, device, load_save_file=False):
    fnm = whoami()

    print('{}: device:{}({}), toch.cuda.device_count():{}, load_save_file:{}'.format(fnm, device, type(device), torch.cuda.device_count(), load_save_file))

    if load_save_file:
        model.load_state_dict(torch.load(load_save_file)) 
        pass
    else:
        for param in model.parameters():
            if param.dim() == 1:
                continue
                nn.init.constant(param, 0)
            else:
                #nn.init.normal(param, 0.0, 0.15)
                nn.init.xavier_normal_(param)
                pass
            pass # end of for param ...
        pass # end of else

    #if 1 < torch.cuda.device_count():
    if 1 < torch.cuda.device_count() and str(device).lower() != 'cpu':
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
        pass

    model.to(device)

    return model


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    #print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
        pass
    return list(map(lambda s: x == s, allowable_set))

def atom_feature(m, atom_i, i_donor=None, i_acceptor=None):
    atom = m.GetAtomWithIdx(atom_i)
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])    # (10, 6, 5, 6, 1) --> total 28
    '''
    symbol = atom.GetSymbol()

    atom_types = one_of_k_encoding_unk(atom.GetSymbol(), ATOM_TYPES)     # 11
    degrees    = one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) #  6
    misc_info  = [atom.GetTotalNumHs(), atom.GetImplicitValence()]       #  2 (1 + 1)
    aromatic_info = [0.0, 1.0] if atom.GetIsAromatic() else [1.0, 0.0]   #  2

    #
    # return feature which length is: 11 + 6 + (1 + 1) + 2
    #
    #feature = np.array(atom_types + degrees + misc_info + aromatic_info, dtype=np.float64)
    #return feature
    return np.array(atom_types + degrees + misc_info + aromatic_info, dtype=np.float64)
    '''
