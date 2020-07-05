# -*- coding: utf-8 -*-

import os

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import utils
from misc import whoami
import numpy as np
import torch
import random
from rdkit import Chem
from scipy.spatial import distance_matrix
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import pickle
random.seed(0)

#N_ATOM_FEATURES = 21
#N_ATOM_FEATURES = 20
N_ATOM_FEATURES = 28
N_FEATURES = N_ATOM_FEATURES * 2

ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']  # original types in the paper
#ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H', 'U'] # 'U': Unknown

N_PADDED_LIGAND  = 40
N_PADDED_PROTEIN = 60
N_PADDED_ALL     = N_PADDED_LIGAND + N_PADDED_PROTEIN

N_PADDED_LIGAND_MAX  = 300
N_PADDED_PROTEIN_MAX = 500


#def get_atom_feature(m, is_ligand=True):
def get_atom_feature(m, n, is_ligand=True):
    #n = m.GetNumAtoms()
    H = list()

    for i in range(n):
        H.append(utils.atom_feature(m, i, None, None))
        pass

    H = np.array(H)        

    if is_ligand:
        ##H = np.concatenate([H, np.zeros((n,28))], 1)
        #H = np.concatenate([H, np.zeros((n, N_ATOM_FEATURES))], axis=1)
        H_padded = np.zeros((N_PADDED_LIGAND, N_FEATURES), dtype=np.float64)

        if n <= N_PADDED_LIGAND:
            H_padded[:n, :N_ATOM_FEATURES] = H
            pass
        else:
            H_padded[:, :N_ATOM_FEATURES] = H[:N_PADDED_LIGAND,:]
            pass
        H = H_padded
        pass
    else:
        ##H = np.concatenate([np.zeros((n,28)), H], 1)
        #H = np.concatenate([np.zeros((n, N_ATOM_FEATURES)), H], axis=1)
        H_padded = np.zeros((N_PADDED_PROTEIN, N_FEATURES), dtype=np.float64)

        if n <= N_PADDED_PROTEIN:
            H_padded[:n, N_ATOM_FEATURES:] = H
            pass
        else:
            H_padded[:, N_ATOM_FEATURES:] = H[:N_PADDED_PROTEIN,:]
            pass
        H = H_padded
        pass
    return H


class MolDataset(Dataset):
    def __init__(self, keys, data_dir):
        fnm = __class__.__name__ + '.' + whoami()
        print('{}:len(keys):{}, keys[:5]:{}, data_dir:{}'.format(fnm, len(keys), keys[:5], data_dir))

        self.keys = keys
        self.data_dir = data_dir

        self.proc_info_printed = False
        self.n_queried         = 0
        self.n_max_n1          = 0
        self.n_max_n2          = 0
        self.n_max_adj         = 0
        self.n_file_opened     = 0
        pass

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        fnm = __class__.__name__ + '.' + whoami()
        self.n_queried += 1

        #idx = 0
        key = self.keys[idx]
        data_file_path = os.path.join(self.data_dir, key)

        #with open(self.data_dir+'/'+key, 'rb') as f:
        with open(data_file_path, 'rb') as f:
            m1, m2 = pickle.load(f)

            self.n_file_opened += 1
            pass

        if not self.proc_info_printed:
            print('{}: data_file_path:{}, type(m1):{}, type(m2):{}'.format(fnm, data_file_path, type(m1), type(m2)))
            pass

        #
        # prepare ligand
        #
        #m1   = Chem.AddHs(m1, addCoords=True, addResidueInfo=True) # 2020-03-26 added by caleb
        n1   = m1.GetNumAtoms()
        c1   = m1.GetConformers()[0]  # m1.GetConformers() 함수는 1개의 rdkit.Chem.rdchem.Conformer object 만을 되돌려 줌
        d1   = np.array(c1.GetPositions())
        #adj1 = GetAdjacencyMatrix(m1) + np.eye(n1)
        adj = GetAdjacencyMatrix(m1) + np.eye(n1)

        if n1 <= N_PADDED_LIGAND:
            adj1 = np.zeros((N_PADDED_LIGAND,N_PADDED_LIGAND), dtype=np.float64)
            adj1[:n1, :n1] = adj
            pass
        else:
            adj1 = adj[:N_PADDED_LIGAND, :N_PADDED_LIGAND]
            pass

        #H1   = get_atom_feature(m1, True)
        H1   = get_atom_feature(m1, n1, True)

        #
        # prepare protein
        #
        #m2   = Chem.AddHs(m2, addCoords=True, addResidueInfo=True) # 2020-03-26 added by caleb
        n2   = m2.GetNumAtoms()
        c2   = m2.GetConformers()[0]
        d2   = np.array(c2.GetPositions())
        #adj2 = GetAdjacencyMatrix(m2)+np.eye(n2)
        adj = GetAdjacencyMatrix(m2)+np.eye(n2)

        if n2 <= N_PADDED_PROTEIN:
            adj2 = np.zeros((N_PADDED_PROTEIN,N_PADDED_PROTEIN), dtype=np.float64)
            adj2[:n2, :n2] = adj
            pass
        else:
            adj2 = adj[:N_PADDED_PROTEIN, :N_PADDED_PROTEIN]
            pass

        #H2   = get_atom_feature(m2, False)
        H2   = get_atom_feature(m2, n2, False)
        
        # aggregation
        H = np.concatenate([H1, H2], axis=0)

        '''
        agg_adj1 = np.zeros((n1+n2, n1+n2))
        agg_adj1[:n1, :n1] = adj1
        agg_adj1[n1:, n1:] = adj2

        agg_adj2 = np.copy(agg_adj1)

        dm = distance_matrix(d1,d2)
        agg_adj2[:n1,n1:] = np.copy(dm)
        agg_adj2[n1:,:n1] = np.copy(np.transpose(dm))

        #node indice for aggregation
        valid = np.zeros((n1+n2,))
        valid[:n1] = 1
        '''
        agg_adj1 = np.zeros((N_PADDED_ALL, N_PADDED_ALL))
        agg_adj1[:N_PADDED_LIGAND, :N_PADDED_LIGAND] = adj1
        agg_adj1[N_PADDED_LIGAND:, N_PADDED_LIGAND:] = adj2

        agg_adj2 = np.copy(agg_adj1)

        dm = distance_matrix(d1,d2)
        #
        # 2020-03-27
        #   * (계산의 편의를 위해) 무식하게 최대크기(라고 가정한) 매트릭스를 특정값으로 세팅함
        #   * 거리정보가 없는 녀석들은 먼거리(여기서는 100.0)로 세팅해 놓음 --> 그냥 0으로 세팅함
        #
        #dm_padded = np.full((N_PADDED_LIGAND_MAX, N_PADDED_PROTEIN_MAX), fill_value=100.0, dtype=np.float64)
        dm_padded = np.zeros((N_PADDED_LIGAND_MAX, N_PADDED_PROTEIN_MAX), dtype=np.float64)
        dm_padded[:n1, :n2] = dm
        dm = dm_padded[:N_PADDED_LIGAND, :N_PADDED_PROTEIN]

        #agg_adj2[:n1,n1:] = np.copy(dm)
        #agg_adj2[n1:,:n1] = np.copy(np.transpose(dm))
        agg_adj2[:N_PADDED_LIGAND,N_PADDED_LIGAND:] = np.copy(dm)
        agg_adj2[N_PADDED_LIGAND:,:N_PADDED_LIGAND] = np.copy(np.transpose(dm))

        #node indice for aggregation
        #valid = np.zeros((n1+n2,))
        #valid[:n1] = 1
        valid = np.zeros((N_PADDED_ALL,))
        valid[:N_PADDED_LIGAND] = 1
        
        #pIC50 to class
        Y = 1 if 'CHEMBL' in key else 0

        #if n1+n2 > 300 : return None
        sample = {
                  'H'  : H       , \
                  'A1' : agg_adj1, \
                  'A2' : agg_adj2, \
                  'Y'  : Y       , \
                  'V'  : valid   , \
                  'key': key     , \
                 }

        if self.n_max_n1 < n1:
            self.n_max_n1 = n1
            pass

        if self.n_max_n2 < n2:
            self.n_max_n2 = n2
            pass

        if self.n_max_adj < n1 + n2:
            self.n_max_adj = n1 + n2
            pass

        if not self.proc_info_printed:
            #print('{}: n1:{}, n2:{}, H.shape:{}, A1.shape:{}, A2.shape:{}, Y.shape:{}, V.shape:{}, key:{}'.format(
            #       fnm, n1, n2, H.shape, adj1.shape, adj2.shape, Y.shape, V.shape, key))
            #print('{}: n1:{}, n2:{}, type(H):{}, type(adj1):{}, type(adj2):{}, type(Y):{}, type(valid):{}({}), key:{}'.format(
            #       fnm, n1, n2, type(H), type(adj1), type(adj2), type(Y), type(valid)(valid[:10]), key[:10]))
            print('{}: n1:{}, n2:{}, H.shape:{}, adj1.shape:{}, adj2.shape:{}, type(Y):{}, type(valid):{}, type(key):{}:{}'.format(
                   fnm, n1, n2, H.shape, adj1.shape, adj2.shape, type(Y), type(valid), type(key), key))
            pass

        self.proc_info_printed = True

        return sample

    def get_n_queried(self):
        n_queried, n_max_n1, n_max_n2, n_max_adj, n_file_opened = self.n_queried, self.n_max_n1, self.n_max_n2, self.n_max_adj, self.n_file_opened
        self.n_queried, self.n_max_n1, self.n_max_n2, self.n_max_adj, self.n_file_opened = 0, 0, 0, 0, 0
        return n_queried, n_max_n1, n_max_n2, n_max_adj, n_file_opened


class DTISampler(Sampler):
    def __init__(self, weights, n_samples, replacement=True):
        fnm = __class__.__name__ + '.' + whoami()

        print('{}:#1: np.sum(weights):{}, weights.shape:{}, n_samples:{}'.format(fnm, np.sum(weights), len(weights), n_samples))
        weights = np.array(weights)/np.sum(weights)
        print('{}:#2: np.sum(weights):{}, weights.shape:{}, n_samples:{}'.format(fnm, np.sum(weights), weights.shape, n_samples))

        self.weights = weights
        self.n_samples = n_samples
        self.replacement = replacement
        pass
    
    def __iter__(self):
        #return iter(torch.multinomial(self.weights, self.n_samples, self.replacement).tolist())
        retval = np.random.choice(len(self.weights), self.n_samples, replace=self.replacement, p=self.weights) 
        return iter(retval.tolist())

    def __len__(self):
        return self.n_samples


def collate_fn(batch):
    max_natoms = max([len(item['H']) for item in batch if item is not None])
    
    #
    # [BEFORE]
    #
    # 56: internal feature vector size
    #
    #    features of atoms in protein : 28
    #    features of atoms in compound: 28
    #
    #    * atom type                 (10): C, N, O, S, F, P, Cl, Br, B, H (one-hot)
    #    * degree of atom            ( 6): 0, 1, 2, 3, 4, 5 (one-hot)
    #    * #N of hydrogen atom       ( 5): 0, 1, 2, 3, 4    (one-hot)
    #    * implicit valence electrons( 6): 0, 1, 2, 3, 4, 5 (one-hot)
    #    * arotmatic                 ( 1): 0 or 1
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #
    # [AFTER]
    #
    # 42: internal feature vector size
    #
    #    features of atoms in protein : 21
    #    features of atoms in compound: 21
    #
    #    * atom type                 (11): C, N, O, S, F, P, Cl, Br, B, H, U (one-hot) : UuUnknown)
    #    * degree of atom            ( 6): 0, 1, 2, 3, 4, 5 (one-hot)
    #    * #N of hydrogen atom       ( 1): scalar
    #    * implicit valence electrons( 1): scalar
    #    * arotmatic                 ( 2): 0, 1 (one-hot)
    #
    #H  = np.zeros((len(batch), max_natoms, 56))
    H  = np.zeros((len(batch), max_natoms, N_FEATURES))
    A1 = np.zeros((len(batch), max_natoms, max_natoms)) # : adjacent matrix A1
    A2 = np.zeros((len(batch), max_natoms, max_natoms)) # : adjacent matrix A2
    Y  = np.zeros((len(batch),))
    V  = np.zeros((len(batch), max_natoms))

    keys = list()

    for i in range(len(batch)):
        natom = len(batch[i]['H'])

        H [i,:natom]        = batch[i]['H' ]
        A1[i,:natom,:natom] = batch[i]['A1']
        A2[i,:natom,:natom] = batch[i]['A2']
        Y [i]               = batch[i]['Y' ]
        V [i,:natom]        = batch[i]['V' ]

        keys.append(batch[i]['key'])
        pass

    H  = torch.from_numpy( H).float()
    A1 = torch.from_numpy(A1).float()
    A2 = torch.from_numpy(A2).float()
    #Y  = torch.from_numpy( Y).float()
    Y  = torch.from_numpy( Y).long()
    V  = torch.from_numpy( V).float()
    
    return H, A1, A2, Y, V, keys
