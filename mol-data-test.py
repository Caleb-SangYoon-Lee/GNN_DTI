# -*- coding: utf-8 -*-

import os, sys
import pickle
from collections import OrderedDict, Counter
import random
import glob
import pickle

import numpy as np
from rdkit import Chem
from scipy.spatial import distance_matrix
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

from utils import *

N_FILES_TO_TEST = 1
N_FEATURES = 21


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    #print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1] # 현재는 무조건 Hydrogen으로 세팅
        pass
    return list(map(lambda s: x == s, allowable_set))


def atom_feature(m, atom_i, is_aromatic_atom, i_donor=None, i_acceptor=None):
    fnm = whoami()

    atom = m.GetAtomWithIdx(atom_i) # atom: rdkit.Chem.rdchem.Atom object
    symbol = atom.GetSymbol()
    #if symbol == 'C' and is_aromatic_atom:
    #    symbol = 'c'
    #    pass

    #print('{}:#{:>2}:atom:{} --> {}'.format(fnm, atom_i, atom, symbol))

    #atom_types = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']
    #atom_types = ['C', 'c', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H', 'U']
    atom_types = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H', 'U']

    atom_types = one_of_k_encoding_unk(symbol, atom_types)
    #print('  --> atom_types:len:{}/{}'.format(len(atom_types), atom_types))

    degrees = one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    #print('  --> degrees   :len:{}/{}'.format(len(degrees), degrees))

    #explicit_hs = one_of_k_encoding_unk(atom.GetTotalNumHs()     , [0, 1, 2, 3, 4])
    #implicit_hs = one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
    #is_aromatic = [atom.GetIsAromatic()])

    # misc_info: explicit_hs , implicit_hs , is_aromatic
    #misc_info = [atom.GetTotalNumHs(), atom.GetImplicitValence(), atom.GetIsAromatic()]
    misc_info = [atom.GetTotalNumHs(), atom.GetImplicitValence()]
    aromatic_info = [0, 1] if atom.GetIsAromatic() else [1, 0]
    #print('  --> misc_info :len({}/{}'.format(len(misc_info), misc_info))

    #features = np.array(one_of_k_encoding_unk(atom.GetSymbol(),
    #                                  ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']) +
    #                one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
    #                one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
    #                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
    #                [atom.GetIsAromatic()])    # (10, 6, 5, 6, 1) --> total 28
    #features = np.array(atom_types + degrees + misc_info, dtype=np.float64) # 11 + 6 + (1 + 1 + 1)
    features = np.array(atom_types + degrees + misc_info, dtype=np.float64) # 11 + 6 + (1 + 1) + 2
    #print('  {} --> features :len:{}/{}'.format(symbol, len(features), features))

    return features

    

def get_atom_feature(m, n, is_ligand=True):
    fnm = whoami()
    #n = m.GetNumAtoms()

    print('{}: m:{}, n:{}, is_ligand:{}'.format(fnm, m, n, is_ligand))

    aromatic_atoms = m.GetAromaticAtoms() # rdkit.Chem.rdchem._ROQAtomSeq object
    for atom in aromatic_atoms:
        i = atom.GetIdx()
        print('{}:aromatic_atom#[{:>2}]:{}'.format(fnm, i, atom.GetSymbol()))
        pass

    print('+' * 10)

    aromatic_atom_indices = set([atom.GetIdx() for atom in m.GetAromaticAtoms()])
    print('{}:aromatic_atom_indieces:{}'.format(fnm, aromatic_atom_indices))

    print('+' * 10)

    H = list()
    for i in range(n):
        #H.append(utils.atom_feature(m, i, None, None))
        H.append(atom_feature(m, i, i in aromatic_atom_indices, None, None))
        pass

    H = np.array(H)        
    print('{}: H:shape:{}, type:{}'.format(fnm, H.shape, H.dtype), flush=True)

    if is_ligand:
        #H = np.concatenate([H, np.zeros((n,N_FEATURES))], axis=1)
        #padding = np.zeros((n,N_FEATURES))np.zeros((n, N_FEATURES))
        print('{}: n:{}/{}, N_FEATURES:{}/{}'.format(fnm, type(n), n, type(N_FEATURES), N_FEATURES), flush=True)
        padding = np.zeros((n, N_FEATURES))
        H = np.concatenate([H, padding], axis=1)
        pass
    else:
        H = np.concatenate([np.zeros((n,N_FEATURES)), H], axis=1)
        pass
    return H        


def anal_mols(key, m1, m2):
    fnm = whoami()

    #
    # prepare ligand
    #
    m1 = Chem.AddHs(m1, addCoords=True, addResidueInfo=True)
    n1 = m1.GetNumAtoms()
    c1 = m1.GetConformers()[0]  # m1.GetConformers() 함수는 1개의 rdkit.Chem.rdchem.Conformer object 만을 되돌려 줌
    d1 = np.array(c1.GetPositions())
    #print('{}:#1:n_atoms:{} --> shape:{}\n{}'.format(fnm, n1, d1.shape, d1))
    print('{}:#1:n_atoms:{} --> shape:{}'.format(fnm, n1, d1.shape))

    print('+' * 3)

    for i, coord in enumerate(d1):
        symbol = m1.GetAtomWithIdx(i).GetSymbol()
        print('  #{:>3}:{}:{}'.format(i, symbol, coord))
        pass

    print('+' * 10)

    adj1 = GetAdjacencyMatrix(m1) + np.eye(n1)  # adj1.dtype: float64
    print('{}:#2:adj1:shape:{}, dtype:{}\n{}'.format(fnm, adj1.shape, adj1.dtype, adj1))
    print('+' * 3)

    print('{}:m1:{}, n1:{}'.format(fnm, m1, n1))
    H1 = get_atom_feature(m1, n1, True)

    print('#' * 80)

    #
    # prepare protein
    #
    m2 = Chem.AddHs(m2, addCoords=True, addResidueInfo=True)
    n2 = m2.GetNumAtoms()
    c2 = m2.GetConformers()[0]  # m2.GetConformers() 함수는 1개의 rdkit.Chem.rdchem.Conformer object 만을 되돌려 줌
    d2 = np.array(c2.GetPositions())
    print('{}:#1:n_atoms:{} --> shape:{}\n{}'.format(fnm, n2, d2.shape, d2))

    print('+' * 10)

    adj2 = GetAdjacencyMatrix(m2) + np.eye(n2)
    print('{}:#2:adj2:shape:{}, dtype:{}\n{}'.format(fnm, adj2.shape, adj2.dtype, adj2))
    print('+' * 3)

    H2 = get_atom_feature(m2, n2, False)

    print('#' * 80)

    # aggregation
    H = np.concatenate([H1, H2], axis=0)
    print('{}: H:shape:{}, type:{}'.format(fnm, H.shape, H.dtype), flush=True)

    print('+' * 10)
    print('n:{} = n1:{} + n2:{}'.format(n1 + n2, n1, n2))

    #
    # agg_adj1: 인접행렬(1)
    #
    #    - 행렬의 upper-left  부분: ligand  내부의 인접행렬
    #    - 행렬의 lower-right 부분: protein 내부의 인접행렬
    #    - 위의 2영역을 제외한 나머지는 0(zero)로 패딩됨
    #
    agg_adj1 = np.zeros((n1+n2, n1+n2))
    agg_adj1[:n1, :n1] = adj1
    agg_adj1[n1:, n1:] = adj2

    print('{}: agg_adj1:shape:{}, type:{}'.format(fnm, agg_adj1.shape, agg_adj1.dtype), flush=True)

    #
    # agg_adj2: 인접행렬(2)
    #
    #    - 행렬의 upper-left  부분: ligand  내부의 인접행렬
    #    - 행렬의 upper-right 부분: row기준으로(ligand기준 ) ligand와 protein간의 거리
    #    - 행렬의 lower-left  부분: row기준으로(protein기준) protein과 ligand간의 거리
    #    - 행렬의 lower-right 부분: protein 내부의 인접행렬
    #
    agg_adj2 = np.copy(agg_adj1)

    print('{}: agg_adj2:shape:{}, type:{}'.format(fnm, agg_adj2.shape, agg_adj2.dtype), flush=True)

    dm = distance_matrix(d1, d2)
    print('{}: dm:shape:{}, type:{}, min:{}, max:{}'.format(fnm, dm.shape, dm.dtype, dm.min(), dm.max()), flush=True)

    agg_adj2[:n1,n1:] = np.copy(dm)
    agg_adj2[n1:,:n1] = np.copy(np.transpose(dm))

    #node indice for aggregation
    valid = np.zeros((n1+n2,))
    valid[:n1] = 1
    print('valid:{}, sum(valid):{}'.format(valid, sum(valid)))

    #pIC50 to class
    Y = 1 if 'CHEMBL' in key else 0

    sample = {'H'  : H       
             ,'A1' : agg_adj1
             ,'A2' : agg_adj2
             ,'Y'  : Y       
             ,'V'  : valid   
             ,'key': key     
             }
    return sample


def main():
    random.seed(0)

    data_dir = os.path.join('.', 'data')
    file_pattern = os.path.join(data_dir, '*')

    data_files = glob.glob(file_pattern)
    valid_keys = [v.split('/')[-1] for v in data_files]

    for i in range(N_FILES_TO_TEST):
        data_file = data_files[i]
        key = os.path.split(data_file)[-1]
        print('key:{}'.format(key))

        with open(data_file, 'rb') as f:
            m1, m2 = pickle.load(f)
            data = anal_mols(key, m1, m2)
            print('#{:>3}:data:size:{}\n{}'.format(i, sys.getsizeof(data), data))
            pass
        pass

    pass

if __name__ == '__main__':
    main()
    pass
