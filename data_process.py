import os
import dgl
import torch 
import numpy as np  
import pandas as pd
import networkx as nx
from Bio import SeqIO 
from rdkit import Chem
from functools import partial
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from torch_scatter import scatter
from sklearn.metrics import pairwise_distances
from rdkit.Chem.rdmolops import GetAdjacencyMatrix 
from rdkit.Chem.rdchem import HybridizationType, BondType 
from dgl.nn import EGNNConv
from torch.nn.utils.rnn import pad_sequence
from pandas.core.frame import DataFrame
from Bio.PDB import PDBParser
from torch_geometric.data import Data
import re
import deepchem as dc
import scipy.sparse as sp
import pickle
from numpy import flip
from rdkit.Chem import MACCSkeys
import ssl
from random import shuffle
from itertools import compress
from collections import defaultdict
import csv
import mlcrate as mlc
import os

def one_of_k_encoding(x, allowable_set):   # onehot
    if x not in allowable_set:
        raise ValueError("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):  #onehot
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def generate_smiles_nodes_edges_coords_graph_features(type, m):
    if type == 'smile':
        mol = Chem.MolFromSmiles(m)
    elif type == 'mol2':
        mol = Chem.MolFromMol2File(m)
    elif type == 'sdf':
        mol = Chem.MolFromMolFile(m)

    # nodes feature
    atom_type, atom_number, atom_hybridization = [], [], []
    atomHs, atom_charge, atom_imvalence = [], [], []
    atom_aromatic = []
    atoms = mol.GetAtoms()
    num_atoms = mol.GetNumAtoms()
    for i, atom in enumerate(atoms):
        atom_type.append(atom.GetSymbol())  # C,H,O
        atom_number.append(atom.GetAtomicNum())
        atom_hybridization.append(atom.GetHybridization())  # SP, SP2
        atomHs.append(atom.GetTotalNumHs())  # 0,1,2,3
        atom_charge.append(atom.GetFormalCharge()) # 0, +1,-1, +2, -2, +3, -3
        atom_imvalence.append(atom.GetImplicitValence())  # 获得原子的隐式化合价 0 1 2 3
        atom_aromatic.append(1 if atom.GetIsAromatic() else 0)
        
    nodes_hybridization = [one_of_k_encoding(h, [HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3, HybridizationType.SP3D, HybridizationType.SP3D2]) for h in atom_hybridization]
    nodes_type = [one_of_k_encoding(t[0], ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) for t in atom_type]
    atom_Hs = [one_of_k_encoding(t, [0,1,2,3,4]) for t in atomHs]
    atom_imvalence = [one_of_k_encoding(t, [0,1,2,3,4,5,6]) for t in atom_imvalence]
    
    # bonds feature
    bond_type, bond_conj, bond_ring = [], [], []
    row, col = [], []
    bonds = mol.GetBonds()
    
    for bond in bonds:   
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        bond_type += 2 * [bond.GetBondType()]
        bond_ring += 2 * [bond.IsInRing()]
        bond_conj += 2 * [bond.GetIsConjugated()]
    
    edge_conj = [one_of_k_encoding(t, [True, False]) for t in bond_conj]
    edge_ring = [one_of_k_encoding(t, [True, False]) for t in bond_ring]   
    bond_index = torch.LongTensor([row, col])
    edge_type = [one_of_k_encoding(t, [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]) for t in bond_type]
    edge_attri = torch.FloatTensor(edge_type)
    perm = (bond_index[0] * num_atoms + bond_index[1]).argsort()  
    edge_index = bond_index[:, perm]
    edge_attr = edge_attri[perm]
    row, col = edge_index
    
    atnum_sha = (torch.tensor(atom_number, dtype=torch.long) == 1).to(torch.float)  # size.23  
    atnum_hs = scatter(atnum_sha[row], col, dim_size=num_atoms).tolist()  # len 23
    atom_f1 = torch.tensor([atom_number, atom_aromatic, atnum_hs], dtype=torch.float).t().contiguous()
    
    # concatence features
    mol_all_nodes_feature = torch.cat([torch.FloatTensor(nodes_hybridization), torch.FloatTensor(nodes_type), 
                        torch.FloatTensor(atom_Hs), torch.FloatTensor(atom_imvalence), atom_f1], dim=-1)
    mol_all_edges_feature = torch.cat([torch.FloatTensor(edge_conj),torch.FloatTensor(edge_ring), torch.FloatTensor(edge_type)], dim=-1)
    
    # generate_atom_coordinate
    if type == 'smile':
        mh =Chem.AddHs(mol)
        AllChem.EmbedMolecule(mh)
        AllChem.MMFFOptimizeMolecule(mh)
        mol = Chem.RemoveHs(mh)
        mol_all_coords_feature = torch.FloatTensor(mol.GetConformer().GetPositions())   
    elif type == 'mol2' or type == 'sdf':
        mol_all_coords_feature = torch.FloatTensor(mol.GetConformer().GetPositions())

    return mol, mol_all_nodes_feature, mol_all_edges_feature, mol_all_coords_feature, edge_index, edge_attr

def normalize_adj(adj):
    row = []
    for i in range(adj.shape[0]):  
        sum = adj[i].sum()
        row.append(sum)
    rowsum = np.array(row)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.   
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  
    a = d_mat_inv_sqrt.dot(adj)  
    return a

def preprocess_adj(adj, norm=True, sparse=False):
    adj = adj + np.eye(len(adj)) 
    if norm:
        adj = normalize_adj(adj) 
    return adj

def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency, dtype=float)

def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold

def random_scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
                          frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0):
    """
    Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/chainer_chemistry/dataset/splitters/scaffold_splitter.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
    examples with null value in specified task column of the data.y tensor
    prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
    task_idx is provided
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param seed;
    :return: train, valid, test slices of the input dataset obj
    """

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[:, task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)
    for ind, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)

    scaffold_sets = rng.permutation(list(scaffolds.values()))

    n_total_valid = int(np.floor(frac_valid * len(dataset)))
    n_total_test = int(np.floor(frac_test * len(dataset)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    return train_dataset, valid_dataset, test_dataset

def split(type, fili):
    shuffle(fili)
    if type == 'random':
        shuffled = fili
        train_size = int(0.8 * len(shuffled))
        val_size = int(0.1 * len(shuffled))
        trn = shuffled[:train_size]
        val = shuffled[train_size:(train_size + val_size)]
        test = shuffled[(train_size + val_size):]
        return trn, val, test
    elif type == 'scaffold':
        shuffled = fili
        trn, val, test = random_scaffold_split(dataset=shuffled, smiles_list=shuffled.data.smi, null_value=-1, seed='1234')
        return trn, val, test

def get_feature(dataset, path, dir_data, dataname):
    save_path = f'{path}/{dir_data}/{dataname}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    i = 0
    for x, label, w, smile in dataset.itersamples():
        try:
            if len(smile) > 1:
                i = i + 1
                interaction = label
                mol = Chem.MolFromSmiles(smile)
                mol, mol_feature, mol_edges_feature, mol_coord_feature, edge_index, edge_attr = generate_smiles_nodes_edges_coords_graph_features('smile', smile)
                atom_matrix = GetAdjacencyMatrix(mol)
                mol_nc_feas = torch.cat((mol_feature, mol_coord_feature), dim=1)

                fp = []
                Mfp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
                fp.extend(Mfp) 
                Cfp = MACCSkeys.GenMACCSKeys(mol)
                fp.extend(Cfp)
                finger = np.array(fp)

                mol_pocket_protein_fea = Data(smile=smile,
                                                interaction=interaction,
                                                mol_feature=mol_feature,
                                                mol_edges_feature=mol_edges_feature,
                                                mol_coord_feature=mol_coord_feature,
                                                edge_index=edge_index,
                                                edge_attr=edge_attr,
                                                atom_matrix=atom_matrix,
                                                mol_nc_feas=mol_nc_feas,
                                                finger=finger)
                                    
                torch.save(mol_pocket_protein_fea, f'{save_path}/{i}_plp.pt')
                print(f'processing {i}th molecule')
        except Exception as error:
            print(error)
            print(f'there is something wrong with {i}th molecule')
            continue

def get_csv_feature(path, dir_data, dataname):
    save_path = f'{path}/{dir_data}'  # './prepro_data/ECFP_random_IDH1_3D_attr/train'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    i = 0
    m = 0
    for row in dataname:
        print(row)
        smile = row[0]
        try:
            if len(smile) > 1:
                i = i + 1
                interaction = row[1]
                mol = Chem.MolFromSmiles(smile)
                smi = Chem.MolToSmiles(mol)
                mol, mol_feature, mol_edges_feature, mol_coord_feature, edge_index, edge_attr = generate_smiles_nodes_edges_coords_graph_features('smile', smi)
                atom_matrix = GetAdjacencyMatrix(mol)
                mol_nc_feas = torch.cat((mol_feature, mol_coord_feature), dim=1)

                fp = []
                Mfp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=857)
                fp.extend(Mfp) 
                Cfp = MACCSkeys.GenMACCSKeys(mol)
                fp.extend(Cfp)
                finger = np.array(fp)

                mol_pocket_protein_fea = Data(smile=smile,
                                                interaction=interaction,
                                                mol_feature=mol_feature,
                                                mol_edges_feature=mol_edges_feature,
                                                mol_coord_feature=mol_coord_feature,
                                                edge_index=edge_index,
                                                edge_attr=edge_attr,
                                                atom_matrix=atom_matrix,
                                                mol_nc_feas=mol_nc_feas,
                                                finger=finger)
                                    
                torch.save(mol_pocket_protein_fea, f'{save_path}/{i}_plp.pt')
                print(f'processing {i}th molecule')
        except Exception as error:
            m = m + 1
            print(error)
            print(f'there is something wrong with {i}th molecule')
            continue
    print(f'there is total {m} wrong molecules')

def get_breast_feature(path, dir_data, dataname, na):
    save_path = f'{path}/{dir_data}/{na}'  # './prepro_data/BreastCellLines/ECFP_random_BT-20_3D_attr/train'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    i = 0
    m = 0
    for row in dataname:
        smile = row[0]
        try:
            if len(smile) > 1:
                i = i + 1
                interaction = row[1]
                mol = Chem.MolFromSmiles(smile)
                smi = Chem.MolToSmiles(mol)
                mol, mol_feature, mol_edges_feature, mol_coord_feature, edge_index, edge_attr = generate_smiles_nodes_edges_coords_graph_features('smile', smi)
                atom_matrix = GetAdjacencyMatrix(mol)
                mol_nc_feas = torch.cat((mol_feature, mol_coord_feature), dim=1)

                fp = []
                Mfp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
                fp.extend(Mfp) 
                Cfp = MACCSkeys.GenMACCSKeys(mol)
                fp.extend(Cfp)
                finger = np.array(fp)

                mol_pocket_protein_fea = Data(smile=smile,
                                                interaction=interaction,
                                                mol_feature=mol_feature,
                                                mol_edges_feature=mol_edges_feature,
                                                mol_coord_feature=mol_coord_feature,
                                                edge_index=edge_index,
                                                edge_attr=edge_attr,
                                                atom_matrix=atom_matrix,
                                                mol_nc_feas=mol_nc_feas,
                                                finger=finger)
                                    
                torch.save(mol_pocket_protein_fea, f'{save_path}/{i}_plp.pt')
                print(f'processing {i}th molecule')
        except Exception as error:
            m = m + 1
            print(error)
            print(f'there is something wrong with {i}th molecule')
            continue
    print(f'there is total {m} wrong molecules')

def get_tdc_feature(path, dir_data, data):
    save_path = f'{path}/{dir_data}'  # "./prepro_data/ECFP_<class 'type'>_Caco2_Wang_3D_attr/train"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    i = 0
    m = 0
    h = len(data)
    for j in range(0,h):
        smile = data['Drug'][j]
        interaction = data['Y'][j]
        try:
            if len(smile) > 1:
                i = i + 1
                mol = Chem.MolFromSmiles(smile)
                smi = Chem.MolToSmiles(mol)
                mol, mol_feature, mol_edges_feature, mol_coord_feature, edge_index, edge_attr = generate_smiles_nodes_edges_coords_graph_features('smile', smi)
                atom_matrix = GetAdjacencyMatrix(mol)
                mol_nc_feas = torch.cat((mol_feature, mol_coord_feature), dim=1)

                fp = []
                Mfp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=857)
                fp.extend(Mfp) 
                Cfp = MACCSkeys.GenMACCSKeys(mol)
                fp.extend(Cfp)
                finger = np.array(fp)

                mol_pocket_protein_fea = Data(smile=smile,
                                                interaction=interaction,
                                                mol_feature=mol_feature,
                                                mol_edges_feature=mol_edges_feature,
                                                mol_coord_feature=mol_coord_feature,
                                                edge_index=edge_index,
                                                edge_attr=edge_attr,
                                                atom_matrix=atom_matrix,
                                                mol_nc_feas=mol_nc_feas,
                                                finger=finger)
                                    
                torch.save(mol_pocket_protein_fea, f'{save_path}/{i}_plp.pt')
                print(f'processing {i}th molecule')
        except Exception as error:
            m = m + 1
            print(error)
            print(f'there is something wrong with {i}th molecule')
            continue
    print(f'there is total {m} wrong molecules')

#---bbbp---
# ssl._create_default_https_context = ssl._create_unverified_context
# delaney_tasks, delaney_datasets, transformers = dc.molnet.load_bbbp(featurizer='ECFP', splitter='random') 
# dir_data = 'ECFP_random_bbbp_3D_attr'
# save_path = f'./prepro_data_1024/moleculenet'
# train_dataset, valid_dataset, test_dataset = delaney_datasets
# get_feature(train_dataset, save_path, dir_data, dataname='train_data')
# get_feature(valid_dataset, save_path, dir_data, dataname='valid_data')
# get_feature(test_dataset, save_path, dir_data, dataname='test_data')

# # --freesolv--
# ssl._create_default_https_context = ssl._create_unverified_context
# delaney_tasks, freesolv_datasets, transformers = dc.molnet.load_freesolv(featurizer='ECFP', splitter='random') 
# dir_data = 'ECFP_random_freesolv_3D_attr'
# save_path = f'./prepro_data_1024/moleculenet'
# train_dataset, valid_dataset, test_dataset = freesolv_datasets
# get_feature(train_dataset, save_path, dir_data, dataname='train_data')
# get_feature(valid_dataset, save_path, dir_data, dataname='valid_data')
# get_feature(test_dataset, save_path, dir_data, dataname='test_data')

#---lipo---
ssl._create_default_https_context = ssl._create_unverified_context
delaney_tasks, lipo_datasets, transformers = dc.molnet.load_lipo(featurizer='ECFP', splitter='random') 
dir_data = 'ECFP_random_lipophilicity_3D_attr'
save_path = f'./prepro_data_1024/moleculenet'
train_dataset, valid_dataset, test_dataset = lipo_datasets
get_feature(train_dataset, save_path, dir_data, dataname='train_data')
get_feature(valid_dataset, save_path, dir_data, dataname='valid_data')
get_feature(test_dataset, save_path, dir_data, dataname='test_data')

#----hiv----
"""delaney_tasks, hiv_datasets, transformers = dc.molnet.load_hiv(featurizer='ECFP', splitter='random') 
dir_data = 'ECFP_random_hiv_3D_attr'
save_path = f'./prepro_data'
train_dataset, valid_dataset, test_dataset = hiv_datasets
get_feature(train_dataset, save_path, dir_data, dataname='train_data')
get_feature(valid_dataset, save_path, dir_data, dataname='valid_data')
get_feature(test_dataset, save_path, dir_data, dataname='test_data')"""

#----sider---
# ssl._create_default_https_context = ssl._create_unverified_context
# delaney_tasks, sider_datasets, transformers = dc.molnet.load_sider(featurizer='ECFP', splitter='random') 
# dir_data = 'ECFP_random_sider_3D_attr'
# save_path = f'./prepro_data'
# train_dataset, valid_dataset, test_dataset = sider_datasets
# get_feature(train_dataset, save_path, dir_data, dataname='train_data')
# get_feature(valid_dataset, save_path, dir_data, dataname='valid_data')
# get_feature(test_dataset, save_path, dir_data, dataname='test_data')

#----tox21----
# delaney_tasks, tox21_datasets, transformers = dc.molnet.load_tox21(featurizer='ECFP', splitter='random') 
# dir_data = 'ECFP_random_tox21_3D_attr'
# save_path = f'./prepro_data'
# train_dataset, valid_dataset, test_dataset = tox21_datasets
# get_feature(train_dataset, save_path, dir_data, dataname='train_data')
# get_feature(valid_dataset, save_path, dir_data, dataname='valid_data')
# get_feature(test_dataset, save_path, dir_data, dataname='test_data')

#=========toxcast======
# ssl._create_default_https_context = ssl._create_unverified_context
# delaney_tasks, toxcast_datasets, transformers = dc.molnet.load_toxcast(featurizer='ECFP', splitter='random') 
# dir_data = 'ECFP_random_toxcast_3D_attr'
# save_path = f'./prepro_data'
# train_dataset, valid_dataset, test_dataset = toxcast_datasets
# get_feature(train_dataset, save_path, dir_data, dataname='train_data')
# get_feature(valid_dataset, save_path, dir_data, dataname='valid_data')
# get_feature(test_dataset, save_path, dir_data, dataname='test_data')

#=====bace====
# ssl._create_default_https_context = ssl._create_unverified_context
"""delaney_tasks, bace_datasets, transformers = dc.molnet.load_bace_classification(featurizer='ECFP', splitter='random') 
dir_data = 'ECFP_random_bace_3D_attr'
save_path = f'./prepro_data'
train_dataset, valid_dataset, test_dataset = bace_datasets
get_feature(train_dataset, save_path, dir_data, dataname='train_data')
get_feature(valid_dataset, save_path, dir_data, dataname='valid_data')
get_feature(test_dataset, save_path, dir_data, dataname='test_data')"""

#===esol=======
# ssl._create_default_https_context = ssl._create_unverified_context
# delaney_tasks, esol_datasets, transformers = dc.molnet.load_delaney(featurizer='ECFP', splitter='random') 
# dir_data = 'ECFP_random_esol_3D_attr'
# save_path = f'./prepro_data'
# train_dataset, valid_dataset, test_dataset = esol_datasets
# get_feature(train_dataset, save_path, dir_data, dataname='train_data')
# get_feature(valid_dataset, save_path, dir_data, dataname='valid_data')
# get_feature(test_dataset, save_path, dir_data, dataname='test_data')

#=====LIT-PCBA=====
# data = 'IDH1'
# read_path = f'./oringin_data/LIT-PCBA/{data}'
# type = 'random'
# files = os.listdir(read_path)
# for f in files:
#     if f == 'train.csv':
#         print('train已经结束')
#     else:
#         name = f.split('.')[0]  # train
#         fi_path = os.path.join(read_path, f)  # './oringin_data/LIT-PCBA/IDH1/train.csv'
#         dir_data = f'ECFP_{type}_{data}_3D_attr/{name}'    #'ECFP_random_IDH1_3D_attr/train'
#         save_path = f'./prepro_data/LIT-PCBA/{data}'
#         with open(fi_path, 'r') as fe:
#             reader = csv.reader(fe)
#             read_fe = next(reader)
#             get_csv_feature(save_path, dir_data, reader)
        

#=======BreastCellLines======
# def process_smi(fi_path):
#     with open(fi_path, 'r') as fe:
#         name = fi.split('.')[0]
#         reader = list(csv.reader(fe))[1:]
#         train, valid, test = split(type, reader)
#         dir_data = f'{name}_3D_attr'     # 'ECFP_random_BT-20_3D_attr'
#         save_path = f'./prepro_data/BreastCellLines'
#         get_breast_feature(save_path, dir_data, train, na='train')
#         get_breast_feature(save_path, dir_data, valid, na='valid')
#         get_breast_feature(save_path, dir_data, test, na='test')


# read_path = f'./oringin_data/BreastCellLines'
# type = 'random'
# files = os.listdir(read_path)
# fis = []
# for fi in files:
#     if fi == 'MDA-MB-435.csv':
#         fi_path = os.path.join(read_path, fi)  
#         process_smi(fi_path)
#     else:
#         print('ending')

        # fis.append((fi_path))


# pool = mlc.SuperPool(64)
# pool.pool.restart()
# _ = pool.map(process_smi,fis)
# pool.exit()   

# ===================TDC============
# from tdc.single_pred import ADME
# from tdc.single_pred import Tox
# name = 'Caco2_Wang'
# # name = 'HIA_Hou'
# # name = 'Pgp_Broccatelli'
# # name = 'Bioavailability_Ma'
# # name = 'Solubility_AqSolDB'
# # name = 'PPBR_AZ'
# # name = 'VDss_Lombardo'

# # name = 'CYP2D6_Veith'
# # name = 'CYP3A4_Veith'
# # name = 'CYP2C9_Veith'
# # name = 'CYP2C9_Substrate_CarbonMangels'
# # name = 'CYP2D6_Substrate_CarbonMangels'
# #name = 'CYP3A4_Substrate_CarbonMangels'

# # name = 'Half_Life_Obach'
# # name = 'Clearance_Hepatocyte_AZ'

# # name = 'LD50_Zhu'
# # name = 'hERG'
# # name = 'AMES'
# # name = 'DILI'

# data = ADME(name = name)
# # data = Tox(name = name)
# split = data.get_split()
# save_path = f'./prepro_data/tdc_molecule'
# data_list = ['train', 'valid', 'test'] 
# for i in data_list:
#     data = split[i]
#     dir_data = f'{name}_3D_attr/{i}_data' 
#     get_tdc_feature(save_path, dir_data, data)

    
  

    
