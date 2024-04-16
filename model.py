import numpy as np
import sys
import warnings
import os
import pickle
import torch
import torch.nn as nn
import timeit
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from egnn import EGNN_Sparse
from Attention import se_block, cbam_block, eca_block, ExternalAttention, SpatialAttention      #引入注意力机制
from torch_geometric.nn import GATConv, GCNConv, NNConv
from torch.autograd import Variable 
import torch
from torch_geometric.nn import Set2Set, GCNConv, GATConv, GINConv, NNConv,TransformerConv
from torch_geometric.nn.models import AttentiveFP
from torch_geometric.utils import softmax
from torch_geometric.nn import SAGEConv
from torch.nn import Sequential, Linear, ReLU, GRU
import torch.nn.functional as F
        
class GNNSCFDN(nn.Module):
    def __init__(self, device, n_iter, dp, in_dim, dim, mol_edge_in_dim, num_tasks):
        super(GNNSCFDN, self).__init__()
        self.dropout_feature = nn.Dropout(dp)
        self.device = device
        self.n_iter = n_iter
        self.relu = nn.ReLU()
        self.dim = dim
  
        self.batch_norm = nn.BatchNorm1d(66)
        self.smi_egnn = EGNN_Sparse(feats_dim=63, m_dim=63)
        mm = Sequential(Linear(mol_edge_in_dim, in_dim * in_dim))
        self.nnconv = NNConv(in_dim, in_dim, mm, aggr='mean')
        self.gru = GRU(66+dim, 66)
        self.set2set = Set2Set(66, processing_steps=3)
        self.linf = Linear(66+dim, 1)
        self.dnn1 = nn.Linear(1024, 512)
        self.dnn2 = nn.Linear(512, dim * 4)
        self.dnn3 = nn.Linear(dim * 4, dim)
        self.predict_property = nn.Linear(132 + dim, 1*num_tasks)

    def pad(self, matrices, value):
        """Pad adjacency matrices for batch processing."""
        sizes = [d.shape[0] for d in matrices]
        D = sum(sizes)
        pad_matrices = value + np.zeros((D, D))
        m = 0
        for i, d in enumerate(matrices):
            s_i = sizes[i]
            pad_matrices[m:m + s_i, m:m + s_i] = d
            m += s_i
        return torch.FloatTensor(pad_matrices).to(self.device)

    def DNN(self, x_words):
        x_words = F.relu(self.dnn1(x_words))
        x_words = F.relu(self.dnn2(x_words))
        x_words = self.dnn3(x_words)
        return x_words

    def conv1d_spatial_graph_matrix(self, fea_m, adj_m):
        adj_m_graph1 = torch.unsqueeze(adj_m, 1) # 254, 1, 254
        fea_m_graph1 = torch.unsqueeze(fea_m, -1) # 254, 66, 1
        feas = torch.mul(adj_m_graph1, fea_m_graph1) # 254, 66, 254
        features = feas.permute(2, 0, 1) # 254, 254, 66
        conv1d_1 = nn.Conv1d(features.shape[1], self.dim, 3, stride=1, padding=1).to(self.device)
        middle_feature = conv1d_1(features).permute(0, 2, 1) # 254, 66, 32
        conv1d_2 = nn.Conv1d(middle_feature.shape[1], 1, 3, stride=1, padding=1).to(self.device)
        spatial_feature = conv1d_2(middle_feature).permute(0, 2, 1).squeeze(2)   # 254, 32
        outs = torch.cat([fea_m, spatial_feature], 1) # 254, 98
        return outs

    def forward(self, pt):
        smi_fts = self.smi_egnn(pt.mol_nc_feature, edge_index=pt.edge_index, batch=pt.mol_batch) # 254, 66  # 7360,66
        smi_fts = torch.where(torch.isnan(smi_fts), torch.full_like(smi_fts, 0), smi_fts)
        smi_fts = self.batch_norm(smi_fts)  # 254, 66
        Normed_adj = self.pad(pt.atom_matrixs, 0)  # 254, 254

        h = smi_fts.unsqueeze(0)  # 1, 254, 66
        out = smi_fts
        for i in range(self.n_iter):  # conv, focus 的顺序可以互相调换
            m = F.relu(self.nnconv(out, pt.edge_index, pt.edge_attr)) # 254, 66
            dm = self.dropout_feature(m) # 254,66
            cm = self.conv1d_spatial_graph_matrix(dm, Normed_adj)  # 254, 98
            f = self.linf(cm)  # 758, 1
            f = F.leaky_relu(f)
            af = softmax(f, pt.mol_batch)
            out, h = self.gru((af * cm).unsqueeze(0), h) # 1,297,66; 1,297,66
            out = out.squeeze(0) # 254, 66

        out = self.set2set(out, pt.mol_batch)  # 16, 132

        Fringer = list(pt.fingers)
        for i in range(len(Fringer)):
            Fringer[i] = torch.unsqueeze(torch.FloatTensor(Fringer[i]), 0)
        Fringer = torch.cat(Fringer, 0).unsqueeze(0).to(self.device) # 64, 1024
        Fringer = self.DNN(Fringer).squeeze(0)  #16, 64

        y_molecules = torch.cat((out, Fringer), 1)  # 16,196
        z_molecules = self.predict_property(y_molecules) # 16, 12

        return z_molecules

