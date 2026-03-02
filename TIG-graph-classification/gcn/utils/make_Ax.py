import torch
from torch_geometric.data import Data
from utils.SVD import compute_svd
import numpy as np
import scipy as sp
import scipy.sparse as sp


def fea_to_node_idx_map(data,idx):
    return idx+data.x.shape[0]


def make_Ax_individual(data, args):
    non_zero_indices = torch.nonzero(data.x)
    Ax_edge_index = torch.zeros(2, non_zero_indices.shape[0], dtype=torch.int64).cuda()
    Ax_edge_index[0] = non_zero_indices[:, 0]
    Ax_edge_index[1] = fea_to_node_idx_map(data, non_zero_indices[:, 1])

    edge_index = Ax_edge_index
    edge_index = torch.cat((edge_index,edge_index[[1,0]]),dim=1)


    Ax_edge_weight = data.x[non_zero_indices[:, 0], non_zero_indices[:, 1]]
    edge_weight = torch.cat((Ax_edge_weight, Ax_edge_weight), dim=0)



    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    adj_matrix = sp.coo_matrix((edge_weight.cpu().numpy(),(edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy())),shape=(data.x.shape[0]+data.x.shape[1], data.x.shape[0]+data.x.shape[1]))

    adj_matrix = adj_matrix @ adj_matrix
    adj_matrix = adj_matrix[data.x.shape[0]:, data.x.shape[0]:]

    return adj_matrix



