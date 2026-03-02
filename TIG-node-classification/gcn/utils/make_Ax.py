import torch
from torch_geometric.data import Data
from utils.SVD import compute_svd


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
    Ax_x = compute_svd(edge_index , edge_weight , data.x.shape[0]+data.x.shape[1] , data.x.shape[0] , args).cuda()
    edge_index = edge_index[:,int(edge_index.size(1)/2):]
    edge_weight = edge_weight[int(edge_index.size(1)/2):]
    x_original =  torch.zeros(data.x.shape[0], Ax_x.shape[1], dtype=torch.float32).cuda()
    x = torch.cat((x_original,Ax_x),dim=0)
    new_data = Data(x=x, edge_index=edge_index,edge_weight=edge_weight).cuda()
    return new_data



