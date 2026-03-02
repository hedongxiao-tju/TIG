import numpy as np
import scipy as sp
import scipy.sparse as sp
import torch


def compute_svd(edge_index, edge_weight, num_nodes, n, args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    adj_matrix = sp.coo_matrix((edge_weight.cpu().numpy(),(edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy())),shape=(num_nodes, num_nodes))
    adj_matrix = adj_matrix @ adj_matrix
    adj_matrix = adj_matrix[n:, n:]
    row_min = adj_matrix.min(axis=1).toarray().flatten()
    row_max = adj_matrix.max(axis=1).toarray().flatten()
    row_range = row_max - row_min
    row_range[row_range == 0] = 1
    adj_matrix = (adj_matrix - row_min[:, np.newaxis]) / row_range[:, np.newaxis]
    U, S, Vt = sp.linalg.svds(adj_matrix, k=args.svd_k, random_state=args.seed)
    U = torch.tensor(U.copy(), dtype=torch.float32).cuda()
    S = torch.tensor(S.copy(), dtype=torch.float32).cuda()
    Vt = torch.tensor(Vt.copy(), dtype=torch.float32).cuda()
    sqrt_Lambda_S = torch.sqrt(torch.diag(S))
    result = U @ sqrt_Lambda_S + Vt.T @ sqrt_Lambda_S
    return result

